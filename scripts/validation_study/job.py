"""Supervise one backend on one Slurm GPU; stop only processes started here."""

from __future__ import annotations

import argparse
import csv
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from .common import MODES, health_failure, read_json, write_json


def gpu_sample() -> list[dict[str, str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return [
        dict(
            zip(
                ("index", "uuid", "name", "memory_mib", "utilization"),
                (item.strip() for item in row),
                strict=True,
            )
        )
        for row in csv.reader(result.stdout.splitlines())
    ]


def stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True, choices=range(3))
    args = parser.parse_args()
    root = args.root.resolve()
    plan = read_json(root / "plan.json")
    mode = MODES[args.task]
    output = root / "results" / mode
    output.mkdir(parents=True, exist_ok=True)
    started, idle_since = time.time(), None
    processes: list[subprocess.Popen[Any]] = []
    logs = []
    state: dict[str, Any] = {
        "mode": mode,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "started": started,
        "phase": "starting",
        "updated": started,
    }
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(selected) != 1 or not selected[0]:
        raise RuntimeError("expected exactly one Slurm-assigned GPU")
    rows = gpu_sample()
    physical = (os.environ.get("SLURM_JOB_GPUS") or selected[0]).split(",")[0]
    allocated = [row for row in rows if row["index"] == physical or row["uuid"] == physical]
    if len(allocated) != 1:
        raise RuntimeError("cannot bind health telemetry to the allocated GPU")
    if float(allocated[0]["memory_mib"]) > 512:
        raise RuntimeError("allocated GPU is already occupied; refusing to interfere")
    state["gpu"] = allocated[0]
    write_json(output / "health.json", state)
    env = dict(os.environ)
    for key, value in {
        "PYTHONPATH": f"{root / 'python'}:{root}",
        "HF_HUB_CACHE": plan["hf_hub_cache"],
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": selected[0],
    }.items():
        env[key] = value
        env[f"SINGULARITYENV_{key}"] = value
    base = [
        "singularity",
        "exec",
        "--nv",
        "--bind",
        f"{root}:{root}",
        "--pwd",
        str(root),
        plan["container"],
    ]

    def launch(command: list[str], filename: str) -> subprocess.Popen[Any]:
        handle = (output / filename).open("w")
        logs.append(handle)
        child = subprocess.Popen(
            base + command, stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        processes.append(child)
        return child

    def update(phase: str) -> None:
        state.update(phase=phase, updated=time.time())
        state["owned_processes"] = [
            {"pid": child.pid, "exit_code": child.poll()} for child in processes
        ]
        try:
            state["gpu"] = next(row for row in gpu_sample() if row["uuid"] == allocated[0]["uuid"])
        except (subprocess.SubprocessError, StopIteration) as exc:
            state["telemetry_error"] = str(exc)
        write_json(output / "health.json", state)
        with (output / "telemetry.jsonl").open("a") as handle:
            import json

            handle.write(json.dumps(state) + "\n")

    def interrupted(signum: int, frame: Any) -> None:
        raise RuntimeError(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        base_url = None
        server = None
        if mode == "api":
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            base_url = f"http://127.0.0.1:{port}"
            server = launch(
                [
                    "vllm",
                    "serve",
                    plan["model"],
                    "--revision",
                    plan["model_revision"],
                    "--tokenizer-revision",
                    plan["model_revision"],
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--served-model-name",
                    plan["model"],
                    "--dtype",
                    "bfloat16",
                    "--max-model-len",
                    "4096",
                    "--gpu-memory-utilization",
                    str(plan["gpu_memory_utilization"]),
                    "--enforce-eager",
                    "--generation-config",
                    "vllm",
                    "--logprobs-mode",
                    "raw_logprobs",
                ],
                "server.log",
            )
            import urllib.error
            import urllib.request

            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"server exited: {server.returncode}")
                if time.time() - started > 600:
                    raise RuntimeError("server startup exceeded ten minutes")
                update("server_start")
                try:
                    with urllib.request.urlopen(base_url + "/health", timeout=5) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(10)
        command = [
            "python3",
            "-m",
            "scripts.validation_study.worker",
            "--root",
            str(root),
            "--mode",
            mode,
        ]
        if base_url is not None:
            command += ["--base-url", base_url]
        worker_start = time.time()
        worker = launch(command, "worker.log")
        last_tick, resume_grace_until = time.time(), 0.0
        while worker.poll() is None:
            current = time.time()
            if current - last_tick > 45:
                resume_grace_until = current + 60
            last_tick = current
            if server is not None and server.poll() is not None:
                raise RuntimeError("API server died while scoring")
            progress_path = output / "progress.json"
            progress = (
                read_json(progress_path)
                if progress_path.exists()
                else {"phase": "starting", "updated": worker_start, "step": "starting"}
            )
            update(progress["phase"])
            state["progress"] = progress
            utilization = float(state["gpu"]["utilization"])
            idle_since = (idle_since or time.time()) if utilization < 1 else None
            failure = health_failure(
                phase=progress["phase"],
                elapsed=time.time() - worker_start,
                progress_age=time.time() - progress["updated"],
                idle_for=0 if idle_since is None else time.time() - idle_since,
            )
            if failure and time.time() >= resume_grace_until:
                raise RuntimeError(f"watchdog: {failure}; last step {progress['step']}")
            time.sleep(10)
        if worker.returncode != 0:
            raise RuntimeError(f"backend worker exited with status {worker.returncode}")
        for case in ("binary", "multiclass"):
            if not (output / f"{case}.json").is_file():
                raise RuntimeError(f"missing {case} result")
        state["success"] = True
    except BaseException as exc:
        state.update(success=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for process in reversed(processes):
            stop_process(process)
        for handle in logs:
            handle.close()
        state.update(phase="complete", updated=time.time())
        write_json(output / "health.json", state)


if __name__ == "__main__":
    main()
