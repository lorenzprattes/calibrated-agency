"""Persist scheduler checks and launch bounded, recurring agent audits."""

from __future__ import annotations

import argparse
import fcntl
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .common import TERMINAL, read_json, write_json


def run(command: list[str], timeout: int = 90) -> str:
    return subprocess.check_output(command, text=True, timeout=timeout, stderr=subprocess.STDOUT)


def start_audit(root: Path, state: dict[str, Any], terminal: bool) -> bool:
    previous = state.get("audit_session")
    if previous:
        status = json.loads(run(["happier", "session", "status", previous, "--json"], timeout=30))
        history = json.loads(
            run(
                [
                    "happier",
                    "session",
                    "history",
                    previous,
                    "--limit",
                    "12",
                    "--format",
                    "raw",
                    "--include-meta",
                    "--include-structured-payload",
                    "--json",
                ],
                timeout=30,
            )
        )
        types = [
            message.get("raw", {}).get("content", {}).get("data", {}).get("type")
            for message in history.get("data", {}).get("messages", [])
        ]
        significant = [kind for kind in types if kind and kind != "token_count"]
        ready = bool(significant) and significant[-1] == "ready"
        active = status.get("data", {}).get("session", {}).get("active", False)
        if active and not ready:
            state["next_audit"] = time.time() + 300
            state["next_probe"] = min(state.get("next_probe", time.time() + 300), time.time() + 300)
            return False
    prompt = (
        "Audit only the calibrated-agency validation study recorded in this working directory. "
        "Read plan.json, run.json, monitor-state.json, scheduler-status.json, "
        "and available results. "
        "Use /home/lorenz/.agents/skills/experiment-operations/SKILL.md. "
        "Check progress, scheduler pending reasons or exit codes, "
        "and per-role GPU health telemetry. "
        "The monitor polls and can cancel an owned job with a confirmed dead supervisor. "
        "The in-job watchdog cleans up its child processes. "
        "Do not submit, extend, "
        "resubmit, or cancel jobs, kill other processes, edit the library, or publish anything. "
        "Diagnose idle/stalled work using phase and real progress, not one utilization sample. "
        "Write a concise audit to audits/ using UTC time in its filename. Read-only remote checks "
        "of this study are allowed if local telemetry is insufficient. "
        "If jobs remain pending or running, verify the active user timer named in run.json. "
        "If needed, write audit-next.json with a next_audit Unix timestamp and a reason. "
        "Do not edit monitor-state.json; "
        "the monitor maintains one chain and renews it automatically. Do not create another timer. "
        "If all jobs are terminal, inspect the reports and explain any failures; no further audit "
        "is needed after your final report. Summarize actionable problems in the session. "
        f"This is {'the terminal audit' if terminal else 'a scheduled progress audit'}."
    )
    payload = json.loads(
        run(
            [
                "happier",
                "session",
                "create",
                "--path",
                str(root),
                "--backend",
                "codex",
                "--permission-mode",
                "default",
                "--title",
                "Calibrated-agency experiment audit",
                "--tag",
                f"cal-agency-{read_json(root / 'run.json')['job_id']}",
                "--prompt",
                prompt,
                "--json",
            ],
            timeout=60,
        )
    )
    if payload.get("ok") is False:
        raise RuntimeError(f"agent audit creation failed: {payload}")
    state.setdefault("audit_launches", []).append({"time": time.time(), "response": payload})
    data = payload.get("data", {})
    state["audit_session"] = data.get("sessionId") or data.get("session", {}).get("id")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-agent", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / "monitor.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        record = read_json(root / "run.json")
        state_path = root / "monitor-state.json"
        state = read_json(state_path) if state_path.exists() else {}
        suggestion = root / "audit-next.json"
        if suggestion.exists():
            requested = read_json(suggestion).get("next_audit")
            if (
                isinstance(requested, (int, float))
                and time.time() + 60 <= requested <= time.time() + 21600
            ):
                state["next_audit"] = requested
            suggestion.unlink()
        if not args.force and time.time() < state.get("next_probe", 0):
            return
        command = [
            "python3",
            "-m",
            "scripts.validation_study.remote_status",
            "--root",
            record["remote_root"],
            "--job-id",
            record["job_id"],
        ]
        try:
            status = json.loads(
                run(
                    [
                        "ssh",
                        "-o",
                        "BatchMode=yes",
                        "-o",
                        "ConnectTimeout=10",
                        "gsc1",
                        f"cd {shlex.quote(record['remote_root'])} && {shlex.join(command)}",
                    ]
                )
            )
            state["last_successful_probe"] = time.time()
            state.pop("probe_error", None)
            write_json(root / "scheduler-status.json", status)
            suspects = state.setdefault("stale_supervisors", {})
            for role in status["roles"]:
                health = role["health"]
                stale = (
                    role["state"] == "RUNNING"
                    and role["ownership_verified"]
                    and health is not None
                    and status["updated"] - health["updated"] > 240
                )
                previous = suspects.get(role["job_id"])
                if not stale:
                    suspects.pop(role["job_id"], None)
                elif (
                    previous
                    and previous["health_timestamp"] == health["updated"]
                    and time.time() - previous["observed"] > 90
                ):
                    run(["ssh", "gsc1", shlex.join(["scancel", role["job_id"]])], timeout=30)
                    state.setdefault("cleanup_actions", []).append(
                        {
                            "time": time.time(),
                            "job_id": role["job_id"],
                            "reason": "stale supervisor confirmed by two running-state probes",
                        }
                    )
                    suspects.pop(role["job_id"], None)
                else:
                    suspects[role["job_id"]] = {
                        "health_timestamp": health["updated"],
                        "observed": time.time(),
                    }
            states = [role["state"] for role in status["roles"]]
            terminal = all(value in TERMINAL for value in states)
            running = any(value in {"RUNNING", "COMPLETING", "CONFIGURING"} for value in states)
            state["next_probe"] = time.time() + (120 if running else 900)
            if running:
                state["next_audit"] = min(
                    state.get("next_audit", time.time() + 600), time.time() + 600
                )
            else:
                state.setdefault("next_audit", time.time() + 7200)
            if terminal:
                run(
                    [
                        "rsync",
                        "-az",
                        "-e",
                        "ssh",
                        f"gsc1:{record['remote_root']}/results/",
                        str(root / "results") + "/",
                    ]
                )
                complete = all(value == "COMPLETED" for value in states)
                if complete:
                    for case in ("binary", "multiclass"):
                        run(
                            [
                                sys.executable,
                                "-m",
                                "scripts.compare_live",
                                "--api",
                                str(root / "results/api" / f"{case}.json"),
                                "--vllm-native",
                                str(root / "results/vllm-native" / f"{case}.json"),
                                "--transformers",
                                str(root / "results/transformers" / f"{case}.json"),
                                "--output",
                                str(root / f"summary-{case}.json"),
                                "--report",
                                str(root / f"report-{case}.md"),
                            ]
                        )
                state["terminal"] = True
                state["jobs_completed"] = complete
                if complete:
                    state["validation_passed"] = all(
                        read_json(root / f"summary-{case}.json")["passed"]
                        for case in ("binary", "multiclass")
                    )
                write_json(state_path, state)
                if not args.no_agent and not state.get("terminal_audit_launched"):
                    state["terminal_audit_launched"] = start_audit(root, state, True)
                    if not state["terminal_audit_launched"]:
                        write_json(state_path, state)
                        return
                write_json(state_path, state)
                run(["systemctl", "--user", "stop", record["timer"]], timeout=15)
                return
            if (
                not args.no_agent
                and time.time() >= state["next_audit"]
                and start_audit(root, state, False)
            ):
                state["next_audit"] = time.time() + (1800 if running else 7200)
        except (subprocess.SubprocessError, ValueError, OSError, RuntimeError) as exc:
            detail = getattr(exc, "output", None)
            state["probe_error"] = str(exc) + (f"\n{str(detail)[-4000:]}" if detail else "")
            state["next_probe"] = time.time() + 300
            # Missing telemetry must not suppress an overdue agent investigation.
            write_json(state_path, state)
            if not args.no_agent and time.time() >= state.get("next_audit", 0):
                try:
                    if start_audit(root, state, False):
                        state["next_audit"] = time.time() + 1800
                except (subprocess.SubprocessError, ValueError, OSError, RuntimeError) as audit_exc:
                    state["audit_error"] = str(audit_exc)
        write_json(state_path, state)
        print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
