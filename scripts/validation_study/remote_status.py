"""Read only this study's scheduler entries and progress files on the login node."""

from __future__ import annotations

import argparse
import getpass
import json
import subprocess
import time
from pathlib import Path

from .common import MODES, read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    if not args.job_id.isdigit():
        raise ValueError("invalid scheduler job ID")
    queue = subprocess.run(
        ["squeue", "-h", "-r", "-j", args.job_id, "-o", "%i|%T|%u|%S|%R"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    # Completed arrays may already have disappeared from the live queue.
    if queue.returncode and not (
        queue.returncode == 1 and "Invalid job id specified" in queue.stderr
    ):
        raise subprocess.CalledProcessError(
            queue.returncode, queue.args, output=queue.stdout, stderr=queue.stderr
        )
    accounting = subprocess.run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            args.job_id,
            "--format=JobID,State,User,Elapsed,ExitCode",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    entries = {}
    for line in accounting.stdout.splitlines():
        columns = line.split("|")
        if len(columns) >= 5:
            entries[columns[0]] = {
                "state": columns[1].split()[0].rstrip("+"),
                "owner": columns[2],
                "elapsed": columns[3],
                "exit_code": columns[4],
            }
    for line in queue.stdout.splitlines():
        columns = line.split("|", 4)
        if len(columns) == 5:
            entries[columns[0]] = {
                "state": columns[1],
                "owner": columns[2],
                "start": columns[3],
                "reason": columns[4],
            }
    roles = []
    for index, mode in enumerate(MODES):
        job = f"{args.job_id}_{index}"
        entry = entries.get(job, {"state": "UNKNOWN"})
        path = args.root / "results" / mode / "health.json"
        roles.append(
            {
                "mode": mode,
                "job_id": job,
                **entry,
                "ownership_verified": entry.get("owner") == getpass.getuser(),
                "health": read_json(path) if path.exists() else None,
            }
        )
    print(json.dumps({"updated": time.time(), "roles": roles}))


if __name__ == "__main__":
    main()
