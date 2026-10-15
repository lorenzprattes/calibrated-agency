"""Focused checks for multiclass comparison and progress-aware health decisions."""

from scripts.compare_live import compare_pair
from scripts.validation_study.common import health_failure


def test_study_comparison_supports_multiclass_and_score_shifts() -> None:
    first = {
        "scores": [
            {
                "context_id": 0,
                "raw_scores": {"a": 2.0, "b": -1.0, "c": 0.0},
                "token_ids": {"a": 1, "b": 2, "c": 3},
            }
        ],
        "probability_mean_prior": {"a": 0.2, "b": 0.3, "c": 0.5},
        "target": {"a": 0.4, "b": 0.4, "c": 0.2},
    }
    second = {
        "scores": [
            {
                "context_id": 0,
                "raw_scores": {"c": 100.0, "b": 99.0, "a": 102.0},
                "token_ids": {"a": 1, "b": 2, "c": 3},
            }
        ]
    }
    result = compare_pair(first, second, name="api_vs_vllm_native")
    assert result["passed"]
    assert result["max_absolute_log_odds_error"] == 0


def test_one_idle_sample_or_progressing_cpu_phase_is_not_a_stall() -> None:
    assert health_failure(phase="active", elapsed=100, progress_age=2, idle_for=100) is None
    assert health_failure(phase="active", elapsed=100, progress_age=100, idle_for=2) is None
    assert health_failure(phase="model_load", elapsed=200, progress_age=200, idle_for=200) is None


def test_sustained_idle_without_progress_and_hung_compute_are_stopped() -> None:
    assert (
        health_failure(phase="active", elapsed=500, progress_age=130, idle_for=130)
        == "idle_without_progress"
    )
    assert (
        health_failure(phase="active", elapsed=500, progress_age=181, idle_for=0) == "no_progress"
    )
    assert (
        health_failure(phase="model_load", elapsed=601, progress_age=601, idle_for=601)
        == "startup_timeout"
    )


def test_completed_array_is_read_from_accounting_when_absent_from_queue(
    monkeypatch, tmp_path, capsys
):
    import getpass
    import json
    import subprocess
    import sys

    from scripts.validation_study import remote_status

    def fake_run(command, **kwargs):
        if command[0] == "squeue":
            return subprocess.CompletedProcess(
                command, 1, "", "slurm_load_jobs error: Invalid job id specified"
            )
        rows = "".join(f"123_{i}|COMPLETED|{getpass.getuser()}|00:01:00|0:0\n" for i in range(3))
        return subprocess.CompletedProcess(command, 0, rows, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["remote_status", "--root", str(tmp_path), "--job-id", "123"])
    remote_status.main()
    roles = json.loads(capsys.readouterr().out)["roles"]
    assert len(roles) == 3
    assert all(role["state"] == "COMPLETED" and role["ownership_verified"] for role in roles)


def test_probe_failure_still_launches_due_audit(monkeypatch, tmp_path):
    import json
    import subprocess
    import sys

    from scripts.validation_study import monitor

    (tmp_path / "run.json").write_text(json.dumps({"job_id": "123", "remote_root": "/example"}))
    (tmp_path / "monitor-state.json").write_text(json.dumps({"next_audit": 0}))
    launches = []

    def failed_probe(command):
        raise subprocess.CalledProcessError(1, command, output="scheduler unavailable")

    def audit(root, state, terminal):
        launches.append(terminal)
        return True

    monkeypatch.setattr(monitor, "run", failed_probe)
    monkeypatch.setattr(monitor, "start_audit", audit)
    monkeypatch.setattr(sys, "argv", ["monitor", "--root", str(tmp_path), "--force"])
    monitor.main()
    state = json.loads((tmp_path / "monitor-state.json").read_text())
    assert launches == [False]
    assert "scheduler unavailable" in state["probe_error"]
    assert state["next_audit"] > 0


def test_diagnostic_uses_log_odds_and_rejects_mismatched_actions():
    import pytest

    from scripts.validation_study.summarize_diagnostic import log_odds_error

    assert log_odds_error({"a": 2.0, "b": 1.0}, {"b": 101.0, "a": 102.0}) == 0
    with pytest.raises(ValueError, match="identical action sets"):
        log_odds_error({"a": 1.0, "b": 2.0}, {"a": 1.0, "c": 2.0})
