"""Verify cached inputs, exact tokenization, and native correction on CPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

from .common import CASES, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    from calibrated_agency import bayesian_correct
    from calibrated_agency.backends.tokenization import resolve_action_tokens
    from calibrated_agency.backends.transformers_native import FirstTokenBayesianLogitsProcessor
    from calibrated_agency.backends.vllm_native import (
        BayesianActionLogitsProcessor,
        make_vllm_extra_args,
    )
    from scripts.live_backend import requests

    plan = read_json(args.root / "plan.json")
    snapshot = Path(
        snapshot_download(plan["model"], revision=plan["model_revision"], local_files_only=True)
    )
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    for name in set(index["weight_map"].values()):
        if not (snapshot / name).is_file() or (snapshot / name).stat().st_size == 0:
            raise ValueError(f"missing model shard {name}")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    cases = {}
    for case, labels in CASES.items():
        sample_requests = requests(labels)
        tokens = resolve_action_tokens(
            sample_requests[0].action_space,
            lambda label: tokenizer.encode(label, add_special_tokens=False),
            lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
        )
        lengths = []
        for request in sample_requests:
            messages = [message.as_dict() for message in request.messages]
            messages.append({"role": "assistant", "content": request.assistant_prefix})
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=False,
                continue_final_message=True,
                enable_thinking=False,
            )
            if not ids or len(ids) + 512 > 4096:
                raise ValueError("fixture exceeds generation envelope")
            lengths.append(len(ids))
        cases[case] = {
            "tokens": tokens,
            "prompt_lengths": lengths,
            "contexts": len(sample_requests),
        }

    maximum_error = 0.0
    prior, target = {"a": 0.6, "b": 0.4}, {"a": 0.2, "b": 0.8}
    for beta in (0, 0.15, 0.5, 1):
        for temperature in (0.7, 1):
            raw = {"a": 2.0, "b": -1.0}
            expected = bayesian_correct(raw, prior, target, beta).probabilities
            processor = FirstTokenBayesianLogitsProcessor(
                prompt_length=1,
                action_token_ids={"a": 0, "b": 1},
                prior=prior,
                target=target,
                beta=beta,
                sampling_temperature=temperature,
            )
            result = processor(torch.tensor([[9]]), torch.tensor([[2.0, -1.0, 8.0]]))
            probabilities = torch.softmax(result[0] / temperature, dim=-1)
            maximum_error = max(
                maximum_error,
                *(abs(float(probabilities[i]) - expected[a]) for i, a in enumerate(raw)),
            )
            native = BayesianActionLogitsProcessor(None, torch.device("cpu"), False)
            extra = make_vllm_extra_args(
                action_token_ids={"a": 0, "b": 1},
                prior=prior,
                target=target,
                beta=beta,
                apply_at_first_token=True,
            )
            native.update_state(
                SimpleNamespace(
                    removed=[],
                    moved=[],
                    added=[(0, SamplingParams(temperature=temperature, extra_args=extra), [9], [])],
                )
            )
            result = native.apply(torch.tensor([[2.0, -1.0, 8.0]]))
            probabilities = torch.softmax(result[0] / temperature, dim=-1)
            maximum_error = max(
                maximum_error,
                *(abs(float(probabilities[i]) - expected[a]) for i, a in enumerate(raw)),
            )
    if maximum_error > 1e-6:
        raise AssertionError(f"native CPU correction mismatch: {maximum_error}")
    report = {
        "passed": True,
        "device": "cpu",
        "model_snapshot": str(snapshot),
        "cases": cases,
        "native_max_probability_error": maximum_error,
        "chat_template_sha256": hashlib.sha256(
            json.dumps(tokenizer.chat_template, sort_keys=True).encode()
        ).hexdigest(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "vllm", "calibrated-agency")
        },
    }
    write_json(args.root / "preflight.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
