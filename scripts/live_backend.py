#!/usr/bin/env python3
"""Run one live backend on a synthetic, redistribution-safe fixture."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

from calibrated_agency import (
    ActionSpace,
    ChatMessage,
    ScoreRequest,
    bayesian_correct,
    probability_mean_prior,
)

CONTEXTS = (
    "A new topic has not been discussed yet and you have a distinct argument to introduce.",
    "Another participant directly asked you a specific question in an existing discussion.",
    "The feed contains a broad announcement and no comment is directed at you.",
    "A participant made a factual claim you can correct with a short reply.",
    "Several discussions repeat familiar points, while you have a new question for everyone.",
    "An existing thread contains a nuanced argument that merits a direct response.",
    "The feed is quiet and you have a self-contained observation to share.",
    "A reply to your earlier message asks for clarification.",
)


def requests() -> list[ScoreRequest]:
    action_space = ActionSpace.from_mapping({"post": "post", "comment": "comment"})
    system = ChatMessage(
        "system",
        "Choose one forum action. The valid XML action is "
        "<action>post</action> or <action>comment</action>.",
    )
    return [
        ScoreRequest(
            messages=(system, ChatMessage("user", context)),
            action_space=action_space,
            assistant_prefix="<action>",
            seed=20260805 + index,
        )
        for index, context in enumerate(CONTEXTS)
    ]


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def base_environment(mode: str, model: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "model": model,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in ("calibrated-agency", "vllm", "transformers", "torch")
        },
    }


def serialize_score(
    identifier: int, score: Any, request: ScoreRequest
) -> dict[str, Any]:
    return {
        "context_id": identifier,
        "request": {
            "messages": [message.as_dict() for message in request.messages],
            "assistant_prefix": request.assistant_prefix,
            "actions": [
                {"action": candidate.action, "label": candidate.label}
                for candidate in request.action_space.candidates
            ],
            "seed": request.seed,
        },
        "raw_scores": dict(score.raw_scores),
        "probabilities": dict(score.probabilities),
        "token_ids": dict(score.token_ids),
        "selected_action": score.selected_action,
        "metadata": dict(score.metadata),
        "raw_response": (
            None if score.raw_response is None else dict(score.raw_response)
        ),
    }


def load_backend(mode: str, model_id: str, base_url: str | None) -> Any:
    chat_kwargs = {"enable_thinking": False}
    if mode == "api":
        if base_url is None:
            raise ValueError("--base-url is required in API mode")
        from calibrated_agency.backends.vllm_api import VllmApiBackend

        return VllmApiBackend(
            base_url,
            model=model_id,
            logprobs_mode="raw_logprobs",
            chat_template_kwargs=chat_kwargs,
        )
    if mode == "vllm-native":
        from vllm import LLM

        from calibrated_agency.backends.vllm_native import VllmNativeBackend

        llm = LLM(
            model=model_id,
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.88,
            enforce_eager=True,
            logits_processors=[
                "calibrated_agency.backends.vllm_native:BayesianActionLogitsProcessor"
            ],
            logprobs_mode="processed_logprobs",
        )
        return VllmNativeBackend(
            llm,
            model_id=model_id,
            logprobs_mode="processed_logprobs",
            chat_template_kwargs=chat_kwargs,
        )
    if mode == "transformers":
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForMultimodalLM,
            AutoTokenizer,
        )

        from calibrated_agency.backends.transformers_native import TransformersNativeBackend

        config = AutoConfig.from_pretrained(model_id)
        model_class = (
            AutoModelForMultimodalLM
            if config.model_type == "qwen3_5"
            else AutoModelForCausalLM
        )
        model = model_class.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
        ).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return TransformersNativeBackend(
            model,
            tokenizer,
            model_id=model_id,
            chat_template_kwargs=chat_kwargs,
        )
    raise ValueError(f"unknown mode {mode}")


def run(mode: str, model_id: str, base_url: str | None) -> dict[str, Any]:
    backend = load_backend(mode, model_id, base_url)
    score_requests = requests()
    scores = [backend.score(request) for request in score_requests]
    batch_scores = (
        backend.score_batch(score_requests) if mode == "vllm-native" else []
    )
    prior = probability_mean_prior([score.probabilities for score in scores])
    target = {"post": 0.2, "comment": 0.8}
    payload = base_environment(mode, model_id)
    payload.update(
        {
            "score_mode": "raw_logprobs" if mode == "api" else (
                "processed_logprobs" if mode == "vllm-native" else "raw_logits"
            ),
            "contexts": list(CONTEXTS),
            "scores": [
                serialize_score(index, score, score_requests[index])
                for index, score in enumerate(scores)
            ],
            "batch_scores": [
                serialize_score(index, score, score_requests[index])
                for index, score in enumerate(batch_scores)
            ],
            "probability_mean_prior": prior,
            "target": target,
            "core_corrections": [
                {
                    "context_id": context_id,
                    "beta": beta,
                    "probabilities": dict(
                        bayesian_correct(score.raw_scores, prior, target, beta).probabilities
                    ),
                }
                for context_id, score in enumerate(scores)
                for beta in (0.0, 0.15, 0.5, 1.0)
            ],
            "native_decisions": [],
        }
    )

    if mode in {"vllm-native", "transformers"}:
        decisions: list[dict[str, Any]] = []
        for context_id, request in enumerate(score_requests[:3]):
            for beta in (0.0, 0.15, 0.5, 1.0):
                for temperature in (0.7, 1.0):
                    decision = backend.decide_native(
                        request,
                        prior=prior,
                        target=target,
                        beta=beta,
                        temperature=temperature,
                    )
                    if mode == "vllm-native":
                        reference_score = backend.score(request)
                        expected_probabilities = bayesian_correct(
                            reference_score.raw_scores, prior, target, beta
                        ).probabilities
                    else:
                        expected_probabilities = decision.expected_probabilities
                    decisions.append(
                        {
                            "context_id": context_id,
                            "beta": beta,
                            "temperature": temperature,
                            "action": decision.action,
                            "token_id": decision.token_id,
                            "raw_scores": (
                                None
                                if decision.raw_scores is None
                                else dict(decision.raw_scores)
                            ),
                            "expected_probabilities": dict(expected_probabilities),
                            "backend_reported_probabilities": dict(
                                decision.sampled_probabilities
                            ),
                        }
                    )
        payload["native_decisions"] = decisions
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("api", "vllm-native", "transformers"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.mode, args.model, args.base_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
