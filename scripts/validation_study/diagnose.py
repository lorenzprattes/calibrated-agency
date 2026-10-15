"""Trace identical synthetic prompts through adapters and direct runtime calls.

Run each mode in a separate process so its CUDA allocation is released on exit.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import time
from pathlib import Path

from scripts.live_backend import base_environment, load_backend, requests
from scripts.validation_study.common import CASES, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["transformers", "vllm-native"], required=True)
    args = parser.parse_args()
    plan = read_json(args.root / "plan.json")
    output = args.root / "diagnostic"
    output.mkdir(exist_ok=True)

    def progress(step: str) -> None:
        write_json(
            output / "progress.json", {"updated": time.time(), "step": f"{args.mode}:{step}"}
        )

    progress("loading")
    if args.mode == "transformers":
        backend = load_backend(args.mode, plan["model"], None, revision=plan["model_revision"])
    else:
        from huggingface_hub import snapshot_download
        from vllm import LLM

        from calibrated_agency.backends.vllm_native import VllmNativeBackend

        snapshot = snapshot_download(
            plan["model"], revision=plan["model_revision"], local_files_only=True
        )
        llm = LLM(
            model=snapshot,
            tokenizer=snapshot,
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.35,
            enforce_eager=True,
            enable_prefix_caching=False,
            generation_config="vllm",
            logprobs_mode="processed_logits",
            logits_processors=[
                "calibrated_agency.backends.vllm_native:BayesianActionLogitsProcessor"
            ],
        )
        backend = VllmNativeBackend(
            llm,
            model_id=plan["model"],
            logprobs_mode="processed_logits",
            chat_template_kwargs={"enable_thinking": False},
        )
    result = base_environment(args.mode, plan["model"])
    result.update(model_revision=plan["model_revision"], rows=[])
    if args.mode == "transformers":
        import torch

        model = backend.model
        result["settings"] = {
            "model_class": type(model).__name__,
            "dtype": str(model.dtype),
            "attention": model.config.text_config._attn_implementation,
            "use_cache": model.config.text_config.use_cache,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        }
        source = inspect.getsource(type(model).forward)
        result["forward_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        head = model.get_output_embeddings()
        captured = {}

        def capture(module, inputs):
            captured["hidden"] = inputs[0][:, -1:, :].detach().clone()

        handle = head.register_forward_pre_hook(capture)
    for case, labels in CASES.items():
        case_requests = requests(labels)
        original = read_json(args.root / "results/api" / f"{case}.json")
        batch = backend.score_batch(case_requests) if args.mode == "vllm-native" else None
        for index, request in enumerate(case_requests):
            progress(f"{case}:{index}")
            ids, tokens = backend.resolve_token_ids(request)
            row = {
                "case": case,
                "context_id": index,
                "prompt_token_ids": ids,
                "token_ids": tokens,
                "api_prompt_ids_equal": ids
                == original["scores"][index]["metadata"]["prompt_token_ids"],
                "api_action_ids_equal": tokens == original["scores"][index]["token_ids"],
                "adapter": dict(backend.score(request).raw_scores),
                "repeat": dict(backend.score(request).raw_scores),
            }
            if args.mode == "transformers":
                input_ids = backend._input_tensor(ids)

                def forward(input_ids=input_ids, tokens=tokens, **kwargs):
                    with torch.inference_mode():
                        out = model(
                            input_ids=input_ids, attention_mask=torch.ones_like(input_ids), **kwargs
                        )
                    return {a: float(out.logits[0, -1, t].item()) for a, t in tokens.items()}

                row["direct"] = forward()
                hidden = captured["hidden"]
                # Same hidden state, same candidate weights: isolate output rounding.
                with torch.inference_mode():
                    weights = head.weight[list(tokens.values())].float()
                    projected = torch.nn.functional.linear(hidden.float(), weights).flatten()
                row["fp32_head"] = {a: float(projected[i].item()) for i, a in enumerate(tokens)}
                row["fp32_head_rounded_bf16"] = {
                    a: float(projected[i].bfloat16().item()) for i, a in enumerate(tokens)
                }
                row["last_token_only"] = forward(logits_to_keep=1)
                row["no_cache"] = forward(use_cache=False)
                previous = model.config.text_config._attn_implementation
                model.set_attn_implementation("eager")
                row["eager_attention"] = forward()
                model.set_attn_implementation(previous)
            else:
                from vllm import SamplingParams

                params = SamplingParams(
                    temperature=1.0,
                    max_tokens=1,
                    seed=request.seed,
                    logprobs=len(tokens),
                    logprob_token_ids=list(tokens.values()),
                )
                raw = backend.llm.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[0]
                row["runtime_prompt_ids_equal"] = list(raw.prompt_token_ids) == ids
                row["direct"] = backend._extract_logprobs(raw.outputs[0], tokens)
                row["batch"] = dict(batch[index].raw_scores)
            result["rows"].append(row)
            write_json(output / f"{args.mode}.json", result)
    if args.mode == "transformers":
        handle.remove()
    progress("complete")


if __name__ == "__main__":
    main()
