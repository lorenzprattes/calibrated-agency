from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from calibrated_agency.backends.transformers_native import FirstTokenBayesianLogitsProcessor
from calibrated_agency.backends.vllm_native import (
    VLLM_EXTRA_KEY,
    _RequestState,
    make_vllm_extra_args,
)


@dataclass
class FakeSamplingParams:
    extra_args: dict[str, Any]
    temperature: float = 0.7


def test_vllm_extra_args_normalize_probabilities_and_select_activation() -> None:
    extra = make_vllm_extra_args(
        action_token_ids={"post": 10, "comment": 11},
        prior={"post": 6, "comment": 4},
        target={"post": 2, "comment": 8},
        beta=0.15,
        apply_at_first_token=True,
    )
    config = extra[VLLM_EXTRA_KEY]
    assert config["log_prior"]["post"] == pytest.approx(-0.5108256238)
    assert config["log_target"]["post"] == pytest.approx(-1.6094379124)
    assert config["compensate_temperature"] is True


def test_vllm_extra_args_require_one_activation_mode() -> None:
    base = dict(
        action_token_ids={"a": 1, "b": 2},
        prior={"a": 1, "b": 1},
        target={"a": 1, "b": 1},
        beta=1.0,
    )
    with pytest.raises(ValueError, match="exactly one"):
        make_vllm_extra_args(**base)
    with pytest.raises(ValueError, match="exactly one"):
        make_vllm_extra_args(
            **base, apply_at_first_token=True, trigger_token_ids=[4, 5]
        )


def test_vllm_first_token_state_uses_live_output_reference() -> None:
    output: list[int] = []
    extra = make_vllm_extra_args(
        action_token_ids={"a": 1, "b": 2},
        prior={"a": 1, "b": 1},
        target={"a": 1, "b": 1},
        beta=1.0,
        apply_at_first_token=True,
    )
    state = _RequestState.from_sampling_params(FakeSamplingParams(extra), [8, 9], output)
    assert state is not None and state.should_apply()
    output.append(1)
    assert not state.should_apply()


def test_vllm_trigger_can_end_in_prompt_or_output() -> None:
    extra = make_vllm_extra_args(
        action_token_ids={"a": 1, "b": 2},
        prior={"a": 1, "b": 1},
        target={"a": 1, "b": 1},
        beta=1.0,
        trigger_token_ids=[8, 9],
    )
    prompt_state = _RequestState.from_sampling_params(
        FakeSamplingParams(extra), [3, 8, 9], []
    )
    output_state = _RequestState.from_sampling_params(
        FakeSamplingParams(extra), [3], [7, 8, 9]
    )
    assert prompt_state is not None and prompt_state.should_apply()
    assert output_state is not None and output_state.should_apply()


def test_transformers_processor_validates_configuration_without_optional_runtime() -> None:
    processor = FirstTokenBayesianLogitsProcessor(
        prompt_length=5,
        action_token_ids={"a": 1, "b": 2},
        prior={"a": 0.5, "b": 0.5},
        target={"a": 0.2, "b": 0.8},
        beta=0.15,
        sampling_temperature=0.7,
    )
    assert processor.sampling_temperature == 0.7
    with pytest.raises(ValueError, match="distinct"):
        FirstTokenBayesianLogitsProcessor(
            prompt_length=5,
            action_token_ids={"a": 1, "b": 1},
            prior={"a": 0.5, "b": 0.5},
            target={"a": 0.2, "b": 0.8},
            beta=0.15,
        )
