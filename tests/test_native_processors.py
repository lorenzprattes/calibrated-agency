from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest

from calibrated_agency.backends.transformers_native import (
    FirstTokenBayesianLogitsProcessor,
    _neutral_generation_config_kwargs,
)
from calibrated_agency.backends.vllm_native import (
    VLLM_EXTRA_KEY,
    BayesianActionLogitsProcessor,
    MoveDirectionality,
    _RequestState,
    make_vllm_extra_args,
)


@dataclass
class FakeSamplingParams:
    extra_args: dict[str, Any]
    temperature: float = 0.7


@dataclass
class FakeBatchUpdate:
    removed: list[int]
    added: list[tuple[int, FakeSamplingParams, list[int] | None, list[int]]]
    moved: list[tuple[int, int, Any]]


class FakeScalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class FakeRow:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def fill_(self, value: float) -> None:
        self.values[:] = [value] * len(self.values)


class FakeTensor:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = [FakeRow(row) for row in rows]

    def __getitem__(self, key: int | tuple[int, int]) -> FakeRow | FakeScalar:
        if isinstance(key, tuple):
            row, column = key
            return FakeScalar(self.rows[row].values[column])
        return self.rows[key]

    def __setitem__(self, key: tuple[int, int], value: float) -> None:
        row, column = key
        self.rows[row].values[column] = value


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
    with pytest.raises(ValueError, match="nonnegative integers"):
        make_vllm_extra_args(**base, trigger_token_ids=[-1])
    with pytest.raises(ValueError, match="nonnegative integers"):
        make_vllm_extra_args(
            **{**base, "action_token_ids": {"a": True, "b": 2}},
            apply_at_first_token=True,
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


def test_vllm_processor_rejects_nonfinite_runtime_configuration() -> None:
    extra = make_vllm_extra_args(
        action_token_ids={"a": 1, "b": 2},
        prior={"a": 1, "b": 1},
        target={"a": 1, "b": 1},
        beta=1.0,
        apply_at_first_token=True,
    )
    extra[VLLM_EXTRA_KEY]["beta"] = float("nan")
    with pytest.raises(ValueError, match="beta"):
        BayesianActionLogitsProcessor.validate_params(FakeSamplingParams(extra))


def test_vllm_trigger_requires_prompt_ids_when_output_is_short() -> None:
    extra = make_vllm_extra_args(
        action_token_ids={"a": 1, "b": 2},
        prior={"a": 1, "b": 1},
        target={"a": 1, "b": 1},
        beta=1.0,
        trigger_token_ids=[8, 9],
    )
    state = _RequestState.from_sampling_params(FakeSamplingParams(extra), None, [])
    assert state is not None
    with pytest.raises(RuntimeError, match="prompt token"):
        state.should_apply()


def test_vllm_batch_processor_masks_scales_and_moves_rows() -> None:
    first_extra = make_vllm_extra_args(
        action_token_ids={"a": 1, "b": 2},
        prior={"a": 0.5, "b": 0.5},
        target={"a": 0.2, "b": 0.8},
        beta=0.0,
        apply_at_first_token=True,
    )
    second_extra = make_vllm_extra_args(
        action_token_ids={"a": 0, "b": 3},
        prior={"a": 0.5, "b": 0.5},
        target={"a": 0.5, "b": 0.5},
        beta=1.0,
        apply_at_first_token=True,
    )
    processor = BayesianActionLogitsProcessor(None, None, False)
    processor.update_state(
        FakeBatchUpdate(
            removed=[],
            added=[
                (0, FakeSamplingParams(first_extra, 0.7), [5], []),
                (1, FakeSamplingParams(second_extra, 1.0), [6], []),
            ],
            moved=[],
        )
    )
    logits = FakeTensor([[9.0, 4.0, 3.0, 2.0], [2.0, 8.0, 7.0, -1.0]])
    returned = processor.apply(logits)  # type: ignore[arg-type]
    assert returned is logits
    assert logits.rows[0].values[0] == -math.inf
    assert logits.rows[0].values[1] == pytest.approx(0.7 * math.log(0.2))
    assert logits.rows[0].values[2] == pytest.approx(0.7 * math.log(0.8))
    assert logits.rows[0].values[3] == -math.inf
    assert logits.rows[1].values == pytest.approx([2.0, -math.inf, -math.inf, -1.0])

    processor.update_state(
        FakeBatchUpdate(
            removed=[], added=[], moved=[(0, 1, MoveDirectionality.SWAP)]
        )
    )
    assert processor.requests[0].action_token_ids == {"a": 0, "b": 3}
    assert processor.requests[1].action_token_ids == {"a": 1, "b": 2}
    processor.update_state(
        FakeBatchUpdate(
            removed=[], added=[], moved=[(1, 2, MoveDirectionality.SWAP)]
        )
    )
    assert 1 not in processor.requests
    assert 2 in processor.requests
    processor.update_state(FakeBatchUpdate(removed=[2], added=[], moved=[]))
    assert 2 not in processor.requests


def test_vllm_processor_greedy_temperature_does_not_zero_corrected_logits() -> None:
    extra = make_vllm_extra_args(
        action_token_ids={"a": 0, "b": 1},
        prior={"a": 0.5, "b": 0.5},
        target={"a": 0.25, "b": 0.75},
        beta=0.0,
        apply_at_first_token=True,
    )
    processor = BayesianActionLogitsProcessor(None, None, False)
    processor.update_state(
        FakeBatchUpdate(
            removed=[],
            added=[(0, FakeSamplingParams(extra, 0.0), [5], [])],
            moved=[],
        )
    )
    logits = FakeTensor([[2.0, 1.0]])
    processor.apply(logits)  # type: ignore[arg-type]
    assert logits.rows[0].values == pytest.approx([math.log(0.25), math.log(0.75)])


def test_transformers_action_config_neutralizes_inherited_transforms() -> None:
    config = _neutral_generation_config_kwargs(
        eos_token_id=[8, 9], bos_token_id=1, temperature=0.7
    )
    assert config["temperature"] == 0.7
    assert config["top_k"] is None and config["top_p"] is None
    assert config["repetition_penalty"] == 1.0
    assert config["sequence_bias"] is None
    assert config["bad_words_ids"] is None
    assert config["forced_eos_token_id"] is None
    assert config["eos_token_id"] == [8, 9]


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
