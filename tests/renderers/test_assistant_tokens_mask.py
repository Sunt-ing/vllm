from typing import cast

from vllm.inputs import TokensPrompt
from vllm.renderers import TokenizeParams


class FakeTokenizer:
    pad_token_id = 0
    truncation_side = "right"


def _apply_params(
    *,
    token_ids: list[int],
    mask: list[int],
    truncate_prompt_tokens: int | None = None,
    truncation_side: str | None = None,
    pad_prompt_tokens: int | None = None,
) -> tuple[list[int], list[int]]:
    prompt = TokensPrompt(prompt_token_ids=token_ids)
    cast(dict, prompt)["_assistant_tokens_mask"] = mask

    params = TokenizeParams(
        max_total_tokens=16,
        max_output_tokens=0,
        truncate_prompt_tokens=truncate_prompt_tokens,
        truncation_side=truncation_side,  # type: ignore[arg-type]
        pad_prompt_tokens=pad_prompt_tokens,
    )
    params.apply_post_tokenization(FakeTokenizer(), prompt)
    return prompt["prompt_token_ids"], cast(dict, prompt)["_assistant_tokens_mask"]


def test_assistant_tokens_mask_follows_left_truncation():
    token_ids, mask = _apply_params(
        token_ids=[10, 11, 12, 13, 14, 15, 16, 17],
        mask=[0, 0, 0, 0, 1, 1, 0, 0],
        truncate_prompt_tokens=4,
        truncation_side="left",
    )

    assert token_ids == [14, 15, 16, 17]
    assert mask == [1, 1, 0, 0]


def test_assistant_tokens_mask_follows_right_truncation():
    token_ids, mask = _apply_params(
        token_ids=[10, 11, 12, 13, 14, 15, 16, 17],
        mask=[0, 0, 0, 0, 1, 1, 0, 0],
        truncate_prompt_tokens=4,
    )

    assert token_ids == [10, 11, 12, 13]
    assert mask == [0, 0, 0, 0]


def test_assistant_tokens_mask_padding_uses_zero():
    token_ids, mask = _apply_params(
        token_ids=[10, 11, 12],
        mask=[0, 1, 1],
        pad_prompt_tokens=5,
    )

    assert token_ids == [10, 11, 12, 0, 0]
    assert mask == [0, 1, 1, 0, 0]
