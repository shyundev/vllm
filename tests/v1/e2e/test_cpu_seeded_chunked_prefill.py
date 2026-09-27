# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU seeded sampling must not depend on how the prompt prefill is chunked."""

import os

import pytest

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

pytestmark = pytest.mark.cpu_model

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")

MODEL = "facebook/opt-125m"
CHUNK_TOKENS = 64
# Each prompt spans several CHUNK_TOKENS-sized chunks.
PROMPTS = [
    "The history of the city begins " * 30,
    "Solve the following arithmetic step by step. " * 20,
]


def _generate(**overrides) -> list[list[int]]:
    llm = LLM(
        model=MODEL,
        dtype="float32",
        max_model_len=1024,
        enforce_eager=True,
        **overrides,
    )
    params = [
        SamplingParams(temperature=1.0, seed=seed, max_tokens=8)
        for seed in range(len(PROMPTS))
    ]
    outputs = llm.generate(PROMPTS, params)
    del llm
    return [list(o.outputs[0].token_ids) for o in outputs]


def test_seeded_sampling_matches_unchunked_prefill(monkeypatch):
    """Samples discarded during a partial prefill must not advance the seed."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    expected = _generate()
    actual = _generate(enable_chunked_prefill=True, max_num_batched_tokens=CHUNK_TOKENS)
    assert actual == expected
