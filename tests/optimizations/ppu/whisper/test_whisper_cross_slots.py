from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    "lens",
    [
        [1500, 0, 1500, 1500],  # Whisper: every audio is padded to 1500 frames
        [0, 0, 0],  # pure decode step: nothing to write
        [1500],
        [17, 0, 33, 16, 1],  # ragged lengths, partial last block
    ],
)
def test_batched_cross_slot_mapping_matches_vllm(lens):
    from vllm.model_executor.layers.attention import cross_attention

    from vllm_fl.optimizations.ppu.whisper.cross_slots import (
        _batched_cross_slot_mapping,
    )

    orig = cross_attention._get_cross_slot_mapping
    if getattr(orig, "__module__", "").startswith("vllm_fl"):
        pytest.skip("already patched in this process")
    torch.manual_seed(0)
    spec = SimpleNamespace(block_size=16)
    lens = np.array(lens, dtype=np.int32)
    max_blocks = 100
    table = torch.randint(
        0, 5000, (len(lens) + 2, max_blocks), dtype=torch.int32, device="cuda"
    )
    ref = orig(lens, table, spec, torch.device("cuda"))
    out = _batched_cross_slot_mapping(lens, table, spec, torch.device("cuda"))
    assert out.dtype == torch.int64 and out.shape == ref.shape
    assert torch.equal(out, ref)
