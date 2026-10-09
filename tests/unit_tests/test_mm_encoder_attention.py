# SPDX-License-Identifier: Apache-2.0
"""The OOT encoder must use the backend selected by the platform."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("flash_selected", [True, False])
def test_thead_encoder_dispatch(monkeypatch, flash_selected):
    import vllm.model_executor.layers.attention.mm_encoder_attention as mm
    import vllm.platforms as platforms
    import vllm.v1.attention.backends.fa_utils as fa_utils
    from vllm_fl.attention.utils import patch_mm_encoder_attention

    native = Mock(return_value="sdpa")
    flash = Mock(return_value="fa3")

    class Encoder:
        is_flash_attn_backend = flash_selected

        def forward_native(self, *args):
            return native(self, *args)

        def _forward_fa(self, *args):
            return flash(*args)

    bridge = Mock()
    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(vendor_name="thead")
    )
    monkeypatch.setattr(mm, "MMEncoderAttention", Encoder)
    monkeypatch.setattr(mm, "maybe_get_vit_flash_attn_backend", None, raising=False)
    monkeypatch.setattr(fa_utils, "flash_attn_varlen_func", None, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.dispatch.backends.vendor.thead.impl.attention",
        SimpleNamespace(_thead_flash_attn_varlen_func=bridge),
    )
    patch_mm_encoder_attention()
    patched = Encoder.forward_native
    patch_mm_encoder_attention()
    assert Encoder.forward_native is patched
    assert fa_utils.flash_attn_varlen_func is bridge

    encoder = Encoder()
    q, k, v, cu, max_len, lengths = [object() for _ in range(6)]
    result = encoder.forward_native(q, k, v, cu, max_len, lengths)
    if flash_selected:
        assert result == "fa3"
        flash.assert_called_once_with(q, k, v, cu, max_len)
        native.assert_not_called()
    else:
        assert result == "sdpa"
        native.assert_called_once_with(encoder, q, k, v, cu, max_len, lengths)
        flash.assert_not_called()
