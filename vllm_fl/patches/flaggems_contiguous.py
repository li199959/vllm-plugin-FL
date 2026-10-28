# Copyright (c) 2026 BAAI. All rights reserved.
"""Guard FlagGems kernels against inputs their launchers do not handle.

- ``_upsample_bicubic2d_aa`` passes the raw input pointer to a kernel that indexes
  it as contiguous NCHW and ignores strides. ``F.interpolate`` on a permuted view
  (e.g. SAM-ViT's ``get_abs_pos``: ``pos_embed.permute(0, 3, 1, 2)``) therefore
  returns unrelated values (cos 0.0004 vs aten for a 64x64 -> 40x40 resize), which
  corrupts DeepEncoder features for every cropped tile in DeepSeek-OCR-family
  models. The wrapper makes the input contiguous first.
- ``sum`` (full reduction) computes ``block_size = next_power_of_2(ceil(sqrt(0)))
  = 0`` for an empty tensor and raises ZeroDivisionError. DeepSeek-OCR calls
  ``torch.sum(patches)`` on the (empty) crop tensor of every image that fits one
  640x640 tile, which killed the engine. The wrapper returns a zero of the result
  dtype, as aten does.

Entries in ``flag_gems._FULL_CONFIG`` are swapped for the wrappers; other inputs
take the original path. This must run before ``flag_gems.enable()`` (the worker
enables Gems after general plugins load).
"""

import logging

import torch

logger = logging.getLogger(__name__)

_installed = False


def _wrap(fn, guard):
    def wrapper(*args, **kwargs):
        return guard(fn, *args, **kwargs)

    wrapper.__name__ = fn.__name__
    wrapper.__qualname__ = getattr(fn, "__qualname__", fn.__name__)
    wrapper.wrapped = fn
    return wrapper


def _contiguous_input(fn, input, *args, **kwargs):
    return fn(input.contiguous(), *args, **kwargs)


def _empty_sum(fn, inp, *args, dtype=None, **kwargs):
    if inp.numel() == 0:
        if dtype is None:
            dtype = torch.int64 if inp.dtype == torch.bool else inp.dtype
        return torch.zeros((), dtype=dtype, device=inp.device)
    return fn(inp, *args, dtype=dtype, **kwargs)


_GUARDS = {
    "_upsample_bicubic2d_aa": _contiguous_input,
    "sum": _empty_sum,
}


def apply_flaggems_contiguous_patches() -> bool:
    global _installed
    if _installed:
        return True
    try:
        import flag_gems
    except ImportError:
        return False

    patched = []
    config = []
    for entry in flag_gems._FULL_CONFIG:
        guard = _GUARDS.get(entry[0])
        if guard is not None and not hasattr(entry[1], "wrapped"):
            entry = (entry[0], _wrap(entry[1], guard), *entry[2:])
            patched.append(entry[0])
        config.append(entry)
    flag_gems._FULL_CONFIG = tuple(config)
    # only_enable(include=...) looks entries up by function name.
    for entry in config:
        if entry[0] in patched:
            flag_gems.FULL_CONFIG_BY_FUNC[entry[1].__name__] = [
                e for e in config if e[1].__name__ == entry[1].__name__
            ]
    _installed = True
    logger.info("FlagGems input guards: %s", patched)
    return True


__all__ = ["apply_flaggems_contiguous_patches"]
