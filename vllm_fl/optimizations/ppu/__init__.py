# Copyright (c) 2026 BAAI. All rights reserved.
"""T-Head PPU model optimizations. Add new models to ``_MODELS``."""

import importlib

_MODELS: tuple[str, ...] = ()


def model_packages():
    return [importlib.import_module(f"{__name__}.{name}") for name in _MODELS]
