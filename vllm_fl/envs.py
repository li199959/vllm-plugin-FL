# Copyright (c) 2025 BAAI. All rights reserved.

import os
from typing import Any, Callable

from vllm_fl.utils import use_flaggems

fl_vllm_environment_variables: dict[str, Callable[[], Any]] = {
    "VLLM_FL_PREFER_ENABLED": lambda: (
        os.environ.get("VLLM_FL_PREFER_ENABLED", "True").lower() in ("true", "1")
    ),
    "FLAGGEMS_ENABLE_OPLIST_PATH": lambda: os.environ.get(
        "FLAGGEMS_ENABLE_OPLIST_PATH", "/tmp/flaggems_enable_oplist.txt"
    ),
    "USE_FLAGGEMS": use_flaggems,
    "VLLM_FL_MEMORY_PROFILER_DIR": lambda: os.path.abspath(
        os.path.expanduser(os.environ.get("VLLM_FL_MEMORY_PROFILER_DIR", ""))
    ) if os.environ.get("VLLM_FL_MEMORY_PROFILER_DIR") else "",
    "VLLM_FL_MEMORY_PROFILER_MAX_ENTRIES": lambda: int(
        os.environ.get("VLLM_FL_MEMORY_PROFILER_MAX_ENTRIES", "100000")
    ),
    "VLLM_FL_MEMORY_PROFILER_PROFILE_INIT": lambda: os.environ.get(
        "VLLM_FL_MEMORY_PROFILER_PROFILE_INIT", "1"
    ).lower() in ("true", "1", "yes", "on"),
    "VLLM_FL_MEMORY_PROFILER_DUMP_ON_EXCEPTION": lambda: os.environ.get(
        "VLLM_FL_MEMORY_PROFILER_DUMP_ON_EXCEPTION", "1"
    ).lower() in ("true", "1", "yes", "on"),
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in fl_vllm_environment_variables:
        return fl_vllm_environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(fl_vllm_environment_variables.keys())


def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in fl_vllm_environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
