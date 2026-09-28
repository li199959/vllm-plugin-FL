# Copyright (c) 2026 BAAI. All rights reserved.
"""Opt-in reuse of recently freed KV blocks on PPU.

VLLM_FL_LIFO_ALLOC=1 enables the policy. Prefix-cached pools retain
FIFO allocation because their queue order determines LRU eviction.
Attach popright_n when the installed vLLM does not provide it.
"""

import os

_APPLIED = False


def _fl_lifo_requested() -> bool:
    return os.getenv("VLLM_FL_LIFO_ALLOC", "0") == "1"


def _free_queue_cls():
    """Resolve the free-block queue class across vLLM versions.

    Named ``FreeKVCacheBlockQueue`` in this tree; fall back to shape
    detection (has ``popleft_n``) so a rename cannot crash the hook again.
    """
    from vllm.v1.core import kv_cache_utils as kcu

    cls = getattr(kcu, "FreeKVCacheBlockQueue", None)
    if cls is not None:
        return cls
    for name in dir(kcu):
        obj = getattr(kcu, name)
        if isinstance(obj, type) and obj.__module__ == kcu.__name__ and hasattr(
            obj, "popleft_n"
        ):
            return obj
    raise AttributeError(
        "free-block queue class (FreeKVCacheBlockQueue) not found in "
        "vllm.v1.core.kv_cache_utils"
    )


def _ensure_popright_n() -> None:
    """Attach queue.popright_n if the vLLM tree lacks it."""
    cls = _free_queue_cls()

    if hasattr(cls, "popright_n"):
        return

    def popright_n(self, n: int) -> list:
        if n == 0:
            return []
        assert self.num_free_blocks >= n
        self.num_free_blocks -= n

        curr = self.fake_free_list_tail.prev_free_block
        ret = []
        for _ in range(n):
            assert curr is not None
            ret.append(curr)
            last = curr
            curr = curr.prev_free_block
            last.prev_free_block = None
            last.next_free_block = None
        if curr is not None:
            self.fake_free_list_tail.prev_free_block = curr
            curr.next_free_block = self.fake_free_list_tail
        return ret

    cls.popright_n = popright_n


def apply_lifo_alloc_patch() -> bool:
    """Monkeypatch BlockPool.get_new_blocks to allocate LIFO.

    Returns True if the patch is (already) active, False if not requested.
    """
    global _APPLIED
    if _APPLIED:
        return True
    if not _fl_lifo_requested():
        return False

    from vllm.v1.core import block_pool

    bp = getattr(block_pool, "BlockPool", None)
    if bp is None:
        # Circular-import window: register() can be invoked by vllm's own
        # import chain while block_pool is still initializing. The platform
        # hook (PlatformFL.check_and_update_config) retries post-import.
        return False

    if getattr(bp.get_new_blocks, "_fl_lifo", False):
        _APPLIED = True
        return True

    _ensure_popright_n()
    orig = block_pool.BlockPool.get_new_blocks

    def get_new_blocks_lifo(self, num_blocks: int) -> list:
        if self.enable_caching:
            # Prefix caching relies on head-of-queue LRU order.
            return orig(self, num_blocks)
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from the pool"
            )
        ret = self.free_block_queue.popright_n(num_blocks)
        for block in ret:
            assert block.ref_cnt == 0
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_allocated(block)
        return ret

    get_new_blocks_lifo._fl_lifo = True
    bp.get_new_blocks = get_new_blocks_lifo
    _APPLIED = True
    return True
