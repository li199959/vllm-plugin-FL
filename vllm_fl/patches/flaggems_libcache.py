# Copyright (c) 2026 BAAI. All rights reserved.
"""Make FlagGems' persistent autotune cache lookup O(1 table) instead of O(DB).

``SQLPersistantModel.build_sql_model_by_db`` runs ``automap_base().prepare(engine)``,
which reflects *every* table in the tuning DB, to fetch one model class. It is
called on each lookup whose table is not in the in-process pool yet, and misses
are not memoized (newly created tables are also pooled under the unprefixed name,
so they miss again). With the shared ~1500-table PPU DB one reflection takes ~4 s
of CPU in the engine core; serving workloads with varying shapes (Whisper's
encoder/decoder, many new autotune keys) hit it repeatedly and the GPU idles.

The replacement checks ``has_table`` (~0.2 ms) and reflects only that table
(~8 ms). Behaviour is otherwise identical: same class lookup by table name, None
when the table does not exist.
"""

import logging

logger = logging.getLogger(__name__)

_installed = False


def _build_sql_model_by_db(name, engine):
    import sqlalchemy
    from sqlalchemy.ext.automap import automap_base

    if not sqlalchemy.inspect(engine).has_table(name):
        return None
    auto_base = automap_base()
    auto_base.prepare(autoload_with=engine, reflection_options={"only": [name]})
    return auto_base.classes.get(name)


def apply_flaggems_libcache_patch() -> bool:
    global _installed
    if _installed:
        return True
    try:
        from flag_gems.utils.models.sql import SQLPersistantModel
    except ImportError:
        return False
    SQLPersistantModel.build_sql_model_by_db = staticmethod(_build_sql_model_by_db)
    _installed = True
    logger.info("FlagGems tuning cache: per-table reflection")
    return True


__all__ = ["apply_flaggems_libcache_patch"]
