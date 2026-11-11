import pytest


def _make_db(path, n_tables):
    import sqlalchemy

    engine = sqlalchemy.create_engine(f"sqlite:///{path}")
    meta = sqlalchemy.MetaData()
    for i in range(n_tables):
        sqlalchemy.Table(
            f"t{i}",
            meta,
            sqlalchemy.Column("k", sqlalchemy.Integer, primary_key=True),
            sqlalchemy.Column("v", sqlalchemy.Integer),
        )
    meta.create_all(engine)
    return engine


def test_build_model_reflects_only_target_table(tmp_path):
    pytest.importorskip("flag_gems")
    from flag_gems.utils.models.sql import SQLPersistantModel

    from vllm_fl.patches.flaggems_libcache import apply_flaggems_libcache_patch

    assert apply_flaggems_libcache_patch()
    engine = _make_db(tmp_path / "tc.db", 20)
    cls = SQLPersistantModel.build_sql_model_by_db("t7", engine)
    assert cls is not None and cls.__table__.name == "t7"
    assert set(cls.__table__.metadata.tables) == {"t7"}
    assert SQLPersistantModel.build_sql_model_by_db("missing", engine) is None


def test_config_roundtrip_with_patch(tmp_path):
    pytest.importorskip("flag_gems")
    import triton
    from flag_gems.utils.models.sql import SQLPersistantModel

    from vllm_fl.patches.flaggems_libcache import apply_flaggems_libcache_patch

    assert apply_flaggems_libcache_patch()
    url = f"sqlite:///{tmp_path / 'rt.db'}"
    model = SQLPersistantModel(url)
    model.put_config("kern", (128, 256), triton.Config({"BLOCK": 64}, num_warps=4))
    fresh = SQLPersistantModel(
        url
    )  # new process view: table must be found by reflection
    cfg = fresh.get_config("kern", (128, 256))
    assert cfg is not None and cfg.kwargs["BLOCK"] == 64 and cfg.num_warps == 4
    assert fresh.get_config("kern", (1, 2)) is None
    assert fresh.get_config("other", (128, 256)) is None
