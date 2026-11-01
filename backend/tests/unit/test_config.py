from pathlib import Path

import pytest

from app.core.config import PipelineConfig, Settings, load_pipeline_config
from app.core.errors import InvalidConfigError


def test_overrides_are_deep_merged_and_do_not_mutate_original() -> None:
    base = PipelineConfig()
    updated = base.with_overrides({"baseline": {"top_k": 8}})
    assert updated.baseline.top_k == 8
    assert updated.chunking == base.chunking
    assert base.baseline.top_k == 5


@pytest.mark.parametrize(
    "overrides",
    [
        {"baseline": {"top_kk": 8}},  # typo must not silently fall back to the default
        {"baseline": {"top_k": 0}},
        {"chunking": {"chunk_size": 200, "chunk_overlap": 200}},
    ],
)
def test_invalid_overrides_are_rejected(overrides: dict) -> None:
    with pytest.raises(InvalidConfigError):
        PipelineConfig().with_overrides(overrides)


def test_yaml_config_loads(tmp_path: Path) -> None:
    path = tmp_path / "pipeline.yaml"
    path.write_text("baseline:\n  top_k: 3\nchunking:\n  chunk_size: 500\n  chunk_overlap: 50\n", encoding="utf-8")
    config = load_pipeline_config(path)
    assert config.baseline.top_k == 3
    assert config.chunking.chunk_size == 500


def test_settings_derive_storage_paths(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path, openai_api_key="", database_url="")
    assert settings.database_url == f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}"
    assert settings.chroma_dir == tmp_path / "chroma"
    assert settings.llm_configured is False
