import pytest

from src.utils.logging_utils import get_logger, require_env


def test_get_logger_returns_named_logger():
    assert get_logger("x").name == "x"


def test_require_env_raises_listing_all_missing(monkeypatch):
    monkeypatch.delenv("FOO_X", raising=False)
    monkeypatch.delenv("BAR_Y", raising=False)
    with pytest.raises(RuntimeError) as ei:
        require_env(["FOO_X", "BAR_Y"])
    assert "FOO_X" in str(ei.value) and "BAR_Y" in str(ei.value)


def test_require_env_passes_when_present(monkeypatch):
    monkeypatch.setenv("FOO_X", "1")
    require_env(["FOO_X"])  # no raise
