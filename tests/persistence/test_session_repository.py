from __future__ import annotations

from pyrpoc.domain.session_state import SessionState
from pyrpoc.persistence.session_repository import SessionRepository


def make_repo(tmp_path):
    """Build a repository pointed at a temp file, bypassing QStandardPaths."""
    repo = SessionRepository.__new__(SessionRepository)
    repo.path = tmp_path / "session.json"
    repo.last_load_error = None
    return repo


def test_load_default_when_missing(tmp_path):
    repo = make_repo(tmp_path)
    state = repo.load_or_default()
    assert isinstance(state, SessionState)
    assert repo.last_load_error is None


def test_save_then_load_round_trip(tmp_path):
    repo = make_repo(tmp_path)
    repo.save(SessionState(theme_mode="dark"))
    assert repo.path.exists()
    restored = repo.load_or_default()
    assert restored.theme_mode == "dark"
    assert repo.last_load_error is None


def test_save_is_atomic_and_leaves_no_tmp(tmp_path):
    repo = make_repo(tmp_path)
    repo.save(SessionState())
    tmp = repo.path.with_suffix(repo.path.suffix + ".tmp")
    assert not tmp.exists()


def test_corrupt_file_falls_back_to_default(tmp_path):
    repo = make_repo(tmp_path)
    repo.path.write_text("{ this is not valid json", encoding="utf-8")
    state = repo.load_or_default()
    assert isinstance(state, SessionState)
    assert repo.last_load_error is not None
    assert "Failed to load session" in repo.last_load_error
