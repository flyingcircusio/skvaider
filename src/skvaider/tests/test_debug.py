import os
import time
from pathlib import Path

import pytest

from skvaider.debug import cleanup_old_debug_files


def test_cleanup_removes_old_request_files(tmp_path: Path):
    """Files older than TTL are removed."""
    old_file = tmp_path / "req-1.request"
    old_file.write_text("old request data")
    # Set mtime to 10 days ago
    old_time = time.time() - 10 * 86400
    os.utime(old_file, (old_time, old_time))

    ttl_seconds = 4 * 86400  # 4 days
    removed = cleanup_old_debug_files(tmp_path, ttl_seconds)

    assert removed == 1
    assert not old_file.exists()


def test_cleanup_removes_old_response_files(tmp_path: Path):
    """Old .response files are also removed."""
    old_file = tmp_path / "req-1.response"
    old_file.write_text("old response data")
    old_time = time.time() - 5 * 86400
    os.utime(old_file, (old_time, old_time))

    ttl_seconds = 4 * 86400
    removed = cleanup_old_debug_files(tmp_path, ttl_seconds)

    assert removed == 1
    assert not old_file.exists()


def test_cleanup_keeps_recent_files(tmp_path: Path):
    """Files within TTL are kept."""
    recent = tmp_path / "req-2.request"
    recent.write_text("recent data")
    # mtime is now (default), so it's recent

    ttl_seconds = 4 * 86400
    removed = cleanup_old_debug_files(tmp_path, ttl_seconds)

    assert removed == 0
    assert recent.exists()


def test_cleanup_ignores_non_debug_files(tmp_path: Path):
    """Files without .request/.response suffix are ignored."""
    other = tmp_path / "something.log"
    other.write_text("not debug")
    old_time = time.time() - 10 * 86400
    os.utime(other, (old_time, old_time))

    ttl_seconds = 4 * 86400
    removed = cleanup_old_debug_files(tmp_path, ttl_seconds)

    assert removed == 0
    assert other.exists()


def test_cleanup_handles_missing_directory():
    """Returns 0 when directory doesn't exist, doesn't raise."""
    removed = cleanup_old_debug_files(
        Path("/nonexistent/path/that/does/not/exist"), 3600
    )
    assert removed == 0


def test_cleanup_removes_both_pair(tmp_path: Path):
    """When request is old, both .request and .response are removed."""
    req = tmp_path / "req-1.request"
    resp = tmp_path / "req-1.response"
    req.write_text("req")
    resp.write_text("resp")
    old_time = time.time() - 5 * 86400
    os.utime(req, (old_time, old_time))
    os.utime(resp, (old_time, old_time))

    ttl_seconds = 4 * 86400
    removed = cleanup_old_debug_files(tmp_path, ttl_seconds)

    assert removed == 2
    assert not req.exists()
    assert not resp.exists()


def test_cleanup_mixed_ages(tmp_path: Path):
    """Only old files are removed; recent files stay."""
    old_req = tmp_path / "old.request"
    old_req.write_text("old")
    old_time = time.time() - 10 * 86400
    os.utime(old_req, (old_time, old_time))

    new_req = tmp_path / "new.request"
    new_req.write_text("new")

    ttl_seconds = 4 * 86400
    removed = cleanup_old_debug_files(tmp_path, ttl_seconds)

    assert removed == 1
    assert not old_req.exists()
    assert new_req.exists()


def test_cleanup_handles_os_error_on_stat(tmp_path: Path):
    """Files that can't be stat'd are skipped."""
    import unittest.mock

    file = tmp_path / "broken.request"
    file.write_text("data")

    original_stat = Path.stat

    def selective_stat(self_self, *args, **kwargs):
        if str(self_self) == str(file):
            raise OSError("permission denied")
        return original_stat(self_self, *args, **kwargs)

    ttl_seconds = 4 * 86400
    with unittest.mock.patch.object(Path, "stat", selective_stat):
        removed = cleanup_old_debug_files(tmp_path, ttl_seconds)

    assert removed == 0
    assert file.exists()


def test_cleanup_empty_directory(tmp_path: Path):
    """Empty directory returns 0 removed."""
    removed = cleanup_old_debug_files(tmp_path, 3600)
    assert removed == 0


def test_cleanup_keeps_just_under_ttl(tmp_path: Path):
    """File just under the TTL boundary is kept."""
    ttl_seconds = 3600
    req = tmp_path / "under.request"
    req.write_text("data")
    # Set mtime to TTL minus 1 second (under the threshold)
    under_time = time.time() - (ttl_seconds - 1)
    os.utime(req, (under_time, under_time))

    removed = cleanup_old_debug_files(tmp_path, ttl_seconds)

    assert removed == 0
    assert req.exists()


def test_cleanup_boundary_just_over_ttl(tmp_path: Path):
    """File just over the TTL boundary IS removed."""
    req = tmp_path / "over.request"
    req.write_text("data")
    ttl_seconds = 3600
    # Set mtime to TTL + 1 second ago
    over_time = time.time() - ttl_seconds - 1
    os.utime(req, (over_time, over_time))

    removed = cleanup_old_debug_files(tmp_path, ttl_seconds)

    assert removed == 1
    assert not req.exists()
