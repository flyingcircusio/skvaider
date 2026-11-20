import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from skvaider import check
from skvaider.proxy.models import CheckResult, Status


def test_render_check_leaf():
    assert (
        check.render_check(
            CheckResult(name="memory", status="ok", message="ok")
        )
        == "OK memory: ok"
    )


def test_render_check_omits_empty_message():
    assert (
        check.render_check(CheckResult(name="root", status="ok", message=""))
        == "OK root"
    )


@pytest.mark.parametrize(
    "status, expected",
    [
        ("ok", "OK x: msg"),
        ("warning", "WARNING x: msg"),
        ("critical", "CRITICAL x: msg"),
    ],
)
def test_render_check_status_prefix(status: Status, expected: str):
    assert (
        check.render_check(CheckResult(name="x", status=status, message="msg"))
        == expected
    )


def test_render_check_keeps_sibling_order():
    tree = CheckResult(
        name="root",
        status="ok",
        message="",
        children=[
            CheckResult(name=name, status="ok", message="")
            for name in ["b", "c", "a"]
        ],
    )
    assert check.render_check(tree) == "OK root\n    OK b\n    OK c\n    OK a"


def test_render_check_has_no_trailing_newline():
    tree = CheckResult(
        name="root",
        status="ok",
        message="",
        children=[
            CheckResult(
                name="a",
                status="ok",
                message="",
                children=[CheckResult(name="b", status="ok", message="")],
            )
        ],
    )
    rendered = check.render_check(tree)
    assert not rendered.endswith("\n")
    assert "\n\n" not in rendered


def test_render_check_indents_multiline_messages_with_their_check():
    tree = CheckResult(
        name="root",
        status="critical",
        message="",
        children=[
            CheckResult(
                name="completion",
                status="critical",
                message="first line\nsecond line",
            )
        ],
    )
    assert check.render_check(tree) == (
        "CRITICAL root\n    CRITICAL completion: first line\n    second line"
    )


def test_render_check_tree():
    tree = CheckResult(
        name="Skvaider",
        status="critical",
        message="",
        children=[
            CheckResult(
                name="gemma",
                status="ok",
                message="ok",
                children=[
                    CheckResult(
                        name="http://backend-1",
                        status="critical",
                        message="",
                        children=[
                            CheckResult(
                                name="memory", status="ok", message="ok"
                            ),
                            CheckResult(
                                name="completion",
                                status="critical",
                                message="timeout",
                            ),
                        ],
                    )
                ],
            ),
            CheckResult(
                name="embeddinggemma",
                status="critical",
                message="no loaded instance",
            ),
        ],
    )
    assert check.render_check(tree) == (
        "CRITICAL Skvaider\n"
        "    OK gemma: ok\n"
        "        CRITICAL http://backend-1\n"
        "            OK memory: ok\n"
        "            CRITICAL completion: timeout\n"
        "    CRITICAL embeddinggemma: no loaded instance"
    )


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(f"""
backend = []
models = []

[auth]
admin_tokens = ["secret"]

[server]
directory = "{tmp_path}"

[logging]
""")
    return path


class CheckRunner:
    """Runs check.main() against a faked /health response."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, config_file: Path):
        self.monkeypatch = monkeypatch
        self.config_file = config_file
        self.requests: list[dict[str, Any]] = []

    def __call__(self, response: httpx.Response | Exception) -> int | None:
        def fake_get(url: str, **kwargs: Any) -> httpx.Response:
            self.requests.append({"url": url, **kwargs})
            if isinstance(response, Exception):
                raise response
            response.request = httpx.Request("GET", url)
            return response

        self.monkeypatch.setattr(httpx, "get", fake_get)
        self.monkeypatch.setattr(
            sys,
            "argv",
            [
                "check-skvaider",
                "http://proxy/",
                "--config",
                str(self.config_file),
            ],
        )
        with pytest.raises(SystemExit) as exc_info:
            check.main()
        code = exc_info.value.code
        assert code is None or isinstance(code, int)
        return code


@pytest.fixture
def run_check(monkeypatch: pytest.MonkeyPatch, config_file: Path):
    return CheckRunner(monkeypatch, config_file)


def health_response(status: Status, leaf_status: Status) -> httpx.Response:
    tree = CheckResult(
        name="Skvaider",
        status=status,
        message="",
        children=[CheckResult(name="leaf", status=leaf_status, message="x")],
    )
    return httpx.Response(200, content=tree.model_dump_json())


@pytest.mark.parametrize(
    "status, leaf_status, exit_code",
    [
        ("ok", "ok", 0),
        ("warning", "warning", 1),
        ("critical", "critical", 2),
        # Exit code is derived from the tree, not only from the root.
        ("ok", "warning", 1),
        ("ok", "critical", 2),
    ],
)
def test_main_exit_codes(
    run_check: CheckRunner,
    capsys: pytest.CaptureFixture[str],
    status: Status,
    leaf_status: Status,
    exit_code: int,
):
    assert run_check(health_response(status, leaf_status)) == exit_code

    out = capsys.readouterr().out
    assert out.startswith(f"{status.upper()} Skvaider\n")
    assert f"    {leaf_status.upper()} leaf: x" in out

    [request] = run_check.requests
    assert request["url"] == "http://proxy/health"
    assert request["headers"] == {"Authorization": "Bearer secret"}


def test_main_unreachable(
    run_check: CheckRunner, capsys: pytest.CaptureFixture[str]
):
    assert run_check(httpx.ConnectError("refused")) == 2
    assert capsys.readouterr().out.startswith(
        "CHECKS CRITICAL - could not reach http://proxy/health: refused"
    )


def test_main_http_error(
    run_check: CheckRunner, capsys: pytest.CaptureFixture[str]
):
    assert run_check(httpx.Response(401)) == 2
    assert "CHECKS CRITICAL - could not reach" in capsys.readouterr().out


def test_main_invalid_response(
    run_check: CheckRunner, capsys: pytest.CaptureFixture[str]
):
    assert run_check(httpx.Response(200, content=b'{"status": "ok"}')) == 2
    assert "CHECKS CRITICAL - could not reach" in capsys.readouterr().out


def test_main_without_admin_tokens(
    run_check: CheckRunner,
    config_file: Path,
    capsys: pytest.CaptureFixture[str],
):
    config_file.write_text(config_file.read_text().replace('["secret"]', "[]"))
    assert run_check(health_response("ok", "ok")) == 2
    assert capsys.readouterr().out.startswith(
        "CHECKS CRITICAL - no bearer token"
    )
    assert run_check.requests == []
