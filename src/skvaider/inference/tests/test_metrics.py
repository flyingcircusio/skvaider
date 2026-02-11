import json
from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from prometheus_client import Counter

from skvaider.inference import metrics
from skvaider.inference.manager import Manager, Model
from skvaider.inference.routers.models import (
    _extract_token_usage as _extract_token_usage,  # pyright: ignore[reportPrivateUsage]
)
from skvaider.inference.routers.models import (
    _record_usage as _record_usage,  # pyright: ignore[reportPrivateUsage]
)


def _counter_value(counter: Counter, **labels: str) -> float:
    """Read the current value of a prometheus Counter with the given labels."""
    return counter.labels(
        **labels
    )._value.get()  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType, reportReturnType]


def _make_sse(*events: dict[str, Any] | str) -> bytes:
    """Build an SSE byte-stream from a list of events."""
    lines: list[str] = []
    for ev in events:
        if isinstance(ev, str):
            lines.append(f"data: {ev}")
        else:
            lines.append(f"data: {json.dumps(ev)}")
        lines.append("")  # blank line after each event
    return "\n".join(lines).encode()


def test_record_usage_prompt_and_completion():
    before_prompt = _counter_value(
        metrics.inference_tokens_prompt, model="test-unit"
    )
    before_gen = _counter_value(
        metrics.inference_tokens_generated, model="test-unit"
    )

    _record_usage(
        {"usage": {"prompt_tokens": 10, "completion_tokens": 20}},
        "test-unit",
    )

    assert (
        _counter_value(metrics.inference_tokens_prompt, model="test-unit")
        == before_prompt + 10
    )
    assert (
        _counter_value(metrics.inference_tokens_generated, model="test-unit")
        == before_gen + 20
    )


def test_record_usage_ignores_missing_usage():
    before_prompt = _counter_value(
        metrics.inference_tokens_prompt, model="test-unit2"
    )
    _record_usage({"id": "cmpl-1"}, "test-unit2")
    assert (
        _counter_value(metrics.inference_tokens_prompt, model="test-unit2")
        == before_prompt
    )


def test_record_usage_ignores_zero_tokens():
    before_prompt = _counter_value(
        metrics.inference_tokens_prompt, model="test-unit3"
    )
    _record_usage(
        {"usage": {"prompt_tokens": 0, "completion_tokens": 0}},
        "test-unit3",
    )
    assert (
        _counter_value(metrics.inference_tokens_prompt, model="test-unit3")
        == before_prompt
    )


def test_extract_plain_json_response():
    body = json.dumps(
        {
            "id": "cmpl-1",
            "usage": {"prompt_tokens": 5, "completion_tokens": 12},
        }
    ).encode()

    before_prompt = _counter_value(
        metrics.inference_tokens_prompt, model="extract-json"
    )
    before_gen = _counter_value(
        metrics.inference_tokens_generated, model="extract-json"
    )

    _extract_token_usage(body, "extract-json")

    assert (
        _counter_value(metrics.inference_tokens_prompt, model="extract-json")
        == before_prompt + 5
    )
    assert (
        _counter_value(metrics.inference_tokens_generated, model="extract-json")
        == before_gen + 12
    )


def test_extract_sse_stream_with_usage_in_last_event():
    body = _make_sse(
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
        {
            "choices": [],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2},
        },
        "[DONE]",
    )

    before_prompt = _counter_value(
        metrics.inference_tokens_prompt, model="extract-sse"
    )
    before_gen = _counter_value(
        metrics.inference_tokens_generated, model="extract-sse"
    )

    _extract_token_usage(body, "extract-sse")

    assert (
        _counter_value(metrics.inference_tokens_prompt, model="extract-sse")
        == before_prompt + 8
    )
    assert (
        _counter_value(metrics.inference_tokens_generated, model="extract-sse")
        == before_gen + 2
    )


def test_extract_empty_body_does_not_crash():
    _extract_token_usage(b"", "extract-empty")


def test_extract_garbage_body_does_not_crash():
    _extract_token_usage(b"\xff\xfe not json at all", "extract-garbage")


async def test_metrics_endpoint_returns_prometheus_format(
    client: TestClient, gemma: Model
):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    # Should contain at least one of our custom metrics families
    assert "skvaider_inference_requests_total" in body


async def test_proxy_unavailable_increments_counter(
    client: TestClient, manager: Manager
):
    """A 540 response should be counted as 'unavailable'."""
    before = _counter_value(
        metrics.inference_requests_total,
        model="gemma",
        endpoint="/proxy/v1/chat/completions",
        status="unavailable",
    )

    with patch.object(manager, "use_model", new_callable=AsyncMock) as mock:
        mock.return_value = None
        response = client.get("/models/gemma/proxy/v1/chat/completions")
        assert response.status_code == 540

    after = _counter_value(
        metrics.inference_requests_total,
        model="gemma",
        endpoint="/proxy/v1/chat/completions",
        status="unavailable",
    )
    assert after == before + 1
