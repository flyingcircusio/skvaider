"""Sensu check for skvaider proxy functionality.

Reads the proxy config.toml to discover all configured models and their tasks,
then tests each model against its applicable endpoint(s).

Exit codes follow the Nagios/Sensu convention:
  0 = OK
  1 = WARNING (unused, reserved)
  2 = CRITICAL
"""

import argparse
import json
import os
import random
import sys
import tomllib
from typing import Any

import httpx

from skvaider.config import Config


def _check_model_list(
    client: httpx.Client,
    base: str,
    models: dict[str, Any],
    min_expected: int,
) -> None:
    response = client.get(base + "/openai/v1/models")
    response.raise_for_status()
    models.update(response.json())
    assert len(models["data"]) >= min_expected, (
        f"too few models: {len(models['data'])} < {min_expected}"
    )
    assert models["data"][0]["object"] == "model", (
        "first item object field is not 'model'"
    )
    assert "id" in models["data"][0], "model entry missing `id`"
    assert "created" in models["data"][0], "model entry missing `created`"
    assert "owned_by" in models["data"][0], "model entry missing `owned_by`"


def _check_model_details(
    client: httpx.Client,
    base: str,
    models: dict[str, Any],
) -> None:
    model_id = random.choice(models["data"])["id"]
    response = client.get(base + f"/openai/v1/models/{model_id}")
    response.raise_for_status()
    model = response.json()
    assert model["object"] == "model", "object field is not 'model'"
    assert model["id"], f"model has wrong id: {model['id']!r}"
    assert "created" in model, "model missing `created`"
    assert "owned_by" in model, "model missing `owned_by`"


def _check_chat_completions(
    client: httpx.Client, base: str, model_id: str
) -> None:
    response = client.post(
        base + "/openai/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            # Allow enough tokens for thinking models that need a reasoning pass.
            "max_tokens": 1000,
        },
    )
    response.raise_for_status()
    result = response.json()
    assert result["object"] == "chat.completion", (
        f"unexpected object type: {result.get('object')!r}"
    )
    msg = result["choices"][0]["message"]
    assert "content" in msg, "response message missing `content`"
    assert "role" in msg, "response message missing `role`"


def _check_completions(client: httpx.Client, base: str, model_id: str) -> None:
    response = client.post(
        base + "/openai/v1/completions",
        json={
            "model": model_id,
            "prompt": "say hello",
            "stream": False,
            "max_tokens": 1000,
        },
    )
    response.raise_for_status()
    result = response.json()
    assert result["object"] == "text_completion", (
        f"unexpected object type: {result}"
    )
    assert "text" in result["choices"][0], (
        f"missing text in completion: {result}"
    )


def _check_embeddings(client: httpx.Client, base: str, model_id: str) -> None:
    response = client.post(
        base + "/openai/v1/embeddings",
        json={
            "input": "The food was delicious and the waiter...",
            "model": model_id,
            "encoding_format": "float",
        },
    )
    response.raise_for_status()
    result = response.json()
    assert result["object"] == "list", "response object is not 'list'"
    assert len(result["data"]) >= 1, "response data is empty"
    assert result["data"][0]["object"] == "embedding", (
        "first data item is not an embedding"
    )
    assert len(result["data"][0]["embedding"]) > 64, (
        "embedding has fewer than 64 dimensions"
    )
    assert isinstance(result["data"][0]["embedding"][0], float), (
        "embedding element is not a float"
    )


def _check_embeddings_reference(
    client: httpx.Client,
    base: str,
    model_id: str,
    reference: dict[str, dict[str, list[float]]],
) -> None:
    """Compare live embeddings for model_id against pre-recorded reference.

    Tolerance matches inference-side health check: abs(a - b) <= 1e-2.
    Silently skips if model_id is absent from the reference.
    """
    if model_id not in reference:
        return
    expected = reference[model_id]  # {text: vector}
    texts = list(expected.keys())
    response = client.post(
        base + "/openai/v1/embeddings",
        json={"model": model_id, "input": texts, "encoding_format": "float"},
    )
    response.raise_for_status()
    data = response.json()
    for item in data["data"]:
        text = texts[int(item["index"])]
        got = item["embedding"]
        exp = expected[text]
        assert len(got) == len(exp), (
            f"embedding dimension mismatch for {text!r}: "
            f"got {len(got)}, expected {len(exp)}"
        )
        for j, (a, b) in enumerate(zip(got, exp)):
            assert abs(a - b) <= 1e-2, (
                f"embedding value mismatch at dim {j} for {text!r}: "
                f"got {a:.6f}, expected {b:.6f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sensu check for skvaider proxy functionality."
    )
    parser.add_argument(
        "url",
        help="Skvaider proxy base URL (e.g. https://ai.whq.risclog.com)",
    )
    parser.add_argument(
        "keyfile",
        help="Path to file containing the API bearer token",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("SKVAIDER_CONFIG_FILE"),
        metavar="PATH",
        help="Path to skvaider proxy config.toml "
        "(default: $SKVAIDER_CONFIG_FILE)",
    )
    parser.add_argument(
        "--reference-file",
        default=None,
        metavar="PATH",
        help="Path to embedding reference JSON for numerical-stability checks",
    )
    parser.add_argument(
        "--request-timeout",
        default=120,
        type=int,
        help="Timeout for individual requests.",
    )
    args = parser.parse_args()

    if not args.config:
        print(
            "CHECKS CRITICAL - no config file: pass --config or set SKVAIDER_CONFIG_FILE"
        )
        sys.exit(2)

    with open(args.config, "rb") as f:
        config = Config.model_validate(tomllib.load(f))

    key = open(args.keyfile).read().strip()
    url = args.url.rstrip("/")

    reference: dict[str, dict[str, list[float]]] = {}
    if args.reference_file:
        with open(args.reference_file) as _f:
            reference = json.load(_f)

    oks: set[str] = set()
    errors: dict[str, str] = {}

    def run(name: str, fn: Any) -> None:
        try:
            fn()
        except Exception as e:
            errors[name] = str(e)
        else:
            oks.add(name)

    with httpx.Client(
        headers={"Authorization": f"Bearer {key}"},
        timeout=args.request_timeout,
    ) as client:
        # These two checks are not model-specific.
        models: dict[str, Any] = {}
        run(
            "check_model_list",
            lambda: _check_model_list(client, url, models, len(config.models)),
        )
        # Only attempt detail check if we have models to pick from.
        if models.get("data"):
            run(
                "check_model_details",
                lambda: _check_model_details(client, url, models),
            )
        else:
            errors["check_model_details"] = "skipped: model list unavailable"

        # Per-model checks keyed by task.
        for model_cfg in config.models:
            mid = model_cfg.id
            if model_cfg.task == "embedding":
                run(
                    f"check_embeddings[{mid}]",
                    lambda m=mid: _check_embeddings(client, url, m),
                )
                if reference:
                    run(
                        f"check_embeddings_reference[{mid}]",
                        lambda m=mid: _check_embeddings_reference(
                            client, url, m, reference
                        ),
                    )
            else:
                # Default task is "chat": test both completion endpoints.
                run(
                    f"check_chat_completions[{mid}]",
                    lambda m=mid: _check_chat_completions(client, url, m),
                )
                run(
                    f"check_completions[{mid}]",
                    lambda m=mid: _check_completions(client, url, m),
                )

    if errors:
        print("CHECKS CRITICAL - {}".format(", ".join(errors.keys())))
    else:
        print("CHECKS OK - {}".format(", ".join(oks)))

    for check, error in errors.items():
        print(f"CRITICAL {check}: {error}")
    for check in oks:
        print(f"OK {check}")

    sys.exit(2 if errors else 0)


if __name__ == "__main__":
    main()
