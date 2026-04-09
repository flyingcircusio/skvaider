"""Fetch embeddings from the live proxy and write a reference JSON file.

Output format: {model_id: {input_text: [float, ...]}}
This matches the embedding_verification_file format consumed by inference
health checks and by skvaider-check --reference-file.
"""

import argparse
import json
import os
import sys
import tomllib

import httpx

from skvaider.config import Config


def _fetch_embeddings_batch(
    client: httpx.Client,
    base: str,
    model_id: str,
    texts: list[str],
) -> dict[str, list[float]]:
    """POST a batch of texts to the embeddings endpoint and return {text: vector}."""
    response = client.post(
        base + "/openai/v1/embeddings",
        json={"model": model_id, "input": texts, "encoding_format": "float"},
    )
    response.raise_for_status()
    result = response.json()
    # Each item in data carries an index that maps back to the original texts list.
    return {texts[item["index"]]: item["embedding"] for item in result["data"]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch embeddings from skvaider proxy and write a reference JSON file."
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("SKVAIDER_CONFIG_FILE"),
        metavar="PATH",
        help="Path to proxy config.toml (default: $SKVAIDER_CONFIG_FILE)",
    )
    parser.add_argument(
        "--url",
        default=None,
        metavar="URL",
        help="Proxy base URL; overrides value derived from config",
    )
    parser.add_argument(
        "--dataset",
        default="dataset.txt",
        metavar="PATH",
        help="Text file with one input per line (default: dataset.txt)",
    )
    parser.add_argument(
        "--output",
        default="embeddings-reference.json",
        metavar="PATH",
        help="Output JSON file path (default: embeddings-reference.json)",
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
            "error: no config file — pass --config or set SKVAIDER_CONFIG_FILE",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(args.config, "rb") as f:
        config = Config.model_validate(tomllib.load(f))

    embedding_models = [m for m in config.models if m.task == "embedding"]
    if not embedding_models:
        print("error: no embedding models found in config", file=sys.stderr)
        sys.exit(1)

    if args.url:
        base = args.url.rstrip("/")
    else:
        host = config.server.host
        if host == "0.0.0.0":
            host = "127.0.0.1"
        base = f"http://{host}:{config.server.port}"

    if config.auth.static_tokens:
        key = config.auth.static_tokens[0]
    else:
        print(
            "error: no bearer token — set static_tokens in config",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(args.dataset) as f:
        texts = [line.strip() for line in f if line.strip()]

    if not texts:
        print(f"error: dataset file {args.dataset!r} is empty", file=sys.stderr)
        sys.exit(1)

    reference: dict[str, dict[str, list[float]]] = {}

    try:
        with httpx.Client(
            headers={"Authorization": f"Bearer {key}"},
            timeout=args.request_timeout,
        ) as client:
            for model_cfg in embedding_models:
                mid = model_cfg.id
                print(
                    f"fetching embeddings for model {mid!r} ({len(texts)} texts)..."
                )
                reference[mid] = _fetch_embeddings_batch(
                    client, base, mid, texts
                )
                print(f"  done: {len(reference[mid])} vectors")
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    with open(args.output, "w") as f:
        json.dump(reference, f, separators=(",", ":"))
        f.write("\n")

    print(f"wrote reference to {args.output!r}")


if __name__ == "__main__":
    main()
