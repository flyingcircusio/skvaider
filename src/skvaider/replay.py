import argparse
import json

import httpx


def _parse_metadata(comment_line: str) -> dict[str, str]:
    """Parse key=value pairs from a # skvaider-debug/1 comment line."""
    result: dict[str, str] = {}
    # Strip leading "# skvaider-debug/1  " prefix and split on whitespace
    parts = comment_line.lstrip("#").split()
    for part in parts:
        if "=" in part:
            key, _, value = part.partition("=")
            result[key] = value
    return result


def _parse_http_file(
    content: str,
) -> tuple[str, str, dict[str, str], dict[str, object], dict[str, str]]:
    """Parse a .request file in .http format.

    Returns (method, url, headers, body, metadata).
    """
    lines = content.splitlines()
    i = 0

    metadata: dict[str, str] = {}
    while i < len(lines) and (lines[i].startswith("#") or not lines[i].strip()):
        if lines[i].startswith("# skvaider-debug/"):
            metadata = _parse_metadata(lines[i])
        i += 1

    method, url = lines[i].split(" ", 1)
    i += 1

    headers: dict[str, str] = {}
    while i < len(lines) and lines[i].strip():
        key, _, value = lines[i].partition(": ")
        headers[key] = value
        i += 1

    i += 1

    body_str = "\n".join(lines[i:]).strip()
    body: dict[str, object] = json.loads(body_str) if body_str else {}

    return method, url, headers, body, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a captured skvaider debug request"
    )
    parser.add_argument("request_file", help="Path to .request debug file")
    parser.add_argument(
        "--backend",
        action="store_true",
        help="Replay directly against the inference backend instead of the proxy",
    )
    args = parser.parse_args()

    with open(args.request_file) as f:
        content = f.read()

    method, proxy_url, headers, body, metadata = _parse_http_file(content)

    _drop = {"content-length", "transfer-encoding"}
    safe_headers = {k: v for k, v in headers.items() if k.lower() not in _drop}

    def _print_response_headers(r: httpx.Response) -> None:
        print(f"< HTTP/{r.http_version} {r.status_code} {r.reason_phrase}")
        for k, v in r.headers.items():
            print(f"< {k}: {v}")
        print("<")

    def _print_body(r: httpx.Response) -> None:
        content_type = r.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            for chunk in r.iter_text():
                if not chunk.startswith("data: ") or chunk.startswith(
                    "data: [DONE]"
                ):
                    print(chunk, end="", flush=True)
                    continue
                try:
                    parsed = json.loads(chunk[6:].strip())
                    print(
                        "data: " + json.dumps(parsed, indent=2) + "\n",
                        flush=True,
                    )
                except json.JSONDecodeError:
                    print(chunk, end="", flush=True)
        else:
            raw = r.read().decode()
            try:
                print(json.dumps(json.loads(raw), indent=2))
            except json.JSONDecodeError:
                print(raw, end="")

    if args.backend:
        backend = metadata.get("backend", "")
        model = metadata.get("model", "")
        backend_endpoint = metadata.get("backend_endpoint", "")
        url = f"{backend}/models/{model}/proxy{backend_endpoint}"
        with httpx.Client(timeout=120) as client:
            with client.stream(method, url, json=body) as r:
                _print_response_headers(r)
                _print_body(r)
    else:
        with httpx.Client(timeout=120) as client:
            with client.stream(
                method, proxy_url, json=body, headers=safe_headers
            ) as r:
                _print_response_headers(r)
                _print_body(r)
