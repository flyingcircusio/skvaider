"""Sensu check for skvaider proxy functionality.

Calls the skvaider /health endpoint and translates the response to
Nagios/Sensu exit codes:
  0 = OK
  1 = WARNING
  2 = CRITICAL
"""

import argparse
import sys
import tomllib
from textwrap import indent

import httpx

from skvaider.config import Config
from skvaider.proxy.models import CheckResult


def render_check(check: CheckResult) -> str:
    line = f"{check.status.upper()} {check.name}"
    if check.message:
        line += f": {check.message}"
    lines = [line]
    lines.extend(indent(render_check(c), " " * 4) for c in check.children)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sensu check for skvaider proxy functionality."
    )
    parser.add_argument(
        "url",
        help="Skvaider proxy base URL (e.g. https://ai.whq.risclog.com)",
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to skvaider proxy config.toml",
    )
    parser.add_argument(
        "--request-timeout",
        default=120,
        type=int,
        help="Timeout in seconds for the /health request.",
    )
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = Config.model_validate(tomllib.load(f))

    if not config.auth.admin_tokens:
        print("CHECKS CRITICAL - no bearer token: set admin_tokens in config")
        sys.exit(2)

    url = args.url.rstrip("/") + "/health"
    try:
        response = httpx.get(
            url,
            headers={"Authorization": f"Bearer {config.auth.admin_tokens[0]}"},
            timeout=args.request_timeout,
        )
        response.raise_for_status()
        check = CheckResult.model_validate_json(response.content)
    except Exception as e:
        print(f"CHECKS CRITICAL - could not reach {url}: {e}")
        sys.exit(2)

    print(render_check(check))

    sys.exit(check.highest_status().status_code())


if __name__ == "__main__":
    main()
