"""Sensu check for skvaider proxy functionality.

Calls the skvaider /health endpoint and translates the response to
Nagios/Sensu exit codes:
  0 = OK
  1 = WARNING
  2 = CRITICAL
"""

import argparse
import os
import sys
import tomllib

import httpx

from skvaider.config import Config


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
        default=os.environ.get("SKVAIDER_CONFIG_FILE"),
        metavar="PATH",
        help="Path to skvaider proxy config.toml (default: $SKVAIDER_CONFIG_FILE)",
    )
    parser.add_argument(
        "--request-timeout",
        default=120,
        type=int,
        help="Timeout in seconds for the /health request.",
    )
    args = parser.parse_args()

    if not args.config:
        print(
            "CHECKS CRITICAL - no config file: pass --config or set SKVAIDER_CONFIG_FILE"
        )
        sys.exit(2)

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
        data = response.json()
    except Exception as e:
        print(f"CHECKS CRITICAL - could not reach {url}: {e}")
        sys.exit(2)

    status = data.get("status", "critical")
    checks = data.get("checks", {})

    if status == "ok":
        print("CHECKS OK")
    elif status == "warning":
        failing = [n for n, c in checks.items() if c["status"] == "warning"]
        print(f"CHECKS WARNING - {', '.join(failing)}")
    else:
        failing = [n for n, c in checks.items() if c["status"] == "critical"]
        print(f"CHECKS CRITICAL - {', '.join(failing)}")

    for name, check in checks.items():
        s = check["status"].upper()
        print(f"{s} {name}: {check['message']}")

    sys.exit(2 if status == "critical" else 1 if status == "warning" else 0)


if __name__ == "__main__":
    main()
