"""Client for the maintenance mode of an inference server.

Called by the fc-agent maintenance hooks: ``enter`` asks the server to move
its models away and waits for the drain; if it times out, it exits with
``EXIT_TEMPFAIL`` so the agent postpones the maintenance. ``leave`` withdraws
the request once the machine is back.
"""

import argparse
import socket
import sys
import time
from collections.abc import Sequence

import httpx
from pydantic import ValidationError

from skvaider.manifest import MaintenanceStatus

EXIT_OK = 0
EXIT_FAILURE = 1
# fc-agent postpones due maintenance requests when an enter command exits
# with this code (fc.maintenance.state.EXIT_TEMPFAIL).
EXIT_TEMPFAIL = 75

MAINTENANCE_PATH = "/manager/maintenance"
DEFAULT_WAIT_SECONDS = 300.0
DEFAULT_POLL_INTERVAL = 2.0
DEFAULT_REQUEST_TIMEOUT = 10.0
LEAVE_TIMEOUT = 30.0


def maintenance_url(base_url: str) -> str:
    """Return the maintenance endpoint for a base URL (or the URL itself)."""
    base = base_url.rstrip("/")
    if base.endswith(MAINTENANCE_PATH):
        return base
    return base + MAINTENANCE_PATH


def log(message: str) -> None:
    print(f"skvaider-maintenance: {message}", flush=True)


class MaintenanceClient:
    def __init__(
        self,
        base_url: str,
        *,
        wait: float = DEFAULT_WAIT_SECONDS,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        reason: str | None = None,
    ) -> None:
        self.url = maintenance_url(base_url)
        self.wait = wait
        self.poll_interval = poll_interval
        self.request_timeout = request_timeout
        self.reason = reason or (
            f"fc-agent maintenance on {socket.gethostname()}"
        )

    def phase(self) -> str | None:
        """Return the maintenance phase, or None when it cannot be read."""
        try:
            response = httpx.get(self.url, timeout=self.request_timeout)
            response.raise_for_status()
            return MaintenanceStatus.model_validate(response.json()).phase
        except (httpx.HTTPError, ValidationError, ValueError):
            return None

    def request(self) -> httpx.Response:
        return httpx.post(
            self.url,
            json={"reason": self.reason},
            timeout=self.request_timeout,
        )


def enter(client: MaintenanceClient) -> int:
    try:
        client.request().raise_for_status()
    except httpx.ConnectError:
        log("inference server not reachable, nothing to move away")
        return EXIT_OK
    except httpx.HTTPError as e:
        log(f"could not request maintenance mode ({e}), postponing")
        return EXIT_TEMPFAIL

    log(
        "requested maintenance mode, waiting up to "
        f"{client.wait:g}s for models to move away"
    )
    deadline = time.monotonic() + client.wait
    while time.monotonic() < deadline:
        phase = client.phase()
        if phase == "drained":
            log("no models left on this machine, maintenance may continue")
            return EXIT_OK
        if phase == "none":
            log("maintenance request was lost by a restart, requesting again")
            try:
                client.request().raise_for_status()
            except httpx.HTTPError:
                pass
        elif phase not in ("requested", "draining", None):
            log(f"unexpected maintenance phase {phase!r}, waiting")
        time.sleep(client.poll_interval)

    log(f"models are still running after {client.wait:g}s, postponing")
    return EXIT_TEMPFAIL


def leave(client: MaintenanceClient) -> int:
    try:
        response = httpx.delete(client.url, timeout=client.request_timeout)
        response.raise_for_status()
    except httpx.ConnectError:
        log("inference server not reachable, nothing to do")
        return EXIT_OK
    except httpx.HTTPError as e:
        log(f"could not leave maintenance mode ({e})")
        return EXIT_FAILURE

    deadline = time.monotonic() + LEAVE_TIMEOUT
    while time.monotonic() < deadline:
        if client.phase() == "none":
            log("left maintenance mode, models may be placed here again")
            return EXIT_OK
        time.sleep(1.0)

    log(f"server is still in maintenance mode after {LEAVE_TIMEOUT:g}s")
    return EXIT_FAILURE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skvaider-maintenance",
        description=(
            "Move models away from an inference server before maintenance "
            "and let the proxy place them there again afterwards."
        ),
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    enter_parser = subparsers.add_parser(
        "enter", help="request maintenance mode and wait for the drain"
    )
    enter_parser.add_argument(
        "--url",
        required=True,
        help="base URL of the inference server, e.g. http://127.0.0.1:8000",
    )
    enter_parser.add_argument(
        "--wait",
        type=float,
        default=DEFAULT_WAIT_SECONDS,
        help="seconds to wait for the models to move away",
    )
    enter_parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help="seconds between maintenance phase polls",
    )
    enter_parser.add_argument(
        "--reason",
        default=None,
        help="reason recorded by the inference server",
    )

    leave_parser = subparsers.add_parser(
        "leave", help="leave maintenance mode again"
    )
    leave_parser.add_argument(
        "--url",
        required=True,
        help="base URL of the inference server, e.g. http://127.0.0.1:8000",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = MaintenanceClient(
        args.url,
        wait=getattr(args, "wait", DEFAULT_WAIT_SECONDS),
        poll_interval=getattr(args, "poll_interval", DEFAULT_POLL_INTERVAL),
        reason=getattr(args, "reason", None),
    )
    if args.action == "enter":
        return enter(client)
    return leave(client)


if __name__ == "__main__":
    sys.exit(main())
