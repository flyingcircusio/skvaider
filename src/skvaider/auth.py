import asyncio
import base64
import binascii
import concurrent.futures
import json
import multiprocessing
import time
from json import JSONDecodeError
from typing import Annotated, Any, cast

import svcs
from argon2 import PasswordHasher
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import aramaki

_bearer_auth = HTTPBearer()

hasher = PasswordHasher()

VERIFY_POOL_WORKERS = 10

# argon2 verification is CPU-bound and holds the GIL, so running it on the
# event loop (or in a thread) stalls every other request on the worker. Offload
# it to a process pool, which has independent GILs. "spawn" avoids forking a
# process that already runs the asyncio loop and aiosqlite threads.
verify_pool: concurrent.futures.ProcessPoolExecutor | None = None


def start_verify_pool() -> None:
    global verify_pool
    verify_pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=VERIFY_POOL_WORKERS,
        mp_context=multiprocessing.get_context("spawn"),
    )


def stop_verify_pool() -> None:
    global verify_pool
    if verify_pool is None:
        return
    verify_pool.shutdown(wait=False, cancel_futures=True)
    verify_pool = None


def verify_password(secret_hash: str, secret: str) -> bool:
    # Runs in a pool worker process. Catch everything (not just mismatch) and
    # report failure, so an unexpected library error can't leak into the caller
    # as anything other than a rejected token.
    try:
        hasher.verify(secret_hash, secret)
        return True
    except Exception:
        return False


class AuthTokens(aramaki.Collection):
    collection = "fc.directory.ai.token"


class AdminTokens:
    """Tokens that can be provided through the config that aren't managed by Aramaki."""

    def __init__(self, tokens: list[str]):
        self.tokens = set(tokens)


class Cache:
    # XXX turn this into a feature of the collection
    # avoid hitting the session at all if we have a valid cache
    # Also allow caching negative results.

    TTL = 300

    def __init__(self):
        self.cache: dict[str, tuple[float, Any]] = {}

    def __getitem__(self, key: str):
        now = time.time()
        if key not in self.cache:
            raise KeyError(key)
        expiry, payload = self.cache[key]
        if expiry < now:
            del self.cache[key]
            raise KeyError(key)
        return payload

    def add(self, key: str, payload: Any):
        self.cache[key] = (time.time() + self.TTL, payload)


cache = Cache()  # XXX turn into service


async def verify_token(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer_auth)],
    services: svcs.fastapi.DepContainer,
) -> None:
    try:
        token_id = cache[credentials.credentials]
    except KeyError:
        pass
    else:
        request.state.token_id = token_id

    try:
        admin_tokens = services.get(AdminTokens)
    except svcs.exceptions.ServiceNotFoundError:
        pass
    else:
        if credentials.credentials in admin_tokens.tokens:
            request.state.token_id = "<admin-token>"
            return

    # XXX There's a lot of type issues going on here, because the mechanics of passing through
    # the correct types from services.aget() with a factory for a context manager that isn't entered
    # are just too hard for the type system for now afaict.
    try:
        client_token: dict[str, str] = json.loads(
            base64.b64decode(
                credentials.credentials.encode("utf-8"), validate=True
            ).decode("utf-8")
        )
    except (binascii.Error, ValueError, JSONDecodeError):
        raise HTTPException(401, detail="Bad authentication")
    async with await services.aget(AuthTokens) as authtokens:  # pyright: ignore[reportGeneralTypeIssues, reportUnknownVariableType]
        # Keep the DB access session scope confined to this part of the request,
        # otherwise DB sessions stay open while waiting for responses.
        # See PL-135110.
        db_token = cast(
            dict[str, str],
            await authtokens.get(client_token["id"]),  # pyright: ignore[reportUnknownMemberType]
        )
        if not db_token:
            raise HTTPException(401, detail="Bad authentication")
        assert isinstance(db_token["secret_hash"], str)
        loop = asyncio.get_running_loop()
        # verify_pool is None outside a running app (e.g. tests); run_in_executor
        # then falls back to the loop's default thread executor.
        valid = await loop.run_in_executor(
            verify_pool,
            verify_password,
            db_token["secret_hash"],
            client_token["secret"],
        )
        if not valid:
            raise HTTPException(401, detail="Bad authentication")
        request.state.token_id = client_token["id"]
        cache.add(credentials.credentials, client_token["id"])


async def verify_admin_token(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer_auth)],
    services: svcs.fastapi.DepContainer,
) -> None:
    """Accept only admin (static) tokens — not aramaki tokens."""
    try:
        admin_tokens = services.get(AdminTokens)
    except svcs.exceptions.ServiceNotFoundError:
        raise HTTPException(401, detail="No admin tokens configured")
    if credentials.credentials not in admin_tokens.tokens:
        raise HTTPException(401, detail="Bad authentication")
    request.state.token_id = "<admin-token>"
