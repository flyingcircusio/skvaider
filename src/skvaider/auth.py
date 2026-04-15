import base64
import binascii
import json
import time
from json import JSONDecodeError
from typing import Annotated, cast

import svcs
from argon2 import PasswordHasher
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import aramaki

_bearer_auth = HTTPBearer()

hasher = PasswordHasher()


class AuthTokens(aramaki.Collection):
    collection = "fc.directory.ai.token"


class StaticAuthTokens:
    """Tokens that can be provided through the config that aren't managed by Aramaki."""

    def __init__(self, tokens: list[str]):
        self.tokens = set(tokens)


class Cache:
    # XXX turn this into a feature of the collection
    # avoid hitting the session at all if we have a valid cache
    # Also allow caching negative results.

    TTL = 300

    def __init__(self):
        self.cache: dict[str, float] = {}

    def __contains__(self, key: str):
        now = time.time()
        if key not in self.cache:
            return False
        if self.cache[key] < now:
            del self.cache[key]
            return False
        return True

    def add(self, key: str):
        self.cache[key] = time.time() + self.TTL


cache = Cache()  # XXX turn into service


async def verify_token(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer_auth)],
    services: svcs.fastapi.DepContainer,
) -> None:
    if credentials.credentials in cache:
        return

    try:
        admin_tokens = services.get(StaticAuthTokens)
    except svcs.exceptions.ServiceNotFoundError:
        pass
    else:
        if credentials.credentials in admin_tokens.tokens:
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
        try:
            assert isinstance(db_token["secret_hash"], str)
            hasher.verify(db_token["secret_hash"], client_token["secret"])
            cache.add(credentials.credentials)
        # We could specify explicit exceptions here but go the safe route and just catch all in case the lib addes one
        except Exception:
            raise HTTPException(401, detail="Bad authentication")


async def verify_admin_token(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer_auth)],
    services: svcs.fastapi.DepContainer,
) -> None:
    """Accept only admin (static) tokens — not aramaki tokens."""
    try:
        admin_tokens = services.get(StaticAuthTokens)
    except svcs.exceptions.ServiceNotFoundError:
        raise HTTPException(401, detail="No admin tokens configured")
    if credentials.credentials not in admin_tokens.tokens:
        raise HTTPException(401, detail="Bad authentication")
