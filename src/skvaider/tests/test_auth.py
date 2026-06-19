import base64
import json
from typing import Callable

import fastapi.exceptions
import pytest
import svcs
from argon2 import PasswordHasher
from fastapi import Request
from fastapi.security import HTTPAuthorizationCredentials

from skvaider.auth import Cache, verify_token
from skvaider.conftest import DummyTokens

hasher = PasswordHasher()


class FakeClock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    clock = FakeClock()
    monkeypatch.setattr("skvaider.auth.time.time", lambda: clock.now)
    return clock


def test_cache_get_missing_key_raises_keyerror():
    cache = Cache()
    with pytest.raises(KeyError):
        cache["nope"]


def test_cache_add_then_get_returns_payload():
    cache = Cache()
    cache.add("token", "user-id")
    assert cache["token"] == "user-id"


def test_cache_add_overwrites_existing_entry():
    cache = Cache()
    cache.add("token", "first")
    cache.add("token", "second")
    assert cache["token"] == "second"


def test_cache_returns_payload_within_ttl(fake_clock: FakeClock):
    cache = Cache()
    cache.add("token", "user-id")
    fake_clock.advance(Cache.TTL - 1)
    assert cache["token"] == "user-id"


def test_cache_entry_valid_at_exact_expiry(fake_clock: FakeClock):
    # Expiry is checked with `expiry < now`, so an entry is still valid at
    # the exact moment it expires.
    cache = Cache()
    cache.add("token", "user-id")
    fake_clock.advance(Cache.TTL)
    assert cache["token"] == "user-id"


def test_cache_expired_entry_raises_keyerror(fake_clock: FakeClock):
    cache = Cache()
    cache.add("token", "user-id")
    fake_clock.advance(Cache.TTL + 1)
    with pytest.raises(KeyError):
        cache["token"]


def test_cache_expired_entry_is_evicted(fake_clock: FakeClock):
    cache = Cache()
    cache.add("token", "user-id")
    fake_clock.advance(Cache.TTL + 1)
    with pytest.raises(KeyError):
        cache["token"]
    assert "token" not in cache.cache


async def test_verify_token_incorrect_syntax(
    services: svcs.Container, mock_request_factory: Callable[..., Request]
):
    request = mock_request_factory()
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="unknown"
    )
    with pytest.raises(fastapi.exceptions.HTTPException) as e:
        await verify_token(request, credentials, services)
    assert e.value.status_code == 401


async def test_verify_token_unknown_user(
    services: svcs.Container, mock_request_factory: Callable[..., Request]
):
    request = mock_request_factory()
    secret = "asdf"
    auth_token = base64.b64encode(
        json.dumps({"id": "user", "secret": secret}).encode("utf-8")
    ).decode("ascii")
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=auth_token
    )
    with pytest.raises(fastapi.exceptions.HTTPException) as e:
        await verify_token(request, credentials, services)
    assert e.value.status_code == 401


async def test_verify_token_incorrect_password(
    services: svcs.Container,
    token_db: DummyTokens,
    mock_request_factory: Callable[..., Request],
):
    request = mock_request_factory()
    secret = "asdf"
    token_db.data["user"] = {"secret_hash": hasher.hash(secret)}

    auth_token = base64.b64encode(
        json.dumps({"id": "user", "secret": secret + "wrong"}).encode("utf-8")
    ).decode("ascii")
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=auth_token
    )

    with pytest.raises(fastapi.exceptions.HTTPException) as e:
        await verify_token(request, credentials, services)
    assert e.value.status_code == 401


async def test_verify_token_correct_user_and_password(
    services: svcs.Container,
    token_db: DummyTokens,
    mock_request_factory: Callable[..., Request],
):
    request = mock_request_factory()
    secret = "asdf"
    token_db.data["user"] = {"secret_hash": hasher.hash(secret)}

    auth_token = base64.b64encode(
        json.dumps({"id": "user", "secret": "asdf"}).encode("utf-8")
    ).decode("ascii")
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=auth_token
    )
    await verify_token(request, credentials, services)
    assert request.state.token_id == "user"
