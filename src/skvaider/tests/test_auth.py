import base64
import json

import fastapi.exceptions
import pytest
import svcs
from argon2 import PasswordHasher
from fastapi.security import HTTPAuthorizationCredentials

from skvaider.auth import verify_token
from skvaider.conftest import DummyTokens

hasher = PasswordHasher()


async def test_verify_token_incorrect_syntax(services: svcs.Container):
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="unknown"
    )
    with pytest.raises(fastapi.exceptions.HTTPException) as e:
        await verify_token(credentials, services)
    assert e.value.status_code == 401


async def test_verify_token_unknown_user(services: svcs.Container):
    secret = "asdf"
    auth_token = base64.b64encode(
        json.dumps({"id": "user", "secret": secret}).encode("utf-8")
    ).decode("ascii")
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=auth_token
    )
    with pytest.raises(fastapi.exceptions.HTTPException) as e:
        await verify_token(credentials, services)
    assert e.value.status_code == 401


async def test_verify_token_incorrect_password(
    services: svcs.Container, token_db: DummyTokens
):
    secret = "asdf"
    token_db.data["user"] = {"secret_hash": hasher.hash(secret)}

    auth_token = base64.b64encode(
        json.dumps({"id": "user", "secret": secret + "wrong"}).encode("utf-8")
    ).decode("ascii")
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=auth_token
    )

    with pytest.raises(fastapi.exceptions.HTTPException) as e:
        await verify_token(credentials, services)
    assert e.value.status_code == 401


async def test_verify_token_correct_user_and_password(
    services: svcs.Container, token_db: DummyTokens
):
    secret = "asdf"
    token_db.data["user"] = {"secret_hash": hasher.hash(secret)}

    auth_token = base64.b64encode(
        json.dumps({"id": "user", "secret": "asdf"}).encode("utf-8")
    ).decode("ascii")
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=auth_token
    )
    await verify_token(credentials, services)
