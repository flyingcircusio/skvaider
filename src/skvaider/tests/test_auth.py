import base64
import json

import fastapi.exceptions
import pytest
from argon2 import PasswordHasher
from fastapi.security import HTTPAuthorizationCredentials

from aramaki.collection import Record
from skvaider.auth import AuthTokens, verify_token

hasher = PasswordHasher()


async def test_verify_token_incorrect_syntax(services):
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="unknown"
    )
    with pytest.raises(fastapi.exceptions.HTTPException) as e:
        await verify_token(credentials, services)
    assert e.value.status_code == 401


async def test_verify_token_unknown_user(services):
    secret = "asdf"
    auth_token = base64.b64encode(
        json.dumps({"id": "user", "secret": secret}).encode("utf-8")
    )
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=auth_token
    )
    with pytest.raises(fastapi.exceptions.HTTPException) as e:
        await verify_token(credentials, services)
    assert e.value.status_code == 401


async def test_verify_token_incorrect_password(services):
    secret = "the-secret"
    authtokens = await services.aget(AuthTokens)

    async with authtokens.manager.aramaki.db.session() as session:
        await Record.create(
            session,
            collection=authtokens.collection,
            partition="p1",
            record_id="user",
            version="1",
            data=dict(secret_hash=hasher.hash(secret)),
        )

    auth_token = base64.b64encode(
        json.dumps({"id": "user", "secret": secret + "wrong"}).encode("utf-8")
    )
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=auth_token
    )

    with pytest.raises(fastapi.exceptions.HTTPException) as e:
        await verify_token(credentials, services)
    assert e.value.status_code == 401


async def test_verify_token_correct_user_and_password(services):
    secret = "another-secret"

    authtokens = await services.aget(AuthTokens)

    async with authtokens.manager.aramaki.db.session() as session:
        await Record.create(
            session,
            collection=authtokens.collection,
            partition="p1",
            record_id="user",
            version="1",
            data=dict(secret_hash=hasher.hash(secret)),
        )

    auth_token = base64.b64encode(
        json.dumps({"id": "user", "secret": secret}).encode("utf-8")
    )
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=auth_token
    )
    await verify_token(credentials, services)
