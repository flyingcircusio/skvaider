import fastapi.exceptions
import pytest
from fastapi.security import HTTPAuthorizationCredentials

from skvaider.auth import verify_token
from skvaider.models import AuthToken


async def test_verify_token_incorrect_syntax(services):
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="unknown"
    )
    with pytest.raises(fastapi.exceptions.HTTPException) as e:
        await verify_token(credentials, services)
    assert e.value.status_code == 403


async def test_verify_token_unknown_user(services):
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="user-password"
    )
    with pytest.raises(fastapi.exceptions.HTTPException) as e:
        await verify_token(credentials, services)
    assert e.value.status_code == 403


async def test_verify_token_incorrect_password(services, session):
    await AuthToken.create(session, username="user", password="*")

    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="user-password"
    )

    with pytest.raises(fastapi.exceptions.HTTPException) as e:
        await verify_token(credentials, services)
    assert e.value.status_code == 403


async def test_verify_token_correct_user_and_password(services, session):
    await AuthToken.create(session, username="user", password="password")

    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="user-password"
    )
    await verify_token(credentials, services)
