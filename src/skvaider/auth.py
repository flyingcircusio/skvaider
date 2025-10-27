import base64
import binascii
import json
from json import JSONDecodeError
from typing import Annotated

import svcs
from argon2 import PasswordHasher
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import aramaki

_bearer_auth = HTTPBearer()

hasher = PasswordHasher()


class AuthTokens(aramaki.Collection):
    collection = "fc.directory.ai.token"


async def verify_token(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer_auth)],
    services: svcs.fastapi.DepContainer,
):
    authtokens = await services.aget(AuthTokens)

    try:
        client_token = json.loads(
            base64.b64decode(
                credentials.credentials.encode("utf-8"), validate=True
            ).decode("utf-8")
        )
    except (binascii.Error, ValueError, JSONDecodeError):
        raise HTTPException(401, detail="Bad authentication")

    db_token = await authtokens.get(client_token["id"])
    if not db_token:
        raise HTTPException(401, detail="Bad authentication")
    try:
        hasher.verify(db_token["secret_hash"], client_token["secret"])
    # We could specify explicit exceptions here but go the safe route and just catch all in case the lib addes one
    except Exception:
        raise HTTPException(401, detail="Bad authentication")
