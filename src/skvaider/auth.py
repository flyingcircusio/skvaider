from typing import Annotated

import svcs
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import aramaki

_bearer_auth = HTTPBearer()


class AuthTokens(aramaki.Collection):
    collection = "fc.directory.ai.tokens"


async def verify_token(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer_auth)],
    services: svcs.fastapi.DepContainer,
):
    authtokens = await services.aget(AuthTokens)
    token = credentials.credentials
    try:
        username, password = token.split("-", 1)
    except ValueError:
        raise HTTPException(403, detail="Not authenticated")
    token_obj = await authtokens.get(username)
    if token_obj is None:
        # TODO: Add hash calculation here to prevent timing attacks
        raise HTTPException(403, detail="Not authenticated")
    # TODO: Implement hashing function
    if token_obj["password"] != password:
        raise HTTPException(403, detail="Not authenticated")
