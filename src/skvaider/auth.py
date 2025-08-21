from typing import Annotated

import svcs
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from skvaider.db import DBSession
from skvaider.models import AuthToken

_bearer_auth = HTTPBearer()


async def verify_token(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer_auth)],
    services: svcs.fastapi.DepContainer,
):
    db_session = await services.aget(DBSession)
    token = credentials.credentials
    try:
        username, password = token.split("-", 1)
    except ValueError:
        raise HTTPException(403, detail="Not authenticated")
    token_obj = await db_session.get(AuthToken, username)
    if token_obj is None:
        # TODO: Add hash calculation here to prevent timing attacks
        raise HTTPException(403, detail="Not authenticated")
    # TODO: Implement hashing function
    if token_obj.password != password:
        raise HTTPException(403, detail="Not authenticated")
