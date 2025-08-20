from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from skvaider.db import DBSessionDep
from skvaider.models import AuthToken

router = APIRouter()


class AddTokenRequest(BaseModel):
    username: str
    password: str


@router.put("/token")
async def add_token(db_session: DBSessionDep, data: AddTokenRequest):
    token = await db_session.get(AuthToken, data.username)
    if token is not None:
        raise HTTPException(status_code=400, detail="Username already exists")
    await AuthToken.create(
        db_session, username=data.username, password=data.password
    )
    return {"detail": "Added"}


class DeleteTokenRequest(BaseModel):
    username: str


@router.delete("/token")
async def delete_token(db_session: DBSessionDep, data: DeleteTokenRequest):
    token = await db_session.get(AuthToken, data.username)
    if token is not None:
        await db_session.delete(token)
    return {"detail": "Deleted"}
