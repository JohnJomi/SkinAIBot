"""Auth endpoints: register, login, me."""

from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from ....database import get_session
from ....models import User
from ....schemas import Token, UserCreate, UserResponse
from ....services import AuthService
from ...dependencies import get_current_user

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(payload: UserCreate, session: AsyncSession = Depends(get_session)):
    user = await AuthService(session).register(payload.email, payload.password)
    return user


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_session),
):
    token = await AuthService(session).login(form_data.username, form_data.password)
    return Token(access_token=token)


@router.get("/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)):
    return current_user
