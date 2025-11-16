from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, EmailStr
from pymongo.collection import Collection
from bson import ObjectId

from db.mongo import get_users_collection, get_tokens_collection
from core.security import hash_password, verify_password, token_pair

router = APIRouter(prefix="/auth", tags=["auth"])

class RegisterBody(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class LoginBody(BaseModel):
    email: EmailStr
    password: str

def _users(req: Request) -> Collection:
    return get_users_collection(req.app)

def _tokens(req: Request) -> Collection:
    return get_tokens_collection(req.app)

@router.post("/register")
def register(body: RegisterBody, users: Collection = Depends(_users)):
    if users.find_one({"email": body.email.lower()}):
        raise HTTPException(409, "email already registered")
    doc = {
        "_id": str(ObjectId()),
        "email": body.email.lower(),
        "password": hash_password(body.password),
        "full_name": body.name or "",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    users.insert_one(doc)
    return {"ok": True}

@router.post("/login")
def login(body: LoginBody, response: Response,
          users: Collection = Depends(_users),
          tokens: Collection = Depends(_tokens)):
    u = users.find_one({"email": body.email.lower()})
    if not u or not verify_password(body.password, u["password"]):
        raise HTTPException(401, "invalid credentials")
    tok, exp = token_pair(u["_id"])
    tokens.insert_one({
        "token": tok,
        "user_id": u["_id"],
        "created_at": datetime.utcnow(),
        "expires_at": exp
    })
    # два варианта: вернуть токен и/или положить httpOnly-cookie
    response.set_cookie(
        key="auth_token", value=tok, httponly=True, secure=False, samesite="lax", max_age=7*24*3600
    )
    return {"token": tok, "user_id": u["_id"], "expires_at": exp}

@router.post("/logout")
def logout(request: Request, tokens: Collection = Depends(_tokens)):
    tok = request.cookies.get("auth_token")
    if tok:
        tokens.delete_one({"token": tok})
    return {"ok": True}


def _extract_token_from_request(request: Request) -> str | None:
    """
    Пытаемся вытащить токен из:
    - заголовка Authorization: Bearer <token>
    - заголовка X-Auth-Token
    - cookie auth_token
    """
    # 1) Authorization: Bearer xxx
    auth_header = request.headers.get("Authorization")
    if auth_header:
        parts = auth_header.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]

    # 2) X-Auth-Token
    x_token = request.headers.get("X-Auth-Token")
    if x_token:
        return x_token

    # 3) cookie
    cookie_token = request.cookies.get("auth_token")
    if cookie_token:
        return cookie_token

    return None

def get_current_user(
    request: Request,
    users: Collection = Depends(_users),
    tokens: Collection = Depends(_tokens),
) -> dict:
    """
    Достаёт пользователя по токену.
    Если токен невалиден/не найден/просрочен — кидает 401.
    """
    tok = _extract_token_from_request(request)
    if not tok:
        raise HTTPException(401, "auth required")

    tk = tokens.find_one({"token": tok})
    if not tk:
        raise HTTPException(401, "invalid token")

    u = users.find_one({"_id": tk["user_id"]})
    if not u:
        raise HTTPException(401, "user not found")

    return {
        "_id": u["_id"],
        "email": u["email"],
        "full_name": u.get("full_name", ""),
    }

@router.get("/check")
def auth_check(current = Depends(get_current_user)):
    """
    Проверка авторизации.
    Клиент отправляет Bearer-токен (или cookie), сервер:
    - если ок — возвращает данные пользователя и ok: true
    - если нет — 401 из get_current_user
    """
    return {
        "ok": True,
        "user": {
            "id": current["_id"],
            "email": current["email"],
            "full_name": current.get("full_name", ""),
        },
    }