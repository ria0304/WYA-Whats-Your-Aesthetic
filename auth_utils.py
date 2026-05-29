# auth_utils.py
import os
import bcrypt
import logging
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import jwt
from jwt.exceptions import InvalidTokenError
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Optional

from database import get_db

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"
TOKEN_EXPIRE_DAYS = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


# ── Secret resolution (cached so AWS SM is not hit every request) ─────────────

@lru_cache(maxsize=1)
def _load_jwt_secret() -> str:
    """
    Resolve JWT secret once and cache it for the process lifetime.
    Order of precedence:
      1. AWS Secrets Manager  (if USE_SECRETS_MANAGER=true)
      2. JWT_SECRET env var
      3. Raise — never fall back to a hardcoded string in production
    """
    use_sm = os.getenv("USE_SECRETS_MANAGER", "false").lower() == "true"

    if use_sm:
        try:
            from services.secrets_manager import get_jwt_secret
            secret = get_jwt_secret()
            if secret:
                logger.info("JWT secret loaded from AWS Secrets Manager")
                return secret
            logger.warning("Secrets Manager returned empty secret; falling back to env")
        except Exception as exc:
            logger.error("Secrets Manager unavailable: %s", exc)
            raise RuntimeError("Could not load JWT secret from Secrets Manager") from exc

    secret = os.getenv("JWT_SECRET", "")
    if not secret:
        raise RuntimeError(
            "JWT_SECRET env var is not set. "
            "Set it in .env (dev) or via Secrets Manager (prod)."
        )
    return secret


def get_jwt_secret() -> str:
    return _load_jwt_secret()


# ── Models ────────────────────────────────────────────────────────────────────

class UserProfile(BaseModel):
    user_id: str
    email: str
    full_name: str
    birthday: Optional[str] = None
    gender: str
    location: str
    email_notifications: Optional[int] = 1
    created_at: Optional[str] = None


# ── Password helpers ──────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ── Token creation ────────────────────────────────────────────────────────────

def create_access_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=TOKEN_EXPIRE_DAYS)
    return jwt.encode(
        {"sub": user_id, "exp": expire},
        get_jwt_secret(),
        algorithm=ALGORITHM,
    )


# ── Token verification & user resolution ─────────────────────────────────────

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserProfile:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token or token in ("null", "undefined"):
        raise credentials_error

    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if not user_id:
            raise credentials_error
    except InvalidTokenError as exc:
        logger.warning("JWT decode failed: %s", exc)
        raise credentials_error

    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found",
        )

    return UserProfile(**dict(row))
