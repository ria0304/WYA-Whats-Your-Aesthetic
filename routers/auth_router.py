from fastapi import APIRouter, HTTPException, Request
from datetime import datetime
import uuid
import logging

from database import get_db
from auth_utils import hash_password, verify_password, create_access_token
from schemas import UserRegister, UserLogin
from rate_limiter import limiter

router = APIRouter(prefix="/api/auth", tags=["auth"])
logger = logging.getLogger("uvicorn.error")


@router.post("/register")
@limiter.limit("3/minute")
async def register(request: Request, data: UserRegister):
    conn = get_db()
    try:
        user_id = str(uuid.uuid4())
        hashed = hash_password(data.password)
        now = datetime.utcnow().isoformat()
        conn.execute(
            "INSERT INTO users (user_id, email, full_name, birthday, gender, location, hashed_password, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (user_id, data.email, data.full_name, data.birthday or '', data.gender or 'Female', data.location or 'Global', hashed, now)
        )
        conn.commit()
        token = create_access_token(user_id)
        user_row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return {"access_token": token, "user": dict(user_row)}
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=400, detail="Registration failed.")
    finally:
        conn.close()


@router.post("/login")
@limiter.limit("5/minute")
async def login(request: Request, credentials: UserLogin):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (credentials.email,)).fetchone()
    conn.close()
    if not user or not verify_password(credentials.password, user['hashed_password']):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(user['user_id'])
    return {"access_token": token, "user": dict(user)}
