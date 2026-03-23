
import os
import jwt
import bcrypt
import logging
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from database import get_db
from pydantic import BaseModel, EmailStr
from typing import Optional

logger = logging.getLogger(__name__)

SECRET_KEY = os.getenv("JWT_SECRET", "wya-fashion-secret-key-2024")
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

class UserProfile(BaseModel):
    user_id: str
    email: str
    full_name: str
    birthday: Optional[str] = None
    gender: str
    location: str
    email_notifications: Optional[int] = 1
    created_at: Optional[str] = None

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(user_id: str):
    expire = datetime.utcnow() + timedelta(days=30)
    return jwt.encode({"sub": user_id, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # Handle cases where frontend might send string "null" or "undefined"
    if not token or token in ["null", "undefined"]:
        logger.warning("No valid token provided.")
        raise HTTPException(status_code=401, detail="Unauthorized: No token provided")

    # Handle development mock-token
    if token == "mock-token":
        user_id = "mock-user"
    else:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id = payload.get("sub")
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid token payload")
        except Exception as e:
            logger.error(f"JWT Decode error: {e}")
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

    conn = get_db()
    # Auto-create mock user if it doesn't exist for dev convenience
    if user_id == "mock-user":
        user = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if not user:
            conn.execute("INSERT INTO users (user_id, email, full_name, birthday, gender, location, hashed_password) VALUES (?,?,?,?,?,?,?)",
                         ("mock-user", "guest@example.com", "Guest User", "2000-01-01", "Female", "New York", "nopass"))
            conn.commit()
            user = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    else:
        user = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    
    conn.close()
    
    if not user:
        raise HTTPException(status_code=401, detail="User account not found")
    
    return UserProfile(**dict(user))

