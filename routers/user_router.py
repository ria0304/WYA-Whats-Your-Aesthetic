from fastapi import APIRouter, Depends
from typing import Dict, Any
from datetime import datetime, timedelta
import json
import logging

from database import get_db
from auth_utils import get_current_user, UserProfile

router = APIRouter(prefix="/api/user", tags=["user"])
logger = logging.getLogger("uvicorn.error")


@router.get("/profile")
async def get_profile(user: UserProfile = Depends(get_current_user)):
    return user.dict()


@router.put("/profile")
async def update_profile(data: Dict[str, Any], user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    now = datetime.utcnow().isoformat()
    conn.execute(
        "UPDATE users SET full_name = ?, location = ?, birthday = ?, gender = ?, email_notifications = ?, updated_at = ? WHERE user_id = ?",
        (data.get('full_name', user.full_name), data.get('location', user.location),
         data.get('birthday', user.birthday), data.get('gender', user.gender),
         data.get('email_notifications', user.email_notifications), now, user.user_id)
    )
    conn.commit()
    user_row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user.user_id,)).fetchone()
    conn.close()
    return dict(user_row)


@router.get("/preferences")
async def get_preferences(user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    row = conn.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user.user_id,)).fetchone()
    conn.close()
    if row:
        return {"colors": json.loads(row['colors']), "brands": json.loads(row['brands'])}
    return {"colors": [], "brands": []}


@router.put("/preferences")
async def update_preferences(data: Dict[str, Any], user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO user_preferences (user_id, colors, brands, updated_at) VALUES (?,?,?,?)",
        (user.user_id, json.dumps(data.get('colors', [])), json.dumps(data.get('brands', [])), now)
    )
    conn.commit()
    conn.close()
    return {"success": True}


@router.get("/activity")
async def get_activity(user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    items = conn.execute(
        "SELECT name as item, 'Added Item' as action, created_at as date FROM wardrobe_items WHERE user_id = ? ORDER BY created_at DESC LIMIT 5",
        (user.user_id,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in items]


@router.get("/wear-timeline")
async def get_user_wear_timeline(days: int = 90, user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
    timeline = conn.execute(
        """SELECT owh.worn_at, owh.outfit_id, so.name as outfit_name, so.vibe, so.items_json
        FROM outfit_wear_history owh
        JOIN saved_outfits so ON owh.outfit_id = so.outfit_id
        WHERE owh.user_id = ? AND owh.worn_at >= ?
        ORDER BY owh.worn_at DESC""",
        (user.user_id, start_date)
    ).fetchall()
    conn.close()
    return [dict(row) for row in timeline]


# ── Notifications ──────────────────────────────────────────────────────────────

@router.post("/notifications/test-email")
async def test_email_notification(user: UserProfile = Depends(get_current_user)):
    from services.email_service import email_service, EMAIL_ENABLED
    if not EMAIL_ENABLED:
        return {
            "success": False,
            "message": "Email not configured. Set WYA_GMAIL_ADDRESS and WYA_GMAIL_APP_PASS environment variables.",
            "setup_guide": (
                "1. Enable 2FA on your Google account.\n"
                "2. Go to myaccount.google.com → Security → App passwords.\n"
                "3. Generate a password for 'WYA App'.\n"
                "4. Set WYA_GMAIL_ADDRESS=your@gmail.com and WYA_GMAIL_APP_PASS=xxxx in your .env file."
            ),
        }
    sent = await email_service.send_test(user.email)
    return {"success": sent, "sent_to": user.email}
