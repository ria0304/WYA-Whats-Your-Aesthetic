from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from datetime import datetime, timedelta
import json
import logging

from database import get_db
from auth_utils import get_current_user, UserProfile
from schemas import OutfitCreate
import uuid

router = APIRouter(prefix="/api/outfits", tags=["outfits"])
logger = logging.getLogger("uvicorn.error")


@router.get("")
async def get_outfits(user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    outfits = conn.execute(
        "SELECT * FROM saved_outfits WHERE user_id = ? ORDER BY created_date DESC",
        (user.user_id,)
    ).fetchall()
    conn.close()

    result = []
    for outfit in outfits:
        try:
            items = json.loads(outfit['items_json']) if outfit['items_json'] else []
        except Exception:
            items = []
        result.append({
            "id": outfit['outfit_id'], "name": outfit['name'],
            "vibe": outfit['vibe'], "items": items,
            "created_date": outfit['created_date'],
            "worn_count": outfit['worn_count'], "last_worn": outfit['last_worn']
        })
    return result


@router.post("")
async def save_outfit(data: OutfitCreate, user: UserProfile = Depends(get_current_user)):
    outfit_id = str(uuid.uuid4())
    conn = get_db()
    now = datetime.utcnow().isoformat()
    conn.execute(
        """INSERT INTO saved_outfits
           (outfit_id, user_id, name, vibe, items_json, created_date)
           VALUES (?,?,?,?,?,?)""",
        (outfit_id, user.user_id, data.name, data.vibe,
         json.dumps([item.dict() for item in data.items]),
         data.created_date or now)
    )
    conn.commit()
    conn.close()
    return {"success": True, "id": outfit_id}


@router.delete("/{outfit_id}")
async def delete_outfit(outfit_id: str, user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    conn.execute(
        "DELETE FROM saved_outfits WHERE outfit_id = ? AND user_id = ?",
        (outfit_id, user.user_id)
    )
    conn.commit()
    conn.close()
    return {"success": True}


@router.post("/{outfit_id}/worn")
async def log_outfit_wear(outfit_id: str, data: Dict[str, Any], user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    worn_at = data.get('worn_at', datetime.utcnow().isoformat())

    outfit = conn.execute(
        "SELECT * FROM saved_outfits WHERE outfit_id = ? AND user_id = ?",
        (outfit_id, user.user_id)
    ).fetchone()

    if not outfit:
        conn.close()
        raise HTTPException(status_code=404, detail="Outfit not found")

    conn.execute(
        "INSERT INTO outfit_wear_history (outfit_id, user_id, worn_at) VALUES (?,?,?)",
        (outfit_id, user.user_id, worn_at)
    )
    conn.execute(
        "UPDATE saved_outfits SET worn_count = worn_count + 1, last_worn = ? WHERE outfit_id = ?",
        (worn_at, outfit_id)
    )
    conn.commit()
    conn.close()
    return {"success": True, "worn_at": worn_at}


@router.get("/{outfit_id}/wear-history")
async def get_outfit_wear_history(outfit_id: str, user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    history = conn.execute(
        "SELECT worn_at FROM outfit_wear_history WHERE outfit_id = ? AND user_id = ? ORDER BY worn_at DESC",
        (outfit_id, user.user_id)
    ).fetchall()
    conn.close()
    return [dict(row) for row in history]
