from fastapi import APIRouter, HTTPException, Depends, Form
from typing import Optional, Dict, Any
from datetime import datetime
import uuid
import logging

from database import get_db
from auth_utils import get_current_user, UserProfile
from ai_model import FashionAIModel

router = APIRouter(prefix="/api/wardrobe", tags=["wardrobe"])
logger = logging.getLogger("uvicorn.error")


@router.get("")
async def get_wardrobe(user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    items = conn.execute(
        "SELECT * FROM wardrobe_items WHERE user_id = ? ORDER BY created_at DESC",
        (user.user_id,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in items]


@router.post("")
async def add_wardrobe_item(
    name: str = Form(...),
    category: str = Form(...),
    color: str = Form(""),
    fabric: str = Form(""),
    image_url: Optional[str] = Form(None),
    user: UserProfile = Depends(get_current_user)
):
    item_id = str(uuid.uuid4())
    conn = get_db()
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO wardrobe_items (item_id, user_id, name, category, color, fabric, image_url, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (item_id, user.user_id, name, category, color, fabric, image_url, now)
    )
    conn.commit()
    conn.close()
    return {"success": True}


@router.delete("/{item_id}")
async def delete_wardrobe_item(item_id: str, user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    conn.execute("DELETE FROM wardrobe_items WHERE item_id = ? AND user_id = ?", (item_id, user.user_id))
    conn.commit()
    conn.close()
    return {"success": True}


@router.post("/{item_id}/wear")
async def wear_item(item_id: str, data: Dict[str, Any] = None, user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    worn_at = data.get('worn_at', datetime.utcnow().isoformat()) if data else datetime.utcnow().isoformat()
    conn.execute(
        "UPDATE wardrobe_items SET wear_count = wear_count + 1, last_worn = ? WHERE item_id = ? AND user_id = ?",
        (worn_at, item_id, user.user_id)
    )
    conn.commit()
    conn.close()
    return {"success": True}


@router.put("/{item_id}")
async def update_wardrobe_item(
    item_id: str,
    name: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    color: Optional[str] = Form(None),
    fabric: Optional[str] = Form(None),
    image_url: Optional[str] = Form(None),
    user: UserProfile = Depends(get_current_user)
):
    conn = get_db()
    fields = []
    values = []
    if name is not None:
        fields.append("name = ?"); values.append(name)
    if category is not None:
        fields.append("category = ?"); values.append(category)
    if color is not None:
        fields.append("color = ?"); values.append(color)
    if fabric is not None:
        fields.append("fabric = ?"); values.append(fabric)
    if image_url is not None:
        fields.append("image_url = ?"); values.append(image_url)

    if fields:
        sql = f"UPDATE wardrobe_items SET {', '.join(fields)} WHERE item_id = ? AND user_id = ?"
        values += [item_id, user.user_id]
        conn.execute(sql, tuple(values))
        conn.commit()

    conn.close()
    return {"success": True}


@router.post("/{item_id}/remove-bg")
async def remove_background(item_id: str, user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    item = conn.execute(
        "SELECT * FROM wardrobe_items WHERE item_id = ? AND user_id = ?",
        (item_id, user.user_id)
    ).fetchone()

    if not item:
        conn.close()
        raise HTTPException(404, "Item not found")

    image_url: str = item["image_url"] or ""

    if not image_url.startswith("data:image"):
        conn.close()
        return {
            "success": False,
            "bg_removed_url": image_url,
            "message": "Background removal requires an uploaded image, not a hosted URL.",
        }

    try:
        result = await FashionAIModel.remove_background(image_url)

        if result.get("success") and result.get("bg_removed_image"):
            bg_removed_data = f"data:image/png;base64,{result['bg_removed_image']}"
            conn.execute(
                "UPDATE wardrobe_items SET image_url = ? WHERE item_id = ? AND user_id = ?",
                (bg_removed_data, item_id, user.user_id)
            )
            conn.commit()
            conn.close()
            return {
                "success": True,
                "bg_removed_url": bg_removed_data,
                "message": "Background removed — your item now looks like a lookbook photo.",
            }

        conn.close()
        return {
            "success": False,
            "bg_removed_url": image_url,
            "message": result.get("error", "Background removal failed — original image kept."),
        }

    except Exception as exc:
        logger.error("remove-bg endpoint error: %s", exc)
        conn.close()
        return {"success": False, "bg_removed_url": image_url, "message": f"Background removal failed: {exc}"}


@router.post("/{item_id}/archive")
async def archive_item(item_id: str, data: Dict[str, Any], user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM wardrobe_items WHERE item_id = ? AND user_id = ?",
        (item_id, user.user_id)
    ).fetchone()

    if not row:
        conn.close()
        raise HTTPException(404, "Item not found")

    item = dict(row)
    now = datetime.utcnow().isoformat()
    conn.execute(
        """INSERT INTO wardrobe_archive
           (item_id, user_id, name, category, color, fabric, brand, image_url, wear_count, created_at, deleted_at, archive_reason, memory_note)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (item_id, user.user_id, item['name'], item['category'], item['color'],
         item['fabric'], item.get('brand', ''), item['image_url'],
         item.get('wear_count', 0), item['created_at'], now,
         data.get('reason', 'sold'), data.get('memory_note', ''))
    )
    conn.execute("DELETE FROM wardrobe_items WHERE item_id = ? AND user_id = ?", (item_id, user.user_id))
    conn.commit()
    conn.close()
    return {"success": True}


# ── Archive ────────────────────────────────────────────────────────────────────

@router.get("/archive")
async def get_archive(user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    archived = conn.execute(
        "SELECT * FROM wardrobe_archive WHERE user_id = ? ORDER BY deleted_at DESC",
        (user.user_id,)
    ).fetchall()
    conn.close()
    return [{
        "id": item['item_id'], "name": item['name'], "category": item['category'],
        "color": item['color'], "fabric": item['fabric'], "image_url": item['image_url'],
        "wear_count": item['wear_count'], "archived_date": item['deleted_at'],
        "archive_reason": item['archive_reason'], "memory_note": item['memory_note']
    } for item in archived]


@router.delete("/archive/{item_id}")
async def permanent_delete_archive(item_id: str, user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    conn.execute("DELETE FROM wardrobe_archive WHERE item_id = ? AND user_id = ?", (item_id, user.user_id))
    conn.commit()
    conn.close()
    return {"success": True}
