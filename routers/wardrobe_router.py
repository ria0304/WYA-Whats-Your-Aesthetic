import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, Form

from database import get_db
from auth_utils import get_current_user, UserProfile
from ai_model import FashionAIModel
from logger import get_logger

router = APIRouter(prefix="/api/wardrobe", tags=["wardrobe"])
logger = get_logger(__name__)


@router.get("")
async def get_wardrobe(user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    try:
        items = conn.execute(
            "SELECT * FROM wardrobe_items WHERE user_id = ? ORDER BY created_at DESC",
            (user.user_id,)
        ).fetchall()
        logger.info("Wardrobe fetch — user=%s items=%d", user.user_id[:8], len(items))
        return [dict(row) for row in items]
    except Exception as e:
        logger.error("Wardrobe fetch failed — user=%s error=%s", user.user_id[:8], e)
        raise HTTPException(500, "Failed to fetch wardrobe")
    finally:
        conn.close()


@router.post("")
async def add_wardrobe_item(
    name: str = Form(...),
    category: str = Form(...),
    color: str = Form(""),
    fabric: str = Form(""),
    image_url: Optional[str] = Form(None),
    price: float = Form(0.0),
    brand: str = Form(""),
    sustainability_score: int = Form(0),
    user: UserProfile = Depends(get_current_user)
):
    item_id = str(uuid.uuid4())
    conn = get_db()
    try:
        now = datetime.utcnow().isoformat()
        conn.execute(
            """INSERT INTO wardrobe_items 
               (item_id, user_id, name, category, color, fabric, image_url, price, brand, sustainability_score, created_at) 
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (item_id, user.user_id, name, category, color, fabric, image_url, price, brand, sustainability_score, now)
        )
        conn.commit()

        item_dict = {"item_id": item_id, "name": name, "category": category, "color": color, "fabric": fabric}
        try:
            from ai_matcher import _text_to_pseudo_embedding
            from database import save_embedding
            emb = _text_to_pseudo_embedding(item_dict)
            save_embedding(conn, item_id, emb)
            conn.commit()
        except Exception as emb_exc:
            logger.warning("Embedding generation skipped — item=%s error=%s", item_id[:8], emb_exc)

        try:
            import embedding_store
            embedding_store.add_item(user.user_id, item_dict)
        except Exception as faiss_exc:
            logger.warning("FAISS add_item skipped — item=%s error=%s", item_id[:8], faiss_exc)

        logger.info("Wardrobe item added — user=%s item=%s category=%s", user.user_id[:8], item_id[:8], category)
        return {"success": True, "item_id": item_id}
    except Exception as e:
        logger.error("Add wardrobe item failed — user=%s error=%s", user.user_id[:8], e)
        raise HTTPException(500, "Failed to add item")
    finally:
        conn.close()


@router.delete("/{item_id}")
async def delete_wardrobe_item(item_id: str, user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    try:
        conn.execute(
            "DELETE FROM wardrobe_items WHERE item_id = ? AND user_id = ?",
            (item_id, user.user_id)
        )
        conn.commit()

        try:
            import embedding_store
            embedding_store.delete_index(user.user_id)
        except Exception as faiss_exc:
            logger.warning("FAISS delete_index skipped — user=%s error=%s", user.user_id[:8], faiss_exc)

        logger.info("Wardrobe item deleted — user=%s item=%s", user.user_id[:8], item_id[:8])
        return {"success": True}
    except Exception as e:
        logger.error("Delete wardrobe item failed — user=%s item=%s error=%s", user.user_id[:8], item_id[:8], e)
        raise HTTPException(500, "Failed to delete item")
    finally:
        conn.close()


@router.post("/{item_id}/wear")
async def wear_item(item_id: str, data: Dict[str, Any] = None, user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    try:
        worn_at = data.get('worn_at', datetime.utcnow().isoformat()) if data else datetime.utcnow().isoformat()
        occasion = data.get('occasion', None) if data else None
        weather = data.get('weather', None) if data else None
        temperature = data.get('temperature', None) if data else None
        time_of_day = data.get('time_of_day', None) if data else None

        conn.execute(
            """UPDATE wardrobe_items 
               SET wear_count = wear_count + 1, last_worn = ? 
               WHERE item_id = ? AND user_id = ?""",
            (worn_at, item_id, user.user_id)
        )
        conn.commit()

        from database import log_wear
        log_wear(conn, user.user_id, item_id, None, occasion, weather, temperature, time_of_day)
        conn.commit()

        logger.info("Item worn — user=%s item=%s", user.user_id[:8], item_id[:8])
        return {"success": True}
    except Exception as e:
        logger.error("Wear item failed — user=%s item=%s error=%s", user.user_id[:8], item_id[:8], e)
        raise HTTPException(500, "Failed to log wear")
    finally:
        conn.close()


@router.put("/{item_id}")
async def update_wardrobe_item(
    item_id: str,
    name: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    color: Optional[str] = Form(None),
    fabric: Optional[str] = Form(None),
    image_url: Optional[str] = Form(None),
    price: Optional[float] = Form(None),
    brand: Optional[str] = Form(None),
    sustainability_score: Optional[int] = Form(None),
    user: UserProfile = Depends(get_current_user)
):
    conn = get_db()
    try:
        fields, values = [], []
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
        if price is not None:
            fields.append("price = ?"); values.append(price)
        if brand is not None:
            fields.append("brand = ?"); values.append(brand)
        if sustainability_score is not None:
            fields.append("sustainability_score = ?"); values.append(sustainability_score)

        if fields:
            sql = f"UPDATE wardrobe_items SET {', '.join(fields)} WHERE item_id = ? AND user_id = ?"
            values += [item_id, user.user_id]
            conn.execute(sql, tuple(values))
            conn.commit()
            logger.info("Wardrobe item updated — user=%s item=%s fields=%s", user.user_id[:8], item_id[:8], fields)

        return {"success": True}
    except Exception as e:
        logger.error("Update wardrobe item failed — user=%s item=%s error=%s", user.user_id[:8], item_id[:8], e)
        raise HTTPException(500, "Failed to update item")
    finally:
        conn.close()


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
        logger.info("Background removal started — user=%s item=%s", user.user_id[:8], item_id[:8])
        start = time.perf_counter()

        result = await FashionAIModel.remove_background(image_url)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("Background removal done — user=%s item=%s time=%.1fms", user.user_id[:8], item_id[:8], elapsed_ms)

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

    except MemoryError:
        logger.error("Background removal OOM — image too large for t2.micro — user=%s item=%s", user.user_id[:8], item_id[:8])
        conn.close()
        return {"success": False, "bg_removed_url": image_url, "message": "Image too large to process. Try a smaller image."}
    except Exception as exc:
        logger.error("Background removal failed — user=%s item=%s error=%s", user.user_id[:8], item_id[:8], exc)
        conn.close()
        return {"success": False, "bg_removed_url": image_url, "message": f"Background removal failed: {exc}"}


@router.post("/{item_id}/archive")
async def archive_item(item_id: str, data: Dict[str, Any], user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM wardrobe_items WHERE item_id = ? AND user_id = ?",
            (item_id, user.user_id)
        ).fetchone()

        if not row:
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
        logger.info("Item archived — user=%s item=%s reason=%s", user.user_id[:8], item_id[:8], data.get('reason'))
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Archive item failed — user=%s item=%s error=%s", user.user_id[:8], item_id[:8], e)
        raise HTTPException(500, "Failed to archive item")
    finally:
        conn.close()


@router.get("/archive")
async def get_archive(user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    try:
        archived = conn.execute(
            "SELECT * FROM wardrobe_archive WHERE user_id = ? ORDER BY deleted_at DESC",
            (user.user_id,)
        ).fetchall()
        logger.info("Archive fetch — user=%s items=%d", user.user_id[:8], len(archived))
        return [{
            "id": item['item_id'], "name": item['name'], "category": item['category'],
            "color": item['color'], "fabric": item['fabric'], "image_url": item['image_url'],
            "wear_count": item['wear_count'], "archived_date": item['deleted_at'],
            "archive_reason": item['archive_reason'], "memory_note": item['memory_note']
        } for item in archived]
    except Exception as e:
        logger.error("Archive fetch failed — user=%s error=%s", user.user_id[:8], e)
        raise HTTPException(500, "Failed to fetch archive")
    finally:
        conn.close()


@router.delete("/archive/{item_id}")
async def permanent_delete_archive(item_id: str, user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    try:
        conn.execute(
            "DELETE FROM wardrobe_archive WHERE item_id = ? AND user_id = ?",
            (item_id, user.user_id)
        )
        conn.commit()
        logger.info("Archive item permanently deleted — user=%s item=%s", user.user_id[:8], item_id[:8])
        return {"success": True}
    except Exception as e:
        logger.error("Archive delete failed — user=%s item=%s error=%s", user.user_id[:8], item_id[:8], e)
        raise HTTPException(500, "Failed to delete archive item")
    finally:
        conn.close()


@router.get("/analytics")
async def get_wardrobe_analytics(user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    try:
        items = conn.execute(
            "SELECT * FROM wardrobe_items WHERE user_id = ?",
            (user.user_id,)
        ).fetchall()
        wardrobe_items = [dict(row) for row in items]

        wear_logs = conn.execute(
            "SELECT * FROM wear_logs WHERE user_id = ? ORDER BY created_at DESC",
            (user.user_id,)
        ).fetchall()
        wear_history = [dict(row) for row in wear_logs]

        from services.outfit_generator import OutfitGenerator
        generator = OutfitGenerator()
        analytics = generator.get_wardrobe_analytics(wardrobe_items, wear_history)

        return analytics
    except Exception as e:
        logger.error("Wardrobe analytics failed — user=%s error=%s", user.user_id[:8], e)
        raise HTTPException(500, "Failed to fetch wardrobe analytics")
    finally:
        conn.close()


@router.get("/analytics/evolution")
async def get_style_evolution(user: UserProfile = Depends(get_current_user)):
    conn = get_db()
    try:
        snapshots = conn.execute(
            "SELECT * FROM style_evolution WHERE user_id = ? ORDER BY snapshot_date ASC",
            (user.user_id,)
        ).fetchall()

        if not snapshots:
            return {
                "has_evolution": False,
                "message": "Not enough data to analyze evolution",
                "trajectory": [],
                "current": None
            }

        snapshot_list = [dict(row) for row in snapshots]
        from services.style_profile import style_profile
        return style_profile.analyze_evolution_over_time(snapshot_list)
    except Exception as e:
        logger.error("Style evolution failed — user=%s error=%s", user.user_id[:8], e)
        raise HTTPException(500, "Failed to fetch style evolution")
    finally:
        conn.close()
