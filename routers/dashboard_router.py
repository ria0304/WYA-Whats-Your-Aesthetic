from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime, timedelta
import json
import logging

from database import get_db
from auth_utils import get_current_user, UserProfile

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])
logger = logging.getLogger("uvicorn.error")


@router.get("/stats")
async def get_dashboard_stats(user: UserProfile = Depends(get_current_user)):
    """Get comprehensive dashboard statistics."""
    conn = get_db()
    try:
        # Wardrobe count
        wardrobe_count = conn.execute(
            "SELECT COUNT(*) FROM wardrobe_items WHERE user_id = ?",
            (user.user_id,)
        ).fetchone()[0]

        # Total wears
        total_wears = conn.execute(
            "SELECT SUM(wear_count) FROM wardrobe_items WHERE user_id = ?",
            (user.user_id,)
        ).fetchone()[0] or 0

        # Most worn item
        most_worn = conn.execute(
            "SELECT name, wear_count FROM wardrobe_items WHERE user_id = ? ORDER BY wear_count DESC LIMIT 1",
            (user.user_id,)
        ).fetchone()

        # Least worn item (with 0 wears)
        least_worn = conn.execute(
            "SELECT name, wear_count FROM wardrobe_items WHERE user_id = ? AND wear_count = 0 LIMIT 1",
            (user.user_id,)
        ).fetchone()

        # Items added in last 30 days
        thirty_days_ago = (datetime.utcnow() - timedelta(days=30)).isoformat()
        new_items = conn.execute(
            "SELECT COUNT(*) FROM wardrobe_items WHERE user_id = ? AND created_at >= ?",
            (user.user_id, thirty_days_ago)
        ).fetchone()[0]

        # Style DNA
        dna_row = conn.execute(
            "SELECT styles FROM style_dna WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
            (user.user_id,)
        ).fetchone()

        archetype = "Not set"
        if dna_row and dna_row["styles"]:
            try:
                styles = json.loads(dna_row["styles"])
                if styles:
                    archetype = styles[0].capitalize()
            except Exception:
                pass

        # Sustainability score
        sustainability = conn.execute(
            "SELECT AVG(sustainability_score) FROM wardrobe_items WHERE user_id = ? AND sustainability_score > 0",
            (user.user_id,)
        ).fetchone()[0] or 0

        # Gap analysis count
        gaps = conn.execute(
            "SELECT COUNT(*) FROM wardrobe_archive WHERE user_id = ? AND archive_reason = 'gap'",
            (user.user_id,)
        ).fetchone()[0]

        # Recent activity
        recent_activity = conn.execute(
            """SELECT 'wear' as type, created_at FROM wear_logs WHERE user_id = ? 
               UNION ALL
               SELECT 'feedback' as type, created_at FROM outfit_feedback WHERE user_id = ?
               ORDER BY created_at DESC LIMIT 5""",
            (user.user_id, user.user_id)
        ).fetchall()

        conn.close()

        return {
            "wardrobe_count": wardrobe_count,
            "total_wears": total_wears,
            "most_worn_item": {
                "name": most_worn["name"] if most_worn else None,
                "wear_count": most_worn["wear_count"] if most_worn else 0
            } if most_worn else None,
            "least_worn_item": {
                "name": least_worn["name"] if least_worn else None,
                "wear_count": least_worn["wear_count"] if least_worn else 0
            } if least_worn else None,
            "new_items_30_days": new_items,
            "style_archetype": archetype,
            "sustainability_score": round(float(sustainability), 1),
            "gap_items_count": gaps,
            "recent_activity": [dict(row) for row in recent_activity],
            "last_updated": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error("Dashboard stats failed — user=%s error=%s", user.user_id[:8], e)
        raise HTTPException(500, "Failed to fetch dashboard stats")
    finally:
        conn.close()


@router.get("/activity")
async def get_activity_timeline(
    user: UserProfile = Depends(get_current_user),
    days: int = 30
):
    """Get user activity timeline."""
    conn = get_db()
    try:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()

        wear_activity = conn.execute(
            """SELECT created_at, item_id, occasion, weather 
               FROM wear_logs 
               WHERE user_id = ? AND created_at >= ?
               ORDER BY created_at DESC""",
            (user.user_id, since)
        ).fetchall()

        feedback_activity = conn.execute(
            """SELECT created_at, action, outfit_id 
               FROM outfit_feedback 
               WHERE user_id = ? AND created_at >= ?
               ORDER BY created_at DESC""",
            (user.user_id, since)
        ).fetchall()

        conn.close()

        return {
            "wear_logs": [dict(row) for row in wear_activity],
            "feedback_logs": [dict(row) for row in feedback_activity],
            "total_wears": len(wear_activity),
            "total_feedback": len(feedback_activity),
            "period_days": days
        }
    except Exception as e:
        logger.error("Activity timeline failed — user=%s error=%s", user.user_id[:8], e)
        raise HTTPException(500, "Failed to fetch activity timeline")
    finally:
        conn.close()
