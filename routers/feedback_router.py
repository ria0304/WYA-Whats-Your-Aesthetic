from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime
import json
import logging

from database import get_db
from auth_utils import get_current_user, UserProfile
from schemas import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/api/feedback", tags=["feedback"])
logger = logging.getLogger("uvicorn.error")


@router.post("")
async def save_feedback(
    data: FeedbackRequest,
    user: UserProfile = Depends(get_current_user)
) -> FeedbackResponse:
    """
    Save user feedback for outfit suggestions.
    Actions: like, dislike, save, wear, skip
    """
    conn = get_db()
    try:
        conn.execute("""
            INSERT INTO outfit_feedback 
            (user_id, outfit_id, item_id, action, context, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user.user_id,
            data.outfit_id,
            data.item_id,
            data.action,
            json.dumps(data.context),
            datetime.utcnow().isoformat()
        ))
        conn.commit()

        logger.info("Feedback saved — user=%s action=%s", user.user_id[:8], data.action)
        return FeedbackResponse(
            status="success",
            message=f"Feedback '{data.action}' recorded",
            action=data.action
        )
    except Exception as e:
        logger.error("Feedback save failed — user=%s error=%s", user.user_id[:8], e)
        raise HTTPException(500, "Failed to save feedback")
    finally:
        conn.close()


@router.get("/history")
async def get_feedback_history(
    user: UserProfile = Depends(get_current_user),
    limit: int = 50
):
    """Get user's feedback history."""
    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT * FROM outfit_feedback 
               WHERE user_id = ? 
               ORDER BY created_at DESC 
               LIMIT ?""",
            (user.user_id, limit)
        ).fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error("Feedback history failed — user=%s error=%s", user.user_id[:8], e)
        raise HTTPException(500, "Failed to fetch feedback history")
    finally:
        conn.close()


@router.get("/stats")
async def get_feedback_stats(
    user: UserProfile = Depends(get_current_user)
):
    """Get feedback statistics for the user."""
    conn = get_db()
    try:
        likes = conn.execute(
            "SELECT COUNT(*) FROM outfit_feedback WHERE user_id = ? AND action = 'like'",
            (user.user_id,)
        ).fetchone()[0]

        dislikes = conn.execute(
            "SELECT COUNT(*) FROM outfit_feedback WHERE user_id = ? AND action = 'dislike'",
            (user.user_id,)
        ).fetchone()[0]

        saves = conn.execute(
            "SELECT COUNT(*) FROM outfit_feedback WHERE user_id = ? AND action = 'save'",
            (user.user_id,)
        ).fetchone()[0]

        wears = conn.execute(
            "SELECT COUNT(*) FROM outfit_feedback WHERE user_id = ? AND action = 'wear'",
            (user.user_id,)
        ).fetchone()[0]

        total = likes + dislikes + saves + wears
        satisfaction_rate = round((likes / max(likes + dislikes, 1)) * 100, 1)

        return {
            "total_feedback": total,
            "likes": likes,
            "dislikes": dislikes,
            "saves": saves,
            "wears": wears,
            "satisfaction_rate": satisfaction_rate
        }
    except Exception as e:
        logger.error("Feedback stats failed — user=%s error=%s", user.user_id[:8], e)
        raise HTTPException(500, "Failed to fetch feedback stats")
    finally:
        conn.close()
