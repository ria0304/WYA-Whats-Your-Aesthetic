# services/notification_service.py
# Push notification service using pywebpush (Web Push Protocol / VAPID).
# Reads VAPID keys from environment variables; gracefully degrades if not set.

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional pywebpush dependency
# ---------------------------------------------------------------------------
try:
    from pywebpush import webpush, WebPushException
    WEBPUSH_AVAILABLE = True
except ImportError:
    WEBPUSH_AVAILABLE = False
    logger.warning(
        "pywebpush not installed — push notifications disabled. "
        "Run: pip install pywebpush"
    )

# ---------------------------------------------------------------------------
# VAPID configuration
# Read from env: WYA_VAPID_PRIVATE_KEY, WYA_VAPID_PUBLIC_KEY, WYA_VAPID_CLAIMS_EMAIL
# Generate keys once with: python -c "from py_vapid import Vapid; v=Vapid(); v.generate_keys(); print(v.private_pem().decode()); print(v.public_key.public_bytes_raw().hex())"
# ---------------------------------------------------------------------------
VAPID_PRIVATE_KEY: Optional[str] = os.getenv("WYA_VAPID_PRIVATE_KEY")
VAPID_PUBLIC_KEY: Optional[str]  = os.getenv("WYA_VAPID_PUBLIC_KEY")
VAPID_CLAIMS_EMAIL: str           = os.getenv("WYA_VAPID_CLAIMS_EMAIL", "mailto:hello@wya.app")

VAPID_ENABLED = bool(VAPID_PRIVATE_KEY and VAPID_PUBLIC_KEY and WEBPUSH_AVAILABLE)
if not VAPID_ENABLED:
    logger.warning(
        "VAPID keys not configured — set WYA_VAPID_PRIVATE_KEY and WYA_VAPID_PUBLIC_KEY "
        "env vars to enable real push notifications."
    )


class NotificationService:
    """
    Handles Web Push notifications for WYA.

    Flow:
      1. Frontend calls /api/notifications/subscribe with the PushSubscription JSON
         (endpoint, p256dh, auth).  Stored in push_subscriptions table.
      2. Backend calls send_daily_drop_notification() on a schedule (cron / APScheduler)
         to push a notification to all subscribed users.
    """

    @staticmethod
    async def save_subscription(
        user_id: str,
        subscription_data: Dict[str, Any],
        db_conn,
    ) -> bool:
        """Persist a PushSubscription from the browser to the DB."""
        try:
            now = datetime.utcnow().isoformat()
            db_conn.execute(
                """INSERT OR REPLACE INTO push_subscriptions
                   (user_id, endpoint, p256dh, auth, created_at, updated_at)
                   VALUES (?,?,?,?,?,?)""",
                (
                    user_id,
                    subscription_data.get("endpoint", ""),
                    subscription_data.get("keys", {}).get("p256dh", ""),
                    subscription_data.get("keys", {}).get("auth", ""),
                    now,
                    now,
                ),
            )
            db_conn.commit()
            logger.info("Saved push subscription for user %s", user_id)
            return True
        except Exception as exc:
            logger.error("Failed to save push subscription: %s", exc)
            return False

    @staticmethod
    async def send_daily_drop_notification(
        user_id: str,
        outfit_name: str,
        db_conn,
    ) -> bool:
        """
        Send a Web Push notification for the Daily Drop to a single user.
        Returns True if sent (or gracefully skipped), False on hard failure.
        """
        try:
            row = db_conn.execute(
                "SELECT endpoint, p256dh, auth FROM push_subscriptions WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        except Exception as exc:
            logger.error("DB error fetching push subscription: %s", exc)
            return False

        if not row:
            logger.debug("No push subscription for user %s — skipping", user_id)
            return True  # Not an error — user simply hasn't opted in

        if not VAPID_ENABLED:
            logger.info(
                "[DRY RUN] Would push 'Daily Drop ready: %s' to user %s",
                outfit_name,
                user_id,
            )
            return True

        subscription_info = {
            "endpoint": row["endpoint"],
            "keys": {"p256dh": row["p256dh"], "auth": row["auth"]},
        }
        payload = json.dumps({
            "title": "WYA — Daily Drop 🌅",
            "body": f"Your AI outfit for today is ready: {outfit_name}",
            "icon": "/icon.png",
            "badge": "/badge.png",
            "data": {"url": "/daily-drop"},
        })

        try:
            webpush(
                subscription_info=subscription_info,
                data=payload,
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims={"sub": VAPID_CLAIMS_EMAIL},
            )
            logger.info("Push sent to user %s: %s", user_id, outfit_name)
            return True
        except WebPushException as exc:
            # 410 Gone = subscription expired; remove it
            if exc.response and exc.response.status_code == 410:
                logger.info("Subscription expired for user %s — removing", user_id)
                try:
                    db_conn.execute(
                        "DELETE FROM push_subscriptions WHERE user_id = ?",
                        (user_id,),
                    )
                    db_conn.commit()
                except Exception:
                    pass
            else:
                logger.error("WebPushException for user %s: %s", user_id, exc)
            return False

    @staticmethod
    async def broadcast_daily_drop(db_conn) -> Dict[str, int]:
        """
        Broadcast a Daily Drop notification to ALL subscribed users.
        Intended to be called by a scheduled job (e.g. APScheduler / cron).
        Returns {'sent': N, 'failed': M, 'skipped': K}.
        """
        try:
            rows = db_conn.execute(
                "SELECT user_id FROM push_subscriptions"
            ).fetchall()
        except Exception as exc:
            logger.error("Could not fetch subscriptions for broadcast: %s", exc)
            return {"sent": 0, "failed": 0, "skipped": 0}

        sent = failed = skipped = 0
        for row in rows:
            uid = row["user_id"]
            ok = await NotificationService.send_daily_drop_notification(
                uid, "Today's Look", db_conn
            )
            if ok:
                sent += 1
            else:
                failed += 1

        logger.info("Broadcast complete: sent=%d failed=%d skipped=%d", sent, failed, skipped)
        return {"sent": sent, "failed": failed, "skipped": skipped}

    @staticmethod
    def get_vapid_public_key() -> Optional[str]:
        """Return the VAPID public key for the frontend to use when subscribing."""
        return VAPID_PUBLIC_KEY
