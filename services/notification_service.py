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
    def get_vapid_public_key() -> Optional[str]:
        """Return the VAPID public key for the frontend to use when subscribing."""
        return VAPID_PUBLIC_KEY
