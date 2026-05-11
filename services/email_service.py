# services/email_service.py
# WYA — Gmail-based email notification service.
#
# Sends transactional emails via Gmail SMTP.
# All credentials are read from environment variables — never hardcoded.
#
# Required env vars:
#   WYA_GMAIL_ADDRESS   — the Gmail account that sends mail (e.g. hello@wya.app or a personal gmail)
#   WYA_GMAIL_APP_PASS  — Gmail App Password (NOT your normal password).
#                         Generate at: myaccount.google.com → Security → App passwords
#                         Requires 2FA to be enabled on the sending account.
#
# Optional:
#   WYA_APP_BASE_URL    — frontend URL used in links (default: http://localhost:5173)

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

logger = logging.getLogger(__name__)

GMAIL_ADDRESS: Optional[str] = os.getenv("WYA_GMAIL_ADDRESS")
GMAIL_APP_PASS: Optional[str] = os.getenv("WYA_GMAIL_APP_PASS")
APP_BASE_URL: str = os.getenv("WYA_APP_BASE_URL", "http://localhost:5173")

EMAIL_ENABLED: bool = bool(GMAIL_ADDRESS and GMAIL_APP_PASS)

if not EMAIL_ENABLED:
    logger.warning(
        "WYA email notifications disabled — set WYA_GMAIL_ADDRESS and "
        "WYA_GMAIL_APP_PASS env vars to enable emails."
    )


class EmailService:

    def _send(self, to_email: str, subject: str, html_body: str) -> bool:
        """Low-level send via Gmail SMTP-SSL. Returns True on success."""
        if not EMAIL_ENABLED:
            logger.info("Email not configured — would have sent '%s' to %s", subject, to_email)
            return False
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"WYA — What's Your Aesthetic <{GMAIL_ADDRESS}>"
            msg["To"] = to_email

            msg.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(GMAIL_ADDRESS, GMAIL_APP_PASS)
                server.sendmail(GMAIL_ADDRESS, to_email, msg.as_string())

            logger.info("Email sent to %s", to_email)
            return True

        except smtplib.SMTPAuthenticationError:
            logger.error(
                "Gmail auth failed for %s — check WYA_GMAIL_APP_PASS. "
                "Generate an App Password at myaccount.google.com → Security → App passwords.",
                GMAIL_ADDRESS,
            )
            return False
        except Exception as exc:
            logger.error("Failed to send email to %s: %s", to_email, exc)
            return False

    async def send_test(self, to_email: str) -> bool:
        """Quick connectivity test — sends a plain email to verify credentials work."""
        html = """
        <html><body style="font-family:sans-serif;padding:32px;">
        <h2 style="color:#1a1714;">WYA email test ✓</h2>
        <p>Your Gmail notification setup is working correctly.</p>
        </body></html>
        """
        return self._send(to_email, "WYA — Email notification test", html)


# Module-level singleton
email_service = EmailService()
