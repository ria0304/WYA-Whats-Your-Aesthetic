# services/email_service.py
# WYA — Gmail-based email notification service.
#
# Sends "Your Daily Drop is ready" digests via Gmail SMTP.
# All credentials are read from environment variables — never hardcoded.
#
# Required env vars:
#   WYA_GMAIL_ADDRESS   — the Gmail account that sends mail (e.g. hello@wya.app or a personal gmail)
#   WYA_GMAIL_APP_PASS  — Gmail App Password (NOT your normal password).
#                         Generate at: myaccount.google.com → Security → App passwords
#                         Requires 2FA to be enabled on the sending account.
#
# Optional:
#   WYA_APP_BASE_URL    — frontend URL used in "Open App" links (default: http://localhost:5173)
#
# Usage (called from main.py or a scheduler):
#   from services.email_service import email_service
#   await email_service.send_daily_drop(user_email, user_name, outfit_data)

import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

GMAIL_ADDRESS: Optional[str] = os.getenv("WYA_GMAIL_ADDRESS")
GMAIL_APP_PASS: Optional[str] = os.getenv("WYA_GMAIL_APP_PASS")
APP_BASE_URL: str = os.getenv("WYA_APP_BASE_URL", "http://localhost:5173")

EMAIL_ENABLED: bool = bool(GMAIL_ADDRESS and GMAIL_APP_PASS)

if not EMAIL_ENABLED:
    logger.warning(
        "WYA email notifications disabled — set WYA_GMAIL_ADDRESS and "
        "WYA_GMAIL_APP_PASS env vars to enable Daily Drop emails."
    )


# ---------------------------------------------------------------------------
# HTML email template
# ---------------------------------------------------------------------------

def _build_daily_drop_html(
    user_name: str,
    outfit_name: str,
    outfit_vibe: str,
    pieces: List[Dict[str, Any]],
    style_note: str,
    weather_snippet: str,
    day_score: int,
    color_palette: List[str],
) -> str:
    """Returns a polished HTML email body for the Daily Drop."""

    # Color swatches row
    swatch_html = "".join(
        f'<span style="display:inline-block;width:18px;height:18px;border-radius:50%;'
        f'background:{c};margin:0 3px;border:1px solid rgba(0,0,0,.1);"></span>'
        for c in (color_palette or ["#c4a882"])[:5]
    )

    # Outfit pieces list
    pieces_html = "".join(
        f"""
        <tr>
          <td style="padding:8px 0;border-bottom:1px solid #f0ede8;">
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
                         background:#c4a882;margin-right:10px;vertical-align:middle;"></span>
            <span style="font-size:14px;color:#3d3632;">{p.get('name','Item')}</span>
            <span style="font-size:12px;color:#9c9891;margin-left:6px;">{p.get('category','')}</span>
          </td>
        </tr>
        """
        for p in (pieces or [])[:6]
    )

    first_name = user_name.split()[0] if user_name else "there"
    weekday = datetime.utcnow().strftime("%A")

    return f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f7f4ef;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;">

  <!-- Outer wrapper -->
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f7f4ef;padding:32px 0;">
    <tr><td align="center">

      <!-- Card -->
      <table width="560" cellpadding="0" cellspacing="0"
             style="background:#ffffff;border-radius:16px;overflow:hidden;
                    box-shadow:0 4px 24px rgba(0,0,0,.06);max-width:560px;width:100%;">

        <!-- Header bar -->
        <tr>
          <td style="background:#1a1714;padding:24px 32px;text-align:left;">
            <span style="font-size:20px;font-weight:600;color:#f5f0e8;letter-spacing:.04em;">WYA</span>
            <span style="font-size:12px;color:#9c9891;margin-left:8px;">What's Your Aesthetic</span>
          </td>
        </tr>

        <!-- Intro -->
        <tr>
          <td style="padding:32px 32px 0;">
            <p style="margin:0 0 4px;font-size:13px;color:#9c9891;text-transform:uppercase;letter-spacing:.08em;">{weekday}</p>
            <h1 style="margin:0 0 8px;font-size:26px;font-weight:600;color:#1a1714;line-height:1.2;">
              Your Daily Drop is ready ✦
            </h1>
            <p style="margin:0;font-size:15px;color:#6b6662;">Hi {first_name} — here's what we pulled for you today.</p>
          </td>
        </tr>

        <!-- Weather pill -->
        {'<tr><td style="padding:16px 32px 0;"><span style="display:inline-block;background:#f7f4ef;border-radius:20px;padding:6px 14px;font-size:13px;color:#6b6662;">' + weather_snippet + '</span></td></tr>' if weather_snippet else ''}

        <!-- Outfit card -->
        <tr>
          <td style="padding:24px 32px;">
            <table width="100%" cellpadding="0" cellspacing="0"
                   style="background:#faf8f5;border-radius:12px;border:1px solid #ede9e2;overflow:hidden;">
              <tr>
                <td style="padding:20px 24px 0;">
                  <p style="margin:0 0 2px;font-size:11px;color:#9c9891;text-transform:uppercase;letter-spacing:.1em;">Today's look</p>
                  <h2 style="margin:0 0 4px;font-size:20px;font-weight:600;color:#1a1714;">{outfit_name}</h2>
                  <p style="margin:0 0 16px;font-size:13px;color:#9c9891;">{outfit_vibe}</p>
                </td>
              </tr>

              <!-- Color palette -->
              <tr>
                <td style="padding:0 24px 16px;">
                  {swatch_html}
                </td>
              </tr>

              <!-- Compatibility score bar -->
              <tr>
                <td style="padding:0 24px 16px;">
                  <p style="margin:0 0 6px;font-size:12px;color:#9c9891;">Color harmony score</p>
                  <table width="100%" cellpadding="0" cellspacing="0">
                    <tr>
                      <td style="background:#ede9e2;border-radius:4px;height:6px;overflow:hidden;">
                        <div style="width:{day_score}%;height:6px;background:#c4a882;border-radius:4px;"></div>
                      </td>
                      <td width="36" style="padding-left:8px;font-size:12px;color:#1a1714;font-weight:600;">{day_score}</td>
                    </tr>
                  </table>
                </td>
              </tr>

              <!-- Pieces -->
              {'<tr><td style="padding:0 24px;"><table width="100%" cellpadding="0" cellspacing="0">' + pieces_html + '</table></td></tr>' if pieces_html else ''}

              <!-- Style note -->
              <tr>
                <td style="padding:16px 24px 20px;">
                  <p style="margin:0;font-size:13px;color:#6b6662;font-style:italic;
                             border-left:3px solid #c4a882;padding-left:12px;line-height:1.5;">
                    {style_note}
                  </p>
                </td>
              </tr>
            </table>
          </td>
        </tr>

        <!-- CTA button -->
        <tr>
          <td style="padding:0 32px 32px;text-align:center;">
            <a href="{APP_BASE_URL}/daily-drop"
               style="display:inline-block;background:#1a1714;color:#f5f0e8;text-decoration:none;
                      font-size:14px;font-weight:500;padding:14px 36px;border-radius:8px;
                      letter-spacing:.03em;">
              Open in WYA →
            </a>
          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="background:#f7f4ef;padding:20px 32px;text-align:center;
                     border-top:1px solid #ede9e2;">
            <p style="margin:0;font-size:12px;color:#b5b0aa;">
              You're receiving this because you enabled Daily Drop notifications.<br>
              <a href="{APP_BASE_URL}/profile" style="color:#9c9891;">Manage preferences</a>
            </p>
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Email service class
# ---------------------------------------------------------------------------

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

            # Plain-text fallback
            plain = f"Your Daily Drop is ready on WYA. Open the app: {APP_BASE_URL}/daily-drop"
            msg.attach(MIMEText(plain, "plain"))
            msg.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(GMAIL_ADDRESS, GMAIL_APP_PASS)
                server.sendmail(GMAIL_ADDRESS, to_email, msg.as_string())

            logger.info("Daily Drop email sent to %s", to_email)
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

    async def send_daily_drop(
        self,
        to_email: str,
        user_name: str,
        outfit_data: Dict[str, Any],
    ) -> bool:
        """
        Send the Daily Drop digest email.

        outfit_data keys (from /api/ai/daily-drop response):
            outfit_name, outfit_vibe, pieces, style_note,
            weather_snippet, day_score, color_palette
        """
        subject = f"✦ Your Daily Drop — {outfit_data.get('outfit_name', 'Today\\'s Look')}"
        html = _build_daily_drop_html(
            user_name=user_name,
            outfit_name=outfit_data.get("outfit_name", "Today's Look"),
            outfit_vibe=outfit_data.get("outfit_vibe", "Curated"),
            pieces=outfit_data.get("pieces", []),
            style_note=outfit_data.get("style_note", "Style is personal — wear what makes you feel confident."),
            weather_snippet=outfit_data.get("weather_snippet", ""),
            day_score=outfit_data.get("day_score", 80),
            color_palette=outfit_data.get("color_palette", []),
        )
        return self._send(to_email, subject, html)

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
