"""Email notification sender using smtplib."""

from __future__ import annotations

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

logger = logging.getLogger(__name__)


def send_email(
    subject: str,
    body: str,
    recipients: list[str],
    smtp_host: str,
    smtp_port: int = 587,
    sender: str = "",
    password: str = "",
    use_tls: bool = True,
) -> bool:
    """Send an email notification. Returns True on success."""
    if not recipients or not smtp_host:
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = sender
        msg["To"]      = ", ".join(recipients)
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            if use_tls:
                server.starttls()
            if sender and password:
                server.login(sender, password)
            server.sendmail(sender, recipients, msg.as_string())
        logger.info("Email sent to %s.", recipients)
        return True
    except Exception as exc:
        logger.error("Email notification failed: %s", exc)
        return False
