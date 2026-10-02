"""
Notifier – unified notification dispatcher.
Listens to NOTIFY_USER events and routes them to desktop / email.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from database.db import get_session
from database.models import NotificationLevel
from database.repositories import NotificationRepository
from supervisor.event_bus import EventType, Event, bus

logger = logging.getLogger(__name__)


class Notifier:
    """
    Subscribes to NOTIFY_USER events and dispatches notifications
    via configured channels (desktop, email).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._cfg = config.get("notifications", {})
        bus.subscribe(EventType.NOTIFY_USER, self._on_notify)

    def _on_notify(self, event: Event) -> None:
        p = event.payload
        title   = p.get("title", "AI Training Supervisor")
        message = p.get("message", "")
        level   = p.get("level", "info")

        # Desktop
        if self._cfg.get("desktop", {}).get("enabled", True):
            from integrations.notifications.desktop import send_desktop
            send_desktop(title, message)

        # Email
        email_cfg = self._cfg.get("email", {})
        if email_cfg.get("enabled", False):
            from integrations.notifications.email import send_email
            send_email(
                subject=f"[ATS] {title}",
                body=message,
                recipients=email_cfg.get("recipients", []),
                smtp_host=email_cfg.get("smtp_host", ""),
                smtp_port=email_cfg.get("smtp_port", 587),
                sender=email_cfg.get("sender", ""),
                password=os.getenv("SMTP_PASSWORD", ""),
                use_tls=email_cfg.get("use_tls", True),
            )

        # Log to DB
        db = get_session()
        try:
            lvl_map = {
                "info":    NotificationLevel.INFO,
                "warning": NotificationLevel.WARNING,
                "error":   NotificationLevel.ERROR,
                "success": NotificationLevel.SUCCESS,
            }
            nl = lvl_map.get(level.lower(), NotificationLevel.INFO)
            entry = NotificationRepository(db).log(
                title=title, message=message, level=nl
            )
            NotificationRepository(db).mark_sent(entry.id)
        finally:
            db.close()
