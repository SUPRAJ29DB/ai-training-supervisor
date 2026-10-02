"""Desktop notification sender using plyer."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def send_desktop(title: str, message: str, timeout: int = 10) -> bool:
    """Send a desktop notification. Returns True on success."""
    try:
        from plyer import notification
        notification.notify(
            title=title,
            message=message[:256],   # plyer has a message length limit
            app_name="AI Training Supervisor",
            timeout=timeout,
        )
        return True
    except Exception as exc:
        logger.debug("Desktop notification failed (plyer): %s", exc)
        return False
