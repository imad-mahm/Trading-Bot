"""Telegram alerts — free, and crash-proof.

A notification failure must NEVER take down the trading loop, so every send is
wrapped in try/except: on any error we log a warning and carry on. If no token
is configured the notifier quietly becomes a no-op (it still logs locally).

Uses only the standard library (urllib), so there's no extra dependency.

Setup (see README): message @BotFather on Telegram, create a bot, copy the
token; then message your new bot once and read your chat id. Put both in
config.yaml under live.telegram, or set TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request

log = logging.getLogger("paper.notifier")


class Notifier:
    def __init__(self, config: dict):
        tg = (config.get("live", {}).get("telegram", {}) or {})
        # Environment variables win over config (keeps secrets out of the file).
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN") or tg.get("bot_token") or ""
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID") or tg.get("chat_id") or ""
        self.enabled = bool(tg.get("enabled", True) and self.token and self.chat_id)
        if tg.get("enabled", True) and not self.enabled:
            log.info("Telegram not configured (no token/chat_id) — alerts go to the log only.")

    def send(self, text: str) -> bool:
        """Send a message. Returns True if it went out, False otherwise. Never
        raises — a dead network can't stop trading."""
        log.info("ALERT: %s", text.replace("\n", " | "))
        if not self.enabled:
            return False
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
            }).encode()
            with urllib.request.urlopen(url, data=data, timeout=10) as resp:
                json.loads(resp.read().decode())
            return True
        except Exception as exc:  # noqa: BLE001 — deliberately swallow everything
            log.warning("Telegram send failed (continuing anyway): %s", exc)
            return False
