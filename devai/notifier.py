import asyncio
import smtplib
from email.message import EmailMessage
import aiohttp
from .config import config, logger

class Notifier:
    """Simple notifier for email and Slack."""

    def __init__(self):
        self.email_enabled = bool(getattr(config, "NOTIFY_EMAIL", ""))
        self.slack_url = getattr(config, "NOTIFY_SLACK", "")
        self.enabled = self.email_enabled or bool(self.slack_url)

    async def _send_slack(self, text: str) -> None:
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:
                await session.post(self.slack_url, json={"text": text})
            logger.info("Notificação Slack enviada")
        except Exception as e:  # pragma: no cover - notification failures tolerated
            logger.error("Falha ao enviar Slack", error=str(e))

    def _run_async(self, coro) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coro)
        else:
            loop.create_task(coro)

    def send(self, subject: str, body: str) -> None:
        if not self.enabled:
            return
        if self.email_enabled:
            try:
                msg = EmailMessage()
                msg["Subject"] = subject
                msg["From"] = config.NOTIFY_EMAIL
                msg["To"] = config.NOTIFY_EMAIL
                msg.set_content(body)
                with smtplib.SMTP("localhost") as s:
                    s.send_message(msg)
                logger.info("Notificação enviada")
            except Exception as e:  # pragma: no cover - notification failures tolerated
                logger.error("Falha ao enviar notificação", error=str(e))

        if self.slack_url:
            text = f"*{subject}*\n{body}"
            self._run_async(self._send_slack(text))
