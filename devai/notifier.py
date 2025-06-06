import smtplib
from email.message import EmailMessage
from .config import config, logger

class Notifier:
    """Simple email notifier."""

    def __init__(self):
        self.enabled = bool(getattr(config, "NOTIFY_EMAIL", ""))

    def send(self, subject: str, body: str) -> None:
        if not self.enabled:
            return
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
