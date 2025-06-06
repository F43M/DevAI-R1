import asyncio
from datetime import datetime

import aiohttp

from .config import config, logger, metrics


class AIModel:
    def __init__(self):
        self.session = aiohttp.ClientSession()
        self.models = config.MODELS or {"default": {
            "name": config.MODEL_NAME,
            "api_key": config.OPENROUTER_API_KEY,
            "url": config.OPENROUTER_URL,
        }}
        self.current = "default"
        if not any(m.get("api_key") for m in self.models.values()):
            logger.error("Nenhuma chave de modelo configurada")
        logger.info("Modelos disponíveis", models=list(self.models.keys()))

    def set_model(self, name: str) -> None:
        if name in self.models:
            self.current = name
            logger.info("Modelo selecionado", model=name)
        else:
            logger.error("Modelo não encontrado", model=name)

    async def generate(self, prompt: str, max_length: int = config.MAX_CONTEXT_LENGTH) -> str:
        model_cfg = self.models.get(self.current, {})
        headers = {
            "Authorization": f"Bearer {model_cfg.get('api_key', '')}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_cfg.get("name", config.MODEL_NAME),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": min(max_length, config.MAX_CONTEXT_LENGTH),
            "temperature": 0.7,
        }
        start = datetime.now()
        try:
            async with self.session.post(
                model_cfg.get("url", config.OPENROUTER_URL),
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
                error = await resp.text()
                metrics.record_error()
                logger.error("Erro na chamada ao OpenRouter", status=resp.status, error=error)
                return f"Erro na API: {resp.status} - {error}"
        except asyncio.TimeoutError:
            metrics.record_error()
            logger.error("Timeout na chamada ao OpenRouter")
            return "Erro: tempo limite excedido ao chamar a API"
        except Exception as e:
            metrics.record_error()
            logger.error("Erro na conexão com OpenRouter", error=str(e))
            return f"Erro de conexão: {str(e)}"
        finally:
            metrics.record_call((datetime.now() - start).total_seconds())

    async def close(self):
        await self.session.close()
