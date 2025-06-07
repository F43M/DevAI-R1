import asyncio
from collections import OrderedDict
from datetime import datetime
from difflib import SequenceMatcher
from typing import Mapping, Sequence, Union
from uuid import uuid4
import json

import aiohttp

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None
    AutoTokenizer = None

from .config import config, logger, metrics


class PromptCache:
    """Cache prompts and responses using fuzzy matching."""

    def __init__(self, max_size: int = 100):
        self.cache: "OrderedDict[str, str]" = OrderedDict()
        self.max_size = max_size

    def get(self, prompt: str) -> str | None:
        for stored, resp in list(self.cache.items()):
            if SequenceMatcher(None, stored, prompt).ratio() > 0.9:
                self.cache.move_to_end(stored)
                return resp
        return None

    def add(self, prompt: str, response: str) -> None:
        self.cache[prompt] = response
        self.cache.move_to_end(prompt)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)


class AIModel:
    def __init__(self):
        self.session = aiohttp.ClientSession()
        self.models = config.MODELS or {
            "default": {
                "name": config.MODEL_NAME,
                "api_key": config.OPENROUTER_API_KEY,
                "url": config.OPENROUTER_URL,
            }
        }
        if config.LOCAL_MODEL:
            self.models.setdefault("local", {"name": config.LOCAL_MODEL, "local": True})
        self.current = "default"
        self.cache = PromptCache()
        if not any(m.get("api_key") for m in self.models.values() if not m.get("local")):
            logger.error("Nenhuma chave de modelo configurada")
        logger.info("Modelos disponíveis", models=list(self.models.keys()))

        if config.LOCAL_MODEL and AutoModelForCausalLM:
            try:
                self.local_tokenizer = AutoTokenizer.from_pretrained(config.LOCAL_MODEL)
                self.local_model = AutoModelForCausalLM.from_pretrained(config.LOCAL_MODEL)
            except Exception as e:  # pragma: no cover - heavy dep
                logger.error("Erro ao carregar modelo local", error=str(e))
                self.local_model = None
                self.local_tokenizer = None
        else:
            self.local_model = None
            self.local_tokenizer = None

    def set_model(self, name: str) -> None:
        if name in self.models:
            self.current = name
            logger.info("Modelo selecionado", model=name)
        else:
            logger.error("Modelo não encontrado", model=name)

    async def generate(self, prompt: Union[str, Sequence[Mapping[str, str]]], max_length: int = config.MAX_CONTEXT_LENGTH) -> str:
        prompt_id = uuid4().hex
        if isinstance(prompt, str):
            key = prompt
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = list(prompt)
            key = json.dumps(messages, sort_keys=True)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        if self.current == "local" and self.local_model and self.local_tokenizer:
            try:
                input_ids = self.local_tokenizer.encode(messages[-1]["content"], return_tensors="pt")
                output = self.local_model.generate(input_ids, max_new_tokens=min(max_length, config.MAX_CONTEXT_LENGTH))
                text = self.local_tokenizer.decode(output[0], skip_special_tokens=True)
                self.cache.add(key, text)
                metrics.record_call(0)
                self._log_prompt(prompt_id, self.current, messages, text)
                return text
            except Exception as e:  # pragma: no cover - heavy dep
                logger.error("Erro no modelo local", error=str(e))

        model_cfg = self.models.get(self.current, {})
        headers = {
            "Authorization": f"Bearer {model_cfg.get('api_key', '')}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_cfg.get("name", config.MODEL_NAME),
            "messages": messages,
            "max_tokens": min(max_length, config.MAX_CONTEXT_LENGTH),
            "temperature": 0.7,
        }
        start = datetime.now()
        try:
            resp = await self.session.post(
                model_cfg.get("url", config.OPENROUTER_URL),
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            )
            if resp.status == 200:
                data = await resp.json()
                text = data["choices"][0]["message"]["content"]
                self.cache.add(key, text)
                self._log_prompt(prompt_id, self.current, messages, text)
                if self._needs_fallback(text):
                    return await self._fallback_generate(prompt, prompt_id, text)
                return text
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

    def _log_prompt(self, pid: str, model: str, prompt: Sequence[Mapping[str, str]], response: str) -> None:
        entry = {
            "prompt_id": pid,
            "model": model,
            "prompt": prompt,
            "response": response,
            "timestamp": datetime.now().isoformat(),
        }
        try:
            with open("prompt_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error("Falha ao registrar prompt", error=str(e))

    def _needs_fallback(self, text: str) -> bool:
        return text.strip() == "" or text.lower().startswith("erro")

    async def _fallback_generate(self, prompt, pid: str, previous: str) -> str:
        for name, cfg in self.models.items():
            if name == self.current:
                continue
            old = self.current
            self.current = name
            logger.info("Usando modelo alternativo", model=name)
            result = await self.generate(prompt)
            self.current = old
            if not self._needs_fallback(result):
                return result
        logger.error("Falha geral – requires human input")
        return previous
