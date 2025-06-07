import asyncio
from collections import OrderedDict
from datetime import datetime
from difflib import SequenceMatcher
from typing import Mapping, Sequence, Union
import json
from uuid import uuid4

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
        if isinstance(prompt, str):
            key = prompt
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = list(prompt)
            key = json.dumps(messages, sort_keys=True)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        prompt_id = uuid4().hex

        if self.current == "local" and self.local_model and self.local_tokenizer:
            try:
                input_ids = self.local_tokenizer.encode(messages[-1]["content"], return_tensors="pt")
                output = self.local_model.generate(input_ids, max_new_tokens=min(max_length, config.MAX_CONTEXT_LENGTH))
                text = self.local_tokenizer.decode(output[0], skip_special_tokens=True)
                self.cache.add(key, text)
                metrics.record_call(0)
                return text
            except Exception as e:  # pragma: no cover - heavy dep
                logger.error("Erro no modelo local", error=str(e))

        async def _call(model_name: str) -> str:
            cfg = self.models.get(model_name, {})
            if cfg.get("local") and self.local_model and self.local_tokenizer:
                try:
                    input_ids = self.local_tokenizer.encode(messages[-1]["content"], return_tensors="pt")
                    output = self.local_model.generate(input_ids, max_new_tokens=min(max_length, config.MAX_CONTEXT_LENGTH))
                    return self.local_tokenizer.decode(output[0], skip_special_tokens=True)
                except Exception as e:  # pragma: no cover - heavy dep
                    logger.error("Erro no modelo local", error=str(e))
                    return f"Erro: {str(e)}"
            headers = {
                "Authorization": f"Bearer {cfg.get('api_key', '')}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": cfg.get("name", config.MODEL_NAME),
                "messages": messages,
                "max_tokens": min(max_length, config.MAX_CONTEXT_LENGTH),
                "temperature": 0.7,
            }
            start = datetime.now()
            try:
                resp = await self.session.post(
                    cfg.get("url", config.OPENROUTER_URL),
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                )
                if resp.status == 200:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
                error = await resp.text()
                logger.error("Erro ao chamar modelo", model=model_name, status=resp.status)
                return f"Erro: {error}"
            except Exception as e:
                logger.error("Falha na conexao", model=model_name, error=str(e))
                return f"Erro: {str(e)}"
            finally:
                metrics.record_call((datetime.now() - start).total_seconds())

        model_order = [self.current] + [m for m in self.models if m != self.current]
        response_text = ""
        for name in model_order:
            text = await _call(name)
            if text and "Erro" not in text:
                response_text = text
                used_model = name
                break
        if not response_text:
            metrics.record_error()
            used_model = "none"
            response_text = "# Falha geral – requires human input"

        response_id = uuid4().hex
        annotation = f"# IA usada: {used_model}\n# PromptID: {prompt_id}\n# RespostaID: {response_id}\n"
        from .symbolic_verification import evaluate_ai_response
        from .decision_log import log_decision
        score, details = evaluate_ai_response(response_text)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": used_model,
            "prompt_id": prompt_id,
            "response_id": response_id,
            "prompt": messages[-1]["content"],
            "response": response_text,
            "evaluation": {"score": score, "details": details},
            "fallback": used_model != self.current,
        }
        try:
            with open("prompt_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error("Erro ao registrar prompt", error=str(e))

        from pathlib import Path
        hist_dir = Path("history/prompts")
        hist_dir.mkdir(parents=True, exist_ok=True)
        hist_file = hist_dir / f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{used_model}.txt"
        hist_file.write_text(response_text)

        fallback = used_model != self.current
        log_decision(
            "resposta",
            "ai_model",
            "geracao",
            used_model,
            response_text,
            score=score,
            fallback=fallback,
        )

        self.cache.add(key, response_text)
        return annotation + response_text

    async def close(self):
        await self.session.close()
