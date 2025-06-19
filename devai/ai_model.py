import asyncio
from collections import OrderedDict
from datetime import datetime
from difflib import SequenceMatcher
from typing import Mapping, Sequence, Union, AsyncGenerator
import json
from uuid import uuid4

SYSTEM_MESSAGE = (
    "Você é um assistente especialista em desenvolvimento de software "
    "com foco em qualidade, segurança e aprendizado simbólico."
)

from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.setup_local_model import setup_local_model

from .config import config, logger, metrics
from .error_handler import friendly_message, log_error
import re
from pathlib import Path
from .memory import MemoryManager


def is_response_incomplete(response: str) -> bool:
    """Heuristically detect if the response looks truncated."""
    text = response.rstrip()
    if not text:
        return True
    if text.endswith("..."):
        return True
    if re.search(r"(truncat|cortad|stop reason)", text, re.IGNORECASE):
        return True
    if re.search(
        r"\b(portanto|logo|entao|isso significa que|ou seja|em conclusao)$",
        text,
        re.IGNORECASE,
    ):
        return True
    if text.count("```") % 2 == 1:
        return True
    if (
        text.count("{") > text.count("}")
        or text.count("[") > text.count("]")
        or text.count("(") > text.count(")")
    ):
        return True
    if (
        (text.startswith("{") or text.startswith("["))
        and not text.endswith("}")
        and not text.endswith("]")
    ):
        return True
    if not text.endswith((".", "!", "?", "\n", "`", "'", '"', ")", "}", "]")):
        return True
    return False


def rebuild_response(original: str, continuation: str) -> str:
    """Join two parts of a response avoiding duplicates."""
    orig = original.rstrip()
    cont = continuation.lstrip()
    for i in range(min(len(orig), len(cont)), 0, -1):
        if orig.endswith(cont[:i]):
            return orig + cont[i:]
    return orig + "\n" + cont


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
        """Initialize and load a local language model."""

        self.cache = PromptCache()
        model_dir = Path(config.LOCAL_MODEL or "DevAI-Model")
        self.local_model, self.local_tokenizer = setup_local_model(out_dir=model_dir)
        self.current = "local"

    async def shutdown(self):
        """Placeholder for compatibility."""
        return

    def set_model(self, name: str) -> None:
        self.current = "local"

    async def generate(
        self,
        prompt: Union[str, Sequence[Mapping[str, str]]],
        max_length: int = config.MAX_CONTEXT_LENGTH,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        if isinstance(prompt, str):
            key = prompt
            messages = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = list(prompt)
            if not any(m.get("role") == "system" for m in messages):
                messages.insert(0, {"role": "system", "content": SYSTEM_MESSAGE})
            key = json.dumps(messages, sort_keys=True)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        prompt_id = uuid4().hex

        try:
            input_ids = self.local_tokenizer.encode(
                messages[-1]["content"], return_tensors="pt"
            )
            output = self.local_model.generate(
                input_ids,
                max_new_tokens=min(max_length, config.MAX_CONTEXT_LENGTH),
                do_sample=True,
                temperature=temperature,
            )
            response_text = self.local_tokenizer.decode(
                output[0], skip_special_tokens=True
            )
            used_model = "local"
            if stream:

                async def _stream() -> AsyncGenerator[str, None]:
                    for tok in response_text.split():
                        yield tok + " "

                metrics.record_call(0)
                return _stream()
        except Exception as e:  # pragma: no cover - heavy dep
            log_error("local_model", e)
            metrics.record_error()
            return friendly_message(e)

        metrics.record_call(0)

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
        hist_file = (
            hist_dir / f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{used_model}.txt"
        )
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

        metrics.record_model_usage(used_model)
        if is_response_incomplete(response_text):
            metrics.record_incomplete()

        self.cache.add(key, response_text)
        return annotation + response_text

    async def close(self):
        await self.shutdown()

    async def safe_api_call(
        self,
        prompt: Union[str, Sequence[Mapping[str, str]]],
        max_tokens: int,
        context: str = "",
        memory: "MemoryManager | None" = None,
        temperature: float = 0.7,
    ) -> str:
        """Call the model ensuring the answer is complete."""
        if isinstance(prompt, str):
            prompt_len = len(prompt.split())
        else:
            prompt_len = sum(len(m.get("content", "").split()) for m in prompt)
        available = max_tokens - prompt_len - len(context.split())
        if available <= 0:
            available = max_tokens
        attempts = 0
        note = ""
        try:
            response = await self.generate(
                prompt, max_length=available, temperature=temperature
            )
        except Exception as e:
            logger.info("Retry attempt triggered after error...")
            note = "⚠️ O modelo demorou para responder. Tentando novamente…\n"
            await asyncio.sleep(2)
            try:
                response = await self.generate(
                    prompt, max_length=available, temperature=temperature
                )
            except Exception:
                log_error("safe_api_call", e)
                return "❌ " + friendly_message(e)
        if isinstance(response, str) and "401" in response:
            return "🚫 Token inválido ou sem autorização"
        while attempts < 3 and (
            is_response_incomplete(response) or len(response.split()) >= available - 1
        ):
            attempts += 1
            if isinstance(prompt, str):
                cont_prompt = f"{context}\nContinue exatamente de onde você parou. Não repita partes da resposta anterior."
            else:
                cont_prompt = list(prompt) + [
                    {
                        "role": "system",
                        "content": "Continue exatamente de onde você parou. Não repita partes da resposta anterior.",
                    }
                ]
            continuation = await self.generate(
                cont_prompt, max_length=available, temperature=temperature
            )
            response = rebuild_response(response, continuation)
        if attempts:
            note = "Essa resposta foi reconstruída após corte automático.\n"
        incomplete = is_response_incomplete(response)
        if incomplete:
            try:
                Path("logs").mkdir(exist_ok=True)
                with open("logs/api_recovery_log.md", "a", encoding="utf-8") as f:
                    preview = prompt if isinstance(prompt, str) else str(prompt)
                    f.write(
                        f"- {datetime.now().isoformat()} | tokens:{available} | tentativas:{attempts} | prompt:{preview[:60]}\n"
                    )
            except Exception:
                pass
            note += "[Resposta possivelmente incompleta]\n"
            if attempts >= 3 and memory is not None:
                memory.save(
                    {
                        "type": "recovery",
                        "memory_type": "resposta_cortada",
                        "content": response,
                        "metadata": {
                            "prompt": (
                                prompt if isinstance(prompt, str) else str(prompt)
                            ),
                            "status": "incompleto",
                        },
                        "resposta_recomposta": attempts > 0,
                    }
                )
        return note + response

    async def safe_api_stream(
        self,
        prompt: Union[str, Sequence[Mapping[str, str]]],
        max_tokens: int,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream response from the model when supported."""
        if isinstance(prompt, str):
            prompt_len = len(prompt.split())
        else:
            prompt_len = sum(len(m.get("content", "").split()) for m in prompt)
        available = max_tokens - prompt_len
        if available <= 0:
            available = max_tokens
        try:
            async for token in self.generate(
                prompt, max_length=available, temperature=temperature, stream=True
            ):
                yield token
        except Exception:
            text = await self.safe_api_call(prompt, max_tokens, "")
            for tok in text.split():
                yield tok
