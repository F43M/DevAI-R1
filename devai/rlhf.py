from __future__ import annotations

import json
from pathlib import Path

from .config import logger, config

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
except Exception as exc:  # pragma: no cover - optional heavy dep
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TrainingArguments = None
    logger.warning(
        "transformers não encontrado; funções RLHF indisponíveis",
        error=str(exc),
    )

try:
    from trl import SFTTrainer
except Exception as exc:  # pragma: no cover - optional dep
    SFTTrainer = None
    logger.warning("trl não encontrado; funções RLHF indisponíveis", error=str(exc))


class RLFineTuner:
    """Simple RLHF fine-tuning helper using TRL."""

    def __init__(self, memory_manager):
        self.memory = memory_manager

    def _collect_from_memory(self) -> list[dict]:
        """Gather scored dialog examples stored in the memory bank."""

        cursor = self.memory.conn.cursor()
        cursor.execute(
            """
            SELECT content, metadata, feedback_score
            FROM memory
            WHERE feedback_score > 0 AND metadata LIKE '%prompt%'
            ORDER BY feedback_score DESC
            LIMIT 100
            """
        )
        examples = []
        for content, meta_json, score in cursor.fetchall():
            try:
                meta = json.loads(meta_json)
            except Exception:
                meta = {}
            prompt = meta.get("prompt")
            if prompt:
                examples.append({"prompt": prompt, "response": content, "score": score})
        return examples

    def _collect_from_logs(self, log_dir: str | None = None) -> list[dict]:
        """Extract simple Q/A pairs from log files."""
        examples: list[dict] = []
        dir_path = Path(log_dir or config.LOG_DIR)
        if not dir_path.exists():
            return examples
        for file in dir_path.glob("*.log"):
            try:
                text = file.read_text()
            except Exception:
                continue
            lines = text.splitlines()
            for i in range(len(lines) - 1):
                if lines[i].startswith("User:") and lines[i + 1].startswith("Assistant:"):
                    q = lines[i].split("User:", 1)[1].strip()
                    a = lines[i + 1].split("Assistant:", 1)[1].strip()
                    if q and a:
                        examples.append({"prompt": q, "response": a, "score": 1})
        return examples

    def collect_examples(self, log_dir: str | None = None) -> list[dict]:
        """Gather examples from memory and optional log files."""

        data = self._collect_from_memory()
        data.extend(self._collect_from_logs(log_dir))
        return data

    async def fine_tune(self, base_model: str, output_dir: str) -> dict:
        """Fine tune the language model with RLHF using the TRL library."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if AutoModelForCausalLM is None or SFTTrainer is None:
            logger.warning("Dependências de RLHF ausentes; pulando treinamento")
            return {"status": "skipped", "error": "missing_dependencies"}

        data = self.collect_examples()
        if not data:
            logger.warning("Sem exemplos para treinamento RLHF")
            return {"status": "no_data"}

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model)

        dataset = [{"text": f"{e['prompt']}\n{e['response']}"} for e in data]

        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            num_train_epochs=1,
            logging_steps=1,
            save_steps=1,
            save_total_limit=1,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            args=args,
        )

        trainer.train()
        trainer.save_model(output_dir)

        metrics = {"status": "success", "num_examples": len(dataset)}
        with open(Path(output_dir) / "metrics.json", "w") as f:
            json.dump(metrics, f)
        return metrics


async def train_from_memory(
    base_model: str, output_dir: str, db: str | None = None, log_dir: str | None = None
) -> dict:
    """Collect examples and fine tune if possible."""
    from .memory import MemoryManager

    mem = MemoryManager(db or config.MEMORY_DB, config.EMBEDDING_MODEL, model=None, index=None)
    tuner = RLFineTuner(mem)
    if not tuner.collect_examples(log_dir):
        logger.warning("Sem exemplos para treinamento RLHF")
        return {"status": "no_data"}
    return await tuner.fine_tune(base_model, output_dir)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for running the fine-tuning procedure."""
    import argparse
    import asyncio

    from .memory import MemoryManager

    parser = argparse.ArgumentParser(description="Run RLHF fine-tuning")
    parser.add_argument("base_model")
    parser.add_argument("output_dir")
    parser.add_argument("--db", default=config.MEMORY_DB)
    args = parser.parse_args(argv)

    result = asyncio.run(
        train_from_memory(args.base_model, args.output_dir, db=args.db)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
