from .config import logger
import json


class RLFineTuner:
    """Placeholder for reinforcement learning fine-tuning."""

    def __init__(self, memory_manager):
        self.memory = memory_manager

    def collect_examples(self):
        """Gather examples from memory for future RLHF datasets."""

        import json

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

    async def fine_tune(self, base_model: str, output_dir: str) -> dict:
        """Fine tune the language model with RLHF using the TRL library."""
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainingArguments,
            )
        except Exception as exc:  # pragma: no cover - optional heavy dep
            logger.warning("transformers não encontrado; pulando RLHF")
            return {"status": "skipped", "error": str(exc)}

        try:
            from trl import SFTTrainer
        except Exception as exc:  # pragma: no cover - optional dep
            logger.warning("trl não encontrado; pulando treinamento RLHF")
            return {"status": "skipped", "error": str(exc)}

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


if __name__ == "__main__":
    import argparse
    import asyncio
    from .memory import MemoryManager
    from .config import config

    parser = argparse.ArgumentParser(description="Run RLHF fine-tuning")
    parser.add_argument("base_model")
    parser.add_argument("output_dir")
    parser.add_argument("--db", default=config.MEMORY_DB)
    args = parser.parse_args()

    mem = MemoryManager(args.db, config.EMBEDDING_MODEL, model=None, index=None)
    tuner = RLFineTuner(mem)
    asyncio.run(tuner.fine_tune(args.base_model, args.output_dir))
