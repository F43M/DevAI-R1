from .config import logger


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
        """Fine tune the language model with RLHF (placeholder)."""
        from pathlib import Path

        logger.warning("RLHF ainda não implementado. Ver #pending_rlhf.")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return {"status": "skipped"}


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
