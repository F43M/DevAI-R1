"""Example script demonstrating dataset export formats."""

from __future__ import annotations

import json
from pathlib import Path

from training import run_pipeline, chunk_text, tokenize_texts
from training.formats import save_hf_dataset, save_tfrecord_dataset, save_arrow_table


def main(in_file: str, out_dir: str = "exported") -> None:
    """Run the training pipeline and export in several formats."""
    run_pipeline(in_file)

    records = json.load(open(in_file, encoding="utf-8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_hf_dataset(records, out / "hf")
    save_tfrecord_dataset(records, out / "data.tfrecord")
    save_arrow_table(records, out / "data.arrow")

    texts = [r.get("content", "") for r in records]
    tokenize_texts(texts)
    for t in texts:
        chunk_text(t, 128, overlap=32)

    print(f"Export completed to: {out.resolve()}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Export dataset examples")
    p.add_argument("input", help="Path to the JSON dataset")
    p.add_argument("--out", default="exported", help="Output directory")
    args = p.parse_args()
    main(args.input, args.out)
