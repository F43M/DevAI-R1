"""Generate a tiny local language model for offline tests."""

from __future__ import annotations

from pathlib import Path
import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer


def load_dataset(path: Path) -> list[tuple[str, str]]:
    data: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        if "\t" in line:
            q, a = line.split("\t", 1)
            data.append((q.strip(), a.strip()))
    return data


def train_tokenizer(
    pairs: list[tuple[str, str]], out_dir: Path
) -> PreTrainedTokenizerFast:
    corpus = [t for pair in pairs for t in pair]
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"])
    tok.train_from_iterator(corpus, trainer)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_file = out_dir / "tokenizer.json"
    tok.save(str(tok_file))
    return PreTrainedTokenizerFast(
        tokenizer_file=str(tok_file), unk_token="[UNK]", pad_token="[PAD]"
    )


def train_model(pairs: list[tuple[str, str]], out_dir: Path) -> None:
    tokenizer = train_tokenizer(pairs, out_dir)
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=16,
        n_ctx=16,
        n_embd=64,
        n_layer=1,
        n_head=2,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(2):
        for q, a in pairs:
            ids = tokenizer.encode(f"{q} {a}", return_tensors="pt")
            out = model(ids, labels=ids)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    data_file = root / "dataset.txt"
    pairs = load_dataset(data_file)
    out_dir = root / "DevAI-Model"
    train_model(pairs, out_dir)


if __name__ == "__main__":
    main()
