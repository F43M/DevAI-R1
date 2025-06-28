import asyncio
import json
import sys
import types
from pathlib import Path

# Stub heavy Scraper_Wiki dependencies
sys.modules.setdefault(
    "transformers",
    types.SimpleNamespace(
        AutoModelForCausalLM=object,
        AutoTokenizer=object,
        GPT2Config=object,
        GPT2LMHeadModel=object,
        PreTrainedTokenizerFast=object,
        TrainingArguments=object,
        Trainer=object,
    ),
)
sys.modules.setdefault("trl", types.SimpleNamespace(SFTTrainer=object))
sys.modules.setdefault(
    "tokenizers",
    types.ModuleType("tokenizers"),
)
tokenizers_models = types.ModuleType("tokenizers.models")
tokenizers_models.WordLevel = object
tokenizers_pre = types.ModuleType("tokenizers.pre_tokenizers")
tokenizers_pre.Whitespace = object
tokenizers_trainers = types.ModuleType("tokenizers.trainers")
tokenizers_trainers.WordLevelTrainer = object
sys.modules.setdefault("tokenizers.models", tokenizers_models)
sys.modules.setdefault("tokenizers.pre_tokenizers", tokenizers_pre)
sys.modules.setdefault("tokenizers.trainers", tokenizers_trainers)
sys.modules["tokenizers"].Tokenizer = object
sys.modules.setdefault("torch", types.SimpleNamespace())
sys.modules.setdefault(
    "sentence_transformers", types.SimpleNamespace(SentenceTransformer=object)
)
sys.modules.setdefault(
    "datasets",
    types.SimpleNamespace(Dataset=object, concatenate_datasets=lambda *a, **k: None),
)
sys.modules.setdefault("spacy", types.SimpleNamespace(load=lambda *a, **k: None))
sys.modules.setdefault("unidecode", types.SimpleNamespace(unidecode=lambda x: x))
sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=lambda x, **k: x))
sys.modules.setdefault("html2text", types.SimpleNamespace())
wiki_mod = types.ModuleType("wikipediaapi")
wiki_mod.WikipediaException = Exception
wiki_mod.Namespace = types.SimpleNamespace(MAIN=0, CATEGORY=14)
wiki_mod.ExtractFormat = types.SimpleNamespace(HTML=0)
wiki_mod.WikipediaPage = object
wiki_mod.Wikipedia = lambda *a, **k: types.SimpleNamespace(
    page=lambda *a, **k: types.SimpleNamespace(exists=lambda: False),
    api=types.SimpleNamespace(article_url=lambda x: ""),
)
sys.modules.setdefault("wikipediaapi", wiki_mod)
aiohttp_stub = types.SimpleNamespace(
    ClientSession=object,
    ClientTimeout=lambda *a, **k: None,
    ClientError=Exception,
    ClientResponseError=Exception,
)
sys.modules.setdefault("aiohttp", aiohttp_stub)
sys.modules.setdefault(
    "backoff",
    types.SimpleNamespace(
        on_exception=lambda *a, **k: (lambda f: f), expo=lambda *a, **k: None
    ),
)
sk_mod = types.ModuleType("sklearn")
sk_mod.cluster = types.SimpleNamespace(KMeans=object)
sk_mod.feature_extraction = types.SimpleNamespace(
    text=types.SimpleNamespace(TfidfVectorizer=object)
)
sys.modules.setdefault("sklearn", sk_mod)
sys.modules.setdefault("sklearn.cluster", sk_mod.cluster)
sys.modules.setdefault("sklearn.feature_extraction", sk_mod.feature_extraction)
sys.modules.setdefault(
    "sklearn.feature_extraction.text", sk_mod.feature_extraction.text
)


def _make_ai(saved):
    from devai.memory import MemoryManager

    mem = object.__new__(MemoryManager)
    mem.index = None

    def save(entry, update_feedback=False, ttl_seconds=None):
        saved.append(entry)

    def get_emb(text):
        return [0.0]

    mem.save = save
    mem._get_embedding = get_emb
    ai = types.SimpleNamespace(
        memory=mem, temp_memory_hours=None, new_ingested_data=False
    )
    return ai


def test_aprender_ingests_data(monkeypatch, tmp_path):
    saved = []
    ai = _make_ai(saved)

    fake_sw = types.ModuleType("scraper_wiki")
    fake_sw.Config = types.SimpleNamespace(OUTPUT_DIR=str(tmp_path))
    fake_sw.main = lambda *a, **k: None
    fake_sw.advanced_clean_text = lambda text, lang, split=True: [text]
    sys.modules["scraper_wiki"] = fake_sw

    data = [{"content": "hello", "language": "en"}]
    Path(tmp_path / "wikipedia_qa.json").write_text(json.dumps(data))

    import devai.scraper_interface as si

    monkeypatch.setattr(
        si, "asyncio", types.SimpleNamespace(to_thread=lambda f, *a, **k: f(*a, **k))
    )

    async def _run_sync(func, *a, **k):
        return func(*a, **k)

    monkeypatch.setattr(si, "_run_sync", _run_sync)

    import devai.command_router as cr

    asyncio.run(
        cr.handle_aprender(ai, None, "AI --lang=en", plain=False, feedback_db=None)
    )

    assert ai.new_ingested_data is True
    assert saved and saved[0]["content"] == "hello"


def test_integrar_command(monkeypatch, tmp_path):
    saved = []
    ai = _make_ai(saved)

    fake_sw = types.ModuleType("scraper_wiki")
    fake_sw.Config = types.SimpleNamespace(OUTPUT_DIR=str(tmp_path))
    fake_sw.advanced_clean_text = lambda text, lang, split=True: [text]
    sys.modules["scraper_wiki"] = fake_sw

    data = [{"content": "foo", "language": "en"}]
    Path(tmp_path / "wikipedia_qa.json").write_text(json.dumps(data))

    import devai.command_router as cr

    monkeypatch.setattr(cr, "rlhf", None)
    import devai.data_ingestion as di

    monkeypatch.setattr(di, "scraper_wiki", fake_sw)

    asyncio.run(cr.handle_integrar(ai, None, "", plain=False, feedback_db=None))

    assert ai.new_ingested_data is True
    assert saved
