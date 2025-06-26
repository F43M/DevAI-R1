import json
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy dependencies
sys.modules.setdefault('sentence_transformers', SimpleNamespace(SentenceTransformer=object))
sys.modules.setdefault('datasets', SimpleNamespace(Dataset=object, concatenate_datasets=lambda *a, **k: None))
sys.modules.setdefault('spacy', SimpleNamespace(load=lambda *a, **k: None))
sys.modules.setdefault('unidecode', SimpleNamespace(unidecode=lambda x: x))
sys.modules.setdefault('tqdm', SimpleNamespace(tqdm=lambda x, **k: x))
sys.modules.setdefault('html2text', SimpleNamespace())
wiki_mod = ModuleType('wikipediaapi')
wiki_mod.WikipediaException = Exception
wiki_mod.Namespace = SimpleNamespace(MAIN=0, CATEGORY=14)
wiki_mod.ExtractFormat = SimpleNamespace(HTML=0)
wiki_mod.WikipediaPage = object
wiki_mod.Wikipedia = lambda *a, **k: SimpleNamespace(page=lambda *a, **k: SimpleNamespace(exists=lambda: False), api=SimpleNamespace(article_url=lambda x: ""))
sys.modules.setdefault('wikipediaapi', wiki_mod)
aiohttp_stub = SimpleNamespace(
    ClientSession=object,
    ClientTimeout=lambda *a, **k: None,
    ClientError=Exception,
    ClientResponseError=Exception,
)
sys.modules.setdefault('aiohttp', aiohttp_stub)
sys.modules.setdefault('backoff', SimpleNamespace(on_exception=lambda *a, **k: (lambda f: f), expo=lambda *a, **k: None))

sk_mod = ModuleType('sklearn')
sk_mod.cluster = SimpleNamespace(KMeans=object)
sk_mod.feature_extraction = SimpleNamespace(text=SimpleNamespace(TfidfVectorizer=object))
sys.modules.setdefault('sklearn', sk_mod)
sys.modules.setdefault('sklearn.cluster', sk_mod.cluster)
sys.modules.setdefault('sklearn.feature_extraction', sk_mod.feature_extraction)
sys.modules.setdefault('sklearn.feature_extraction.text', sk_mod.feature_extraction.text)

sumy_mod = ModuleType('sumy')
parsers_mod = ModuleType('sumy.parsers')
plaintext_mod = ModuleType('sumy.parsers.plaintext')
plaintext_mod.PlaintextParser = object
parsers_mod.plaintext = plaintext_mod
nlp_mod = ModuleType('sumy.nlp')
tokenizers_mod = ModuleType('sumy.nlp.tokenizers')
tokenizers_mod.Tokenizer = object
nlp_mod.tokenizers = tokenizers_mod
summarizers_mod = ModuleType('sumy.summarizers')
lsa_mod = ModuleType('sumy.summarizers.lsa')
lsa_mod.LsaSummarizer = object
summarizers_mod.lsa = lsa_mod
sumy_mod.parsers = parsers_mod
sumy_mod.nlp = nlp_mod
sumy_mod.summarizers = summarizers_mod
sys.modules.setdefault('sumy', sumy_mod)
sys.modules.setdefault('sumy.parsers', parsers_mod)
sys.modules.setdefault('sumy.parsers.plaintext', plaintext_mod)
sys.modules.setdefault('sumy.nlp', nlp_mod)
sys.modules.setdefault('sumy.nlp.tokenizers', tokenizers_mod)
sys.modules.setdefault('sumy.summarizers', summarizers_mod)
sys.modules.setdefault('sumy.summarizers.lsa', lsa_mod)
sys.modules.setdefault('streamlit', SimpleNamespace())
sys.modules.setdefault('psutil', SimpleNamespace(cpu_percent=lambda interval=1: 0, virtual_memory=lambda: SimpleNamespace(percent=0)))

import main
from click.testing import CliRunner


def test_single_url_csv(tmp_path, monkeypatch):
    async def fake_fetch(title, lang):
        return f"<html>{title}-{lang}</html>"

    monkeypatch.setattr(main.scraper_wiki, "fetch_html_content_async", fake_fetch)
    out_dir = tmp_path
    runner = CliRunner()
    result = runner.invoke(main.main, [
        "--url", "https://en.wikipedia.org/wiki/Python",
        "--output", "csv",
        "--out-dir", str(out_dir),
    ])
    assert result.exit_code == 0
    out_file = out_dir / "output.csv"
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "https://en.wikipedia.org/wiki/Python" in content


def test_batch_file_json(tmp_path, monkeypatch):
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text(
        "https://en.wikipedia.org/wiki/Python\nhttps://pt.wikipedia.org/wiki/Programacao"
    )

    async def fake_fetch(title, lang):
        return f"<html>{title}-{lang}</html>"

    monkeypatch.setattr(main.scraper_wiki, "fetch_html_content_async", fake_fetch)

    runner = CliRunner()
    result = runner.invoke(main.main, [
        "--file", str(urls_file),
        "--output", "json",
        "--out-dir", str(tmp_path),
    ])
    assert result.exit_code == 0
    out_file = tmp_path / "output.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert len(data) == 2

