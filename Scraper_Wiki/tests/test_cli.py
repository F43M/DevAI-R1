import json
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy dependencies to avoid installation
sys.modules.setdefault(
    "sentence_transformers", SimpleNamespace(SentenceTransformer=object)
)
sys.modules.setdefault(
    "datasets",
    SimpleNamespace(Dataset=object, concatenate_datasets=lambda *a, **k: None),
)
sys.modules.setdefault("spacy", SimpleNamespace(load=lambda *a, **k: None))
sys.modules.setdefault("unidecode", SimpleNamespace(unidecode=lambda x: x))
sys.modules.setdefault("tqdm", SimpleNamespace(tqdm=lambda x, **k: x))
sys.modules.setdefault("html2text", SimpleNamespace())
ua_mod = ModuleType("fake_useragent")


class _DummyUA:
    def __init__(self, *a, **k):
        self.random = "test-agent"


ua_mod.UserAgent = _DummyUA
sys.modules.setdefault("fake_useragent", ua_mod)

selenium_stub = ModuleType("selenium")
webdriver_stub = ModuleType("selenium.webdriver")
webdriver_stub.Chrome = object
webdriver_stub.ChromeOptions = object
by_stub = ModuleType("selenium.webdriver.common.by")
by_stub.By = SimpleNamespace(ID="id", CSS_SELECTOR="css")
sys.modules.setdefault("selenium", selenium_stub)
sys.modules.setdefault("selenium.webdriver", webdriver_stub)
sys.modules.setdefault(
    "selenium.webdriver.common", ModuleType("selenium.webdriver.common")
)
sys.modules.setdefault("selenium.webdriver.common.by", by_stub)
wiki_mod = ModuleType("wikipediaapi")
wiki_mod.WikipediaException = Exception
wiki_mod.Namespace = SimpleNamespace(MAIN=0, CATEGORY=14)
wiki_mod.ExtractFormat = SimpleNamespace(HTML=0)
wiki_mod.WikipediaPage = object
wiki_mod.Wikipedia = lambda *a, **k: SimpleNamespace(
    page=lambda *a, **k: SimpleNamespace(exists=lambda: False),
    api=SimpleNamespace(article_url=lambda x: ""),
)
sys.modules.setdefault("wikipediaapi", wiki_mod)
aiohttp_stub = SimpleNamespace(
    ClientSession=object,
    ClientTimeout=lambda *a, **k: None,
    ClientError=Exception,
    ClientResponseError=Exception,
)
sys.modules.setdefault("aiohttp", aiohttp_stub)
sys.modules.setdefault(
    "backoff",
    SimpleNamespace(
        on_exception=lambda *a, **k: (lambda f: f), expo=lambda *a, **k: None
    ),
)

sklearn_mod = ModuleType("sklearn")
sklearn_mod.cluster = SimpleNamespace(KMeans=object)
sklearn_mod.feature_extraction = SimpleNamespace(
    text=SimpleNamespace(TfidfVectorizer=object)
)
sys.modules.setdefault("sklearn", sklearn_mod)
sys.modules.setdefault("sklearn.cluster", sklearn_mod.cluster)
sys.modules.setdefault("sklearn.feature_extraction", sklearn_mod.feature_extraction)
sys.modules.setdefault(
    "sklearn.feature_extraction.text", sklearn_mod.feature_extraction.text
)

sumy_mod = ModuleType("sumy")
parsers_mod = ModuleType("sumy.parsers")
plaintext_mod = ModuleType("sumy.parsers.plaintext")
plaintext_mod.PlaintextParser = object
parsers_mod.plaintext = plaintext_mod
nlp_mod = ModuleType("sumy.nlp")
tokenizers_mod = ModuleType("sumy.nlp.tokenizers")
tokenizers_mod.Tokenizer = object
nlp_mod.tokenizers = tokenizers_mod
summarizers_mod = ModuleType("sumy.summarizers")
lsa_mod = ModuleType("sumy.summarizers.lsa")
lsa_mod.LsaSummarizer = object
summarizers_mod.lsa = lsa_mod
sumy_mod.parsers = parsers_mod
sumy_mod.nlp = nlp_mod
sumy_mod.summarizers = summarizers_mod
sys.modules.setdefault("sumy", sumy_mod)
sys.modules.setdefault("sumy.parsers", parsers_mod)
sys.modules.setdefault("sumy.parsers.plaintext", plaintext_mod)
sys.modules.setdefault("sumy.nlp", nlp_mod)
sys.modules.setdefault("sumy.nlp.tokenizers", tokenizers_mod)
sys.modules.setdefault("sumy.summarizers", summarizers_mod)
sys.modules.setdefault("sumy.summarizers.lsa", lsa_mod)
sys.modules.setdefault("streamlit", SimpleNamespace())
sys.modules.setdefault(
    "psutil",
    SimpleNamespace(
        cpu_percent=lambda interval=1: 0,
        virtual_memory=lambda: SimpleNamespace(percent=0),
    ),
)

import wikipediaapi

if not hasattr(wikipediaapi, "WikipediaException"):
    wikipediaapi.WikipediaException = Exception

from typer.testing import CliRunner
import cli


def test_status_command(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.scraper_wiki.Config, "OUTPUT_DIR", str(tmp_path))
    (tmp_path / "example.json").write_text("{}")

    runner = CliRunner()
    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 0
    assert "example.json" in result.output
    assert str(tmp_path) in result.output


def test_queue_command(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "QUEUE_FILE", tmp_path / "queue.jsonl")

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["queue", "--lang", "en", "--category", "Programming", "--format", "json"],
    )

    assert result.exit_code == 0
    queue_content = (tmp_path / "queue.jsonl").read_text().strip()
    assert queue_content
    assert (
        "en" in queue_content
        and "Programming" in queue_content
        and "json" in queue_content
    )


def test_queue_command_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "QUEUE_FILE", tmp_path / "queue.jsonl")

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["queue", "--lang", "en", "--category", "Programming", "--format", "jsonl"],
    )

    assert result.exit_code == 0
    queue_content = (tmp_path / "queue.jsonl").read_text().strip()
    assert queue_content
    assert "jsonl" in queue_content


def test_scrape_command(monkeypatch):
    called = {}

    def fake_main(lang, category, fmt, rate_limit_delay=None, **kwargs):
        called["args"] = {
            "lang": lang,
            "category": category,
            "fmt": fmt,
            "delay": rate_limit_delay,
        }

    monkeypatch.setattr(cli.scraper_wiki, "main", fake_main)
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["scrape", "--lang", "pt", "--category", "AI", "--format", "csv"],
    )

    assert result.exit_code == 0
    assert called["args"] == {
        "lang": ["pt"],
        "category": ["AI"],
        "fmt": "csv",
        "delay": None,
    }


def test_scrape_command_with_delay(monkeypatch):
    called = {}

    def fake_main(lang, category, fmt, rate_limit_delay=None, **kwargs):
        called["args"] = {
            "lang": lang,
            "category": category,
            "fmt": fmt,
            "delay": rate_limit_delay,
        }

    monkeypatch.setattr(cli.scraper_wiki, "main", fake_main)
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "scrape",
            "--lang",
            "pt",
            "--category",
            "AI",
            "--format",
            "csv",
            "--rate-limit-delay",
            "2.5",
        ],
    )

    assert result.exit_code == 0
    assert called["args"] == {
        "lang": ["pt"],
        "category": ["AI"],
        "fmt": "csv",
        "delay": 2.5,
    }


def test_cache_options(monkeypatch):
    monkeypatch.setattr(cli.scraper_wiki, "init_cache", lambda: SimpleNamespace())
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["--cache-backend", "sqlite", "--cache-ttl", "60", "status"],
    )
    assert result.exit_code == 0
    assert cli.scraper_wiki.Config.CACHE_BACKEND == "sqlite"
    assert cli.scraper_wiki.Config.CACHE_TTL == 60


def test_clear_cache_command(monkeypatch):
    called = {}

    def fake_clear():
        called["ok"] = True

    monkeypatch.setattr(cli.scraper_wiki, "clear_cache", fake_clear)
    runner = CliRunner()
    result = runner.invoke(cli.app, ["clear-cache"])
    assert result.exit_code == 0
    assert called.get("ok")


def test_load_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.dashboard, "PROGRESS_FILE", tmp_path / "prog.json")
    progress_data = {"pages_processed": 5}
    (tmp_path / "prog.json").write_text(json.dumps(progress_data))

    loaded = cli.dashboard.load_progress()
    assert loaded == progress_data


def test_parallelism_options(monkeypatch):
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["--max-threads", "5", "--max-processes", "2", "status"],
    )
    assert result.exit_code == 0
    assert cli.scraper_wiki.Config.MAX_THREADS == 5
    assert cli.scraper_wiki.Config.MAX_PROCESSES == 2


def test_scrape_with_training_option(monkeypatch, tmp_path):
    called = {}

    def fake_main(lang, category, fmt, rate_limit_delay=None, **kwargs):
        called["scrape"] = True

    def fake_run(path):
        called["train"] = str(path)

    training_mod = ModuleType("training")
    training_mod.pipeline = SimpleNamespace(run_pipeline=fake_run)
    sys.modules["training"] = training_mod

    monkeypatch.setattr(cli.scraper_wiki, "main", fake_main)
    monkeypatch.setattr(cli.scraper_wiki.Config, "OUTPUT_DIR", str(tmp_path))
    (tmp_path / "wikipedia_qa.json").write_text("[]")

    runner = CliRunner()
    result = runner.invoke(cli.app, ["scrape", "--lang", "pt", "--train"])
    assert result.exit_code == 0
    assert called.get("train")


def test_scrape_distributed(monkeypatch):
    fake_future = SimpleNamespace(result=lambda: None)

    class FakeClient:
        def submit(self, fn, *a, **k):
            return fake_future

    captured = {}

    def fake_main(lang, category, fmt, rate_limit_delay=None, **kwargs):
        captured["client"] = kwargs.get("client")

    import cluster

    monkeypatch.setattr(cluster, "get_client", lambda: FakeClient())
    monkeypatch.setattr(cli.scraper_wiki, "main", fake_main)
    runner = CliRunner()
    result = runner.invoke(cli.app, ["scrape", "--distributed"])
    assert result.exit_code == 0
    assert captured["client"]


def test_scrape_start_page_option(monkeypatch):
    called = {}

    def fake_main(
        lang,
        category,
        fmt,
        rate_limit_delay=None,
        start_pages=None,
        depth=1,
        client=None,
        **kwargs,
    ):
        called["args"] = {
            "lang": lang,
            "category": category,
            "fmt": fmt,
            "delay": rate_limit_delay,
            "start": start_pages,
            "depth": depth,
        }

    monkeypatch.setattr(cli.scraper_wiki, "main", fake_main)
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["scrape", "--start-page", "Python", "--depth", "2"],
    )
    assert result.exit_code == 0
    assert called["args"] == {
        "lang": None,
        "category": None,
        "fmt": "all",
        "delay": None,
        "start": ["Python"],
        "depth": 2,
    }


def test_scrape_revisions_option(monkeypatch):
    captured = {}

    def fake_main(
        lang,
        category,
        fmt,
        rate_limit_delay=None,
        start_pages=None,
        depth=1,
        revisions=False,
        rev_limit=5,
        client=None,
    ):
        captured["rev"] = (revisions, rev_limit)

    monkeypatch.setattr(cli.scraper_wiki, "main", fake_main)
    runner = CliRunner()
    result = runner.invoke(cli.app, ["scrape", "--revisions", "--rev-limit", "3"])
    assert result.exit_code == 0
    assert captured["rev"] == (True, 3)


def test_scrape_incremental_option(monkeypatch):
    captured = {}

    def fake_run_plugin(plg, langs, categories, fmt, incremental=False):
        captured["inc"] = incremental

    import plugins

    monkeypatch.setattr(plugins, "run_plugin", fake_run_plugin)
    monkeypatch.setattr(plugins, "load_plugin", lambda name: object())
    runner = CliRunner()
    result = runner.invoke(
        cli.app, ["scrape", "--plugin", "stackexchange", "--incremental"]
    )
    assert result.exit_code == 0
    assert captured.get("inc") is True


def test_scrape_incremental_wikipedia(monkeypatch):
    captured = {}

    def fake_main(*args, **kwargs):
        captured["inc"] = kwargs.get("incremental")

    monkeypatch.setattr(cli.scraper_wiki, "main", fake_main)
    runner = CliRunner()
    result = runner.invoke(cli.app, ["scrape", "--incremental"])
    assert result.exit_code == 0
    assert captured.get("inc") is True


def test_start_crawler_cli(monkeypatch):
    called = {}
    import crawling.distributed as dist

    monkeypatch.setattr(
        dist, "start_crawler", lambda path=None: called.setdefault("ok", True)
    )
    runner = CliRunner()
    result = runner.invoke(cli.app, ["start-crawler"])
    assert result.exit_code == 0
    assert called.get("ok")


def test_stop_crawler_cli(monkeypatch):
    called = {}
    import crawling.distributed as dist

    monkeypatch.setattr(dist, "stop_crawler", lambda: called.setdefault("ok", True))
    runner = CliRunner()
    result = runner.invoke(cli.app, ["stop-crawler"])
    assert result.exit_code == 0
    assert called.get("ok")


def test_process_pipeline_command(tmp_path, monkeypatch):
    data_file = tmp_path / "data.json"
    data_file.write_text("[]")

    pp_mod = ModuleType("processing.pipeline")
    pp_mod.get_pipeline = lambda name="default": (lambda data: [{"done": True}])
    sys.modules["processing.pipeline"] = pp_mod

    runner = CliRunner()
    result = runner.invoke(
        cli.app, ["process", str(data_file), "--pipeline", "default"]
    )
    assert result.exit_code == 0
    output = json.loads(data_file.read_text(encoding="utf-8"))
    assert output == [{"done": True}]


def test_auto_scrape_command(monkeypatch):
    records = []

    class DummyScraper:
        def __init__(
            self, base_url, driver_path=None, headless=True, backend="selenium"
        ):
            self.base_url = base_url
            self.driver = SimpleNamespace(page_source="")

        def fetch_page(self, url):
            records.append(url)
            return {"url": url}

        def close(self):
            pass

    auto_mod = ModuleType("crawling.auto_learner")
    auto_mod.AutoLearnerScraper = DummyScraper
    sys.modules["crawling.auto_learner"] = auto_mod
    import importlib

    core = importlib.import_module("core")

    monkeypatch.setattr(core, "AutoLearnerScraper", DummyScraper)
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["auto-scrape", "http://example.com/page", "--depth", "0", "--threads", "2"],
    )

    assert result.exit_code == 0
    assert records == ["http://example.com/page"]
    assert json.loads(result.output) == [{"url": "http://example.com/page"}]


def test_auto_scrape_backend_option(monkeypatch):
    called = {}

    class DummyScraper:
        def __init__(
            self, base_url, driver_path=None, headless=True, backend="selenium"
        ):
            called["backend"] = backend
            self.base_url = base_url
            self.driver = SimpleNamespace(page_source="")

        def fetch_page(self, url):
            return {"url": url}

        def close(self):
            pass

    auto_mod = ModuleType("crawling.auto_learner")
    auto_mod.AutoLearnerScraper = DummyScraper
    sys.modules["crawling.auto_learner"] = auto_mod
    import importlib

    core = importlib.import_module("core")

    monkeypatch.setattr(core, "AutoLearnerScraper", DummyScraper)
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "--scraper-backend",
            "playwright",
            "auto-scrape",
            "http://example.com/page",
            "--depth",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert called["backend"] == "playwright"


def test_build_graph_command(tmp_path, monkeypatch):
    triples = [{"subject": "Ada", "relation": "worked at", "object": "IBM"}]
    data_file = tmp_path / "triples.json"
    data_file.write_text(json.dumps(triples))

    monkeypatch.setitem(
        sys.modules,
        "integrations.neo4j_backend",
        SimpleNamespace(save_graph=lambda g: None),
    )

    runner = CliRunner()
    result = runner.invoke(cli.app, ["build-graph", str(data_file), "--persist"])
    assert result.exit_code == 0
