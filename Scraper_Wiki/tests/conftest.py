import sys
from types import SimpleNamespace, ModuleType


class _DummyMetric:
    def __init__(self, *a, **k):
        pass

    def inc(self, *a, **k):
        pass

    def observe(self, *a, **k):
        pass

    def set(self, *a, **k):
        pass


prometheus_stub = SimpleNamespace(
    Counter=lambda *a, **k: _DummyMetric(),
    Histogram=lambda *a, **k: _DummyMetric(),
    Gauge=lambda *a, **k: _DummyMetric(),
    start_http_server=lambda *a, **k: None,
)
sys.modules.setdefault("prometheus_client", prometheus_stub)

pyarrow_stub = ModuleType("pyarrow")
pyarrow_stub.Table = object
pyarrow_stub.parquet = SimpleNamespace(write_table=lambda *a, **k: None)
pyarrow_stub.ipc = SimpleNamespace()
pyarrow_stub.csv = SimpleNamespace(read_csv=lambda *a, **k: None)
pyarrow_stub.dataset = SimpleNamespace(write_dataset=lambda *a, **k: None)
sys.modules.setdefault("pyarrow", pyarrow_stub)
sys.modules.setdefault("pyarrow.parquet", pyarrow_stub.parquet)
sys.modules.setdefault("pyarrow.ipc", pyarrow_stub.ipc)
sys.modules.setdefault("pyarrow.csv", pyarrow_stub.csv)
sys.modules.setdefault("pyarrow.dataset", pyarrow_stub.dataset)


class _DummySession:
    def run(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class _DummyDriver:
    def session(self):
        return _DummySession()


neo4j_stub = SimpleNamespace(
    GraphDatabase=SimpleNamespace(driver=lambda *a, **k: _DummyDriver())
)
sys.modules.setdefault("neo4j", neo4j_stub)

pymilvus_stub = SimpleNamespace(
    connections=SimpleNamespace(connect=lambda *a, **k: None),
    Collection=lambda *a, **k: SimpleNamespace(insert=lambda *a1, **k1: None),
)
sys.modules.setdefault("pymilvus", pymilvus_stub)

weaviate_stub = SimpleNamespace(
    Client=lambda *a, **k: SimpleNamespace(insert=lambda *a1, **k1: None)
)
sys.modules.setdefault("weaviate", weaviate_stub)

spacy_stub = ModuleType("spacy")
spacy_stub.load = lambda *a, **k: None
sys.modules.setdefault("spacy", spacy_stub)

networkx_stub = ModuleType("networkx")
networkx_stub.Graph = object
sys.modules.setdefault("networkx", networkx_stub)

requests_stub = ModuleType("requests")
requests_stub.get = lambda *a, **k: SimpleNamespace(status_code=200, text="")
requests_stub.post = lambda *a, **k: SimpleNamespace(status_code=200, text="")
requests_stub.exceptions = SimpleNamespace(RequestException=Exception)
sys.modules.setdefault("requests", requests_stub)

sys.modules.setdefault("yaml", SimpleNamespace(safe_load=lambda *a, **k: {}))

# Stub optional browser dependencies
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
# Stub Playwright API
playwright_sync_mod = ModuleType("playwright.sync_api")


class _DummyPage:
    def goto(self, url):
        pass

    def content(self):
        return ""


class _DummyBrowser:
    def new_page(self):
        return _DummyPage()


class _DummyPlaywright:
    def __init__(self):
        self.chromium = SimpleNamespace(launch=lambda headless=True: _DummyBrowser())

    def start(self):
        return self

    def stop(self):
        pass


def sync_playwright():
    return _DummyPlaywright()


playwright_sync_mod.sync_playwright = sync_playwright
playwright_mod = ModuleType("playwright")
playwright_mod.sync_api = playwright_sync_mod
sys.modules.setdefault("playwright", playwright_mod)
sys.modules.setdefault("playwright.sync_api", playwright_sync_mod)

# Stub Selenium-based scraper to avoid heavy dependencies
auto_stub = ModuleType("crawling.auto_learner")


class _DummyAutoScraper:
    def __init__(self, base_url, driver_path=None, headless=True):
        self.base_url = base_url
        self.driver = SimpleNamespace(page_source="")

    def fetch_page(self, url):
        return {"url": url}

    def search(self, query):
        return []

    def build_dataset(self, queries, max_pages=5):
        return []

    def close(self):
        pass


auto_stub.AutoLearnerScraper = _DummyAutoScraper
sys.modules.setdefault("crawling.auto_learner", auto_stub)
