import sys
import types
import asyncio

import devai.scraper_interface as si


def test_run_scrape_main(monkeypatch):
    called = {}
    fake_module = types.ModuleType("scraper_wiki")
    def fake_main(langs, cats, fmt, rate_delay=None, *, start_pages=None, depth=1, **kwargs):
        called['langs'] = langs
        called['cats'] = cats
        called['depth'] = depth
    fake_module.main = fake_main
    sys.modules['scraper_wiki'] = fake_module

    async def fake_to_thread(func, *a, **k):
        return func(*a, **k)

    monkeypatch.setattr(si, 'asyncio', types.SimpleNamespace(to_thread=fake_to_thread))

    asyncio.run(si.run_scrape('AI', 'pt', 2))
    assert called == {'langs': ['pt'], 'cats': ['AI'], 'depth': 2}


def test_run_scrape_auto(monkeypatch):
    called = {}
    fake_cli = types.ModuleType('Scraper_Wiki.cli')
    def fake_auto(urls, depth=1, threads=2):
        called['urls'] = urls
        called['depth'] = depth
    fake_cli.auto_scrape = fake_auto
    sys.modules['Scraper_Wiki.cli'] = fake_cli

    async def fake_to_thread(func, *a, **k):
        return func(*a, **k)

    monkeypatch.setattr(si, 'asyncio', types.SimpleNamespace(to_thread=fake_to_thread))

    asyncio.run(si.run_scrape('http://example.com', None, 3, plugin='github'))
    assert called == {'urls': ['http://example.com'], 'depth': 3}

