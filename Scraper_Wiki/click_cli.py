import asyncio
import json
import csv
from pathlib import Path
from urllib.parse import urlparse, unquote

import click

import scraper_wiki


def _parse_wiki_url(url: str) -> tuple[str, str]:
    """Return (title, lang) extracted from a Wikipedia URL."""
    parsed = urlparse(url)
    lang = parsed.hostname.split('.')[0]
    title = parsed.path.split('/wiki/')[-1]
    return unquote(title), lang


@click.command()
@click.option('--url', required=True, help='URL completo da página da Wikipédia')
@click.option('--output', required=True, type=click.Path(), help='Arquivo de saída (.json ou .csv)')
def main(url: str, output: str) -> None:
    """Baixa o HTML de uma página da Wikipédia e salva em JSON ou CSV."""
    title, lang = _parse_wiki_url(url)
    html = asyncio.run(scraper_wiki.fetch_html_content_async(title, lang))

    path = Path(output)
    if path.suffix.lower() == '.json':
        data = {'url': url, 'html': html}
        path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    elif path.suffix.lower() == '.csv':
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['url', 'html'])
            writer.writerow([url, html])
    else:
        click.echo('Formato não suportado. Use extensão .json ou .csv.', err=True)
        raise SystemExit(1)
    click.echo(f'Arquivo salvo em {path}')


if __name__ == '__main__':
    main()
