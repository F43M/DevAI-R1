import asyncio
import csv
import json
from pathlib import Path
from urllib.parse import urlparse, unquote

import click
import pyarrow as pa
import pyarrow.parquet as pq

import scraper_wiki
from utils.text import translate_text


def _parse_wiki_url(url: str) -> tuple[str, str]:
    """Return (title, lang) extracted from a Wikipedia URL."""
    parsed = urlparse(url)
    lang = parsed.hostname.split('.')[0]
    title = parsed.path.split('/wiki/')[-1]
    return unquote(title), lang


@click.command()
@click.option("--url", help="URL de uma página da Wikipédia")
@click.option(
    "--file",
    "urls_file",
    type=click.Path(exists=True, dir_okay=False),
    help="Arquivo com URLs, um por linha",
)
@click.option(
    "--output",
    "output_format",
    type=click.Choice(["json", "csv", "parquet"], case_sensitive=False),
    default="json",
    show_default=True,
    help="Formato de saída",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False),
    default=".",
    show_default=True,
    help="Diretório para salvar o resultado",
)
@click.option(
    "--translate",
    "translate_to",
    help="Traduz o texto para o idioma especificado",
)
def main(url: str | None, urls_file: str | None, output_format: str, out_dir: str, translate_to: str | None) -> None:
    """Baixa o HTML de páginas da Wikipédia e salva em formatos variados."""
    if not url and not urls_file:
        raise click.UsageError("Informe --url ou --file")

    urls = []
    if url:
        urls.append(url)
    if urls_file:
        urls.extend([u.strip() for u in Path(urls_file).read_text().splitlines() if u.strip()])

    records = []
    for u in urls:
        title, lang = _parse_wiki_url(u)
        html = asyncio.run(scraper_wiki.fetch_html_content_async(title, lang))
        if translate_to:
            html = translate_text(html, translate_to)
        records.append({"url": u, "html": html})

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    out_path = out_dir_path / f"output.{output_format.lower()}"
    if output_format.lower() == "json":
        out_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    elif output_format.lower() == "csv":
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["url", "html"])
            writer.writeheader()
            writer.writerows(records)
    else:  # parquet
        table = pa.Table.from_pylist(records)
        pq.write_table(table, out_path)

    click.echo(f"Arquivo salvo em {out_path} ({len(records)} página(s))")


if __name__ == "__main__":
    main()
