# Usage

## CLI Examples

Scrape immediately:

```bash
python cli.py scrape --lang pt --category "Programação" --format json
```

Queue a job:

```bash
python cli.py queue --lang en --category "Algorithms"
```

Start monitoring dashboard:

```bash
python cli.py monitor
```

## API Requests

Run the API:

```bash
uvicorn api.api_app:app --reload
```

Generate dataset:

```bash
curl -X POST http://localhost:8000/scrape -H "Content-Type: application/json" -d '{"lang": ["pt"], "category": ["Programação"], "format": "json"}'
```

Retrieve records:

```bash
curl "http://localhost:8000/records?lang=pt&category=Programação"
```

## Plugins

Enable a plugin with `--plugin` or via API payload. Available options include `infobox_parser`, `table_parser`, `github_scraper`, `wikidata`, `stackoverflow` and more.
