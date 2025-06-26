# Development Guidelines

This repository uses **pytest** for the test suite and `pdoc` for HTML documentation generation. Follow the instructions below when contributing.

## Installing Dependencies

Install the main dependencies with:

```bash
pip install -r requirements.txt
```

The following packages are heavy and optional for a minimal setup:

- `tensorflow`
- `sentence-transformers`
- `transformers`
- `spacy`

Skip them if you only want to run the lightweight parts of the project or the test suite.

## Running Tests

Execute the full test suite from the repository root:

```bash
pytest
```

All tests should pass before any commit.

## Formatting and Style

Code should follow **PEP8** and be formatted with **black** (88 characters per line). Docstrings are written in Google style. Run `black` on all modified files before committing.

## Generating Documentation

Generate HTML documentation using `pdoc`:

```bash
pdoc -o docs -d google integrations core plugins utils api
```

The output will be placed in the `docs/` directory.

## Repository Structure

- `cli.py`, `click_cli.py` – command line interfaces.
- `api/api_app.py` – FastAPI server exposing the scraping API.
- `plugins/` – optional extensions such as `wikidata` and `stackoverflow` parsers.
- `utils/` – text cleaning and helper modules.
- `training/` – utilities for training machine learning models.
- `tests/` – pytest suite covering the whole project.
- `examples/` – sample notebooks and usage demonstrations.

## Contributor Workflow

1. Update or add docstrings when modifying code.
2. Run `pytest` and ensure all tests succeed.
3. Commit your changes only after tests pass.
