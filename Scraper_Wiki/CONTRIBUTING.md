# Contributing to Scraper Wiki

Thank you for your interest in improving Scraper Wiki! This project follows a few simple rules to keep the code base consistent and maintainable.

## Setting up the environment

Install the main dependencies:

```bash
pip install -r requirements.txt
```

Heavy optional packages such as TensorFlow and transformers are not required for running the tests. Skip them if you only need a minimal environment.

## Running the test suite

All contributions must pass the pytest suite before being committed:

```bash
pytest
```

Run the command from the repository root. New features should include appropriate tests whenever possible.

## Code style

- Follow **PEP8** conventions.
- Format all modified files with **black** (line length 88).
- Write docstrings using the **Google** style.

You can format the entire repository by running:

```bash
black .
```

## Pre-commit hooks

Run security and formatting checks with [pre-commit](https://pre-commit.com):

```bash
pre-commit run --all-files
```

The hooks apply **black**, execute **bandit** and audit dependencies using
**pip-audit**.

## Generating documentation

HTML API documentation is generated with **pdoc**. After changing modules or docstrings, run:

```bash
pdoc -o docs -d google integrations core plugins utils api
```

The output will appear inside the `docs/` directory.

## Releases and Versioning

Scraper Wiki follows [Semantic Versioning](https://semver.org/) for all
releases. The current version is defined in `pyproject.toml` and tags are
created as `vMAJOR.MINOR.PATCH`.

- **MAJOR** version when you make incompatible API or dataset changes.
- **MINOR** version when you add functionality in a backward compatible manner.
- **PATCH** version when you make backward compatible bug fixes.


## Workflow

1. Update docstrings and documentation when modifying code.
2. Ensure `pytest` passes without failures.
3. Commit your changes once the code is formatted and tests succeed.

We welcome pull requests that improve the scraper, documentation or tests. Feel free to open issues if you encounter problems.
