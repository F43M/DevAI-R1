# Building the Documentation

Install `sphinx` and then run:

```bash
sphinx-build -b html docs docs/_build
```

The HTML files will be created under `docs/_build`. Open `index.html` in your browser to view the API docs.

To generate the simplified HTML documentation using `pdoc` run:

```bash
pdoc -o docs -d google integrations core plugins utils api
```
