import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Scraper-Wiki"
extensions = ["sphinx.ext.autodoc"]
autodoc_mock_imports = [
    "sentence_transformers",
    "datasets",
    "spacy",
    "unidecode",
    "tqdm",
    "html2text",
    "wikipediaapi",
    "aiohttp",
    "backoff",
    "sklearn",
    "sumy",
    "streamlit",
    "psutil",
    "prometheus_client",
    "tensorflow",
    "transformers",
    "googletrans",
]
exclude_patterns = ["_build"]
html_theme = "alabaster"
