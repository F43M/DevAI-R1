Scraper-Wiki Documentation
==========================

.. toctree::
   :maxdepth: 2

   modules
   code_fine_tuning
   postprocessing
   plugins/index
   airflow

Record Schema
-------------

Example dataset entry:

.. code-block:: json

    {
        "id": "123abc",
        "title": "Title",
        "language": "en",
        "category": "History",
        "quality_score": 0.0,
        "tests": [],
        "context": "Ada summary",
        "docstring": "",
        "raw_code": "",
        "problems": [],
        "fixed_version": "",
        "lessons": "",
        "origin_metrics": {},
        "challenge": "",
        "images": [],
        "metadata": {
            "length": 1000,
            "source": "wikipedia",
            "source_url": "https://en.wikipedia.org/wiki/Title",
            "entity_ids": ["Q1"]
        }
    }
