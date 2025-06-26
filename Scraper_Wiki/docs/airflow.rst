Airflow Deployment
==================

The ``training.airflow_pipeline`` module provides DAGs that automate dataset
creation and publication. To run the pipeline you must configure a few
environment variables and start Airflow.

Environment Variables
---------------------

``LANGUAGES``
  Comma separated list of languages to scrape. Defaults to the values in
  :class:`scraper_wiki.Config`.
``CATEGORIES``
  Comma separated list of categories to scrape.
``STORAGE_BACKEND``
  Name of the storage backend used when publishing the dataset. ``local`` and
  ``s3`` are supported out of the box.
``OUTPUT_DIR``
  Directory used by the ``local`` storage backend.
``AIRFLOW_SCHEDULE``
  Cron expression with the desired schedule for the DAG.

Scheduling
----------

With the variables exported you can launch Airflow using the provided
``docker-compose.airflow.yml`` file::

    docker compose -f docker-compose.airflow.yml up

The pipeline will run according to ``AIRFLOW_SCHEDULE``. For a daily run at 3 AM
set::

    export AIRFLOW_SCHEDULE="0 3 * * *"
