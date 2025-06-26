"""PySpark preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode


def load_dataset(path: str | Path) -> List[Dict]:
    """Load JSON dataset using Spark."""

    spark = SparkSession.builder.getOrCreate()
    df = spark.read.json(str(path))
    return [row.asDict() for row in df.collect()]


def run_spark_pipeline(path: str | Path, out_dir: str | Path) -> None:
    """Simple Spark job that explodes questions and answers."""

    spark = SparkSession.builder.getOrCreate()
    df = spark.read.json(str(path))
    qa = df.select(
        explode(col("questions")).alias("q"), explode(col("answers")).alias("a")
    )
    qa.write.json(str(Path(out_dir) / "qa_pairs"), mode="overwrite")
