"""Environment checks: imports, a Spark Parquet round-trip, experiment tracking."""
import json
import shutil

from awc.config import TMP


def test_imports():
    import cv2  # noqa: F401
    import mediapipe  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import pyspark  # noqa: F401
    import torch  # noqa: F401


def test_spark_roundtrip():
    from awc.spark import get_spark

    out = TMP / "test_parquet"
    if out.exists():
        shutil.rmtree(out)
    spark = get_spark("test", "local[2]")
    df = spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["id", "name"])
    df.write.mode("overwrite").parquet(out.as_posix())
    back = spark.read.parquet(out.as_posix())
    assert back.count() == 3


def test_tracking():
    from awc.training.tracking import new_run

    run_dir = new_run("unit", {"purpose": "test"})
    try:
        payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        assert "git_commit" in payload
        assert payload["purpose"] == "test"
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
