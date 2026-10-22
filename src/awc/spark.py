"""Local SparkSession. Scratch files stay in C:/awc_tmp, never inside OneDrive."""
import os
import subprocess
import sys
from pathlib import Path

from pyspark.sql import SparkSession

from awc.config import HADOOP_HOME, ROOT, SPARK_LOCAL_DIR, SPARK_WAREHOUSE, TMP


def _ensure_java():
    """Point JAVA_HOME at a local JDK 17 if the shell has not picked it up yet."""
    if os.environ.get("JAVA_HOME"):
        return
    roots = [Path("C:/Java"), TMP]
    for root in roots:
        if not root.exists():
            continue
        for cand in sorted(root.glob("jdk-17*")):
            if (cand / "bin" / "java.exe").exists():
                os.environ["JAVA_HOME"] = str(cand)
                os.environ["PATH"] = str(cand / "bin") + os.pathsep + os.environ.get("PATH", "")
                return


def _prepare_dirs():
    SPARK_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    SPARK_WAREHOUSE.mkdir(parents=True, exist_ok=True)
    _ensure_java()
    if HADOOP_HOME.exists():
        home = str(HADOOP_HOME)
        os.environ.setdefault("HADOOP_HOME", home)
        os.environ.setdefault("hadoop.home.dir", home)
        bin_dir = str(HADOOP_HOME / "bin")
        path = os.environ.get("PATH", "")
        if bin_dir.lower() not in path.lower():
            os.environ["PATH"] = bin_dir + os.pathsep + path
    src = str(ROOT / "src")
    py_path = os.environ.get("PYTHONPATH", "")
    if src.lower() not in py_path.lower():
        os.environ["PYTHONPATH"] = src + (os.pathsep + py_path if py_path else "")
    # Spark's Windows scripts split on spaces. Point them at a junction instead.
    python = _spark_python()
    os.environ["PYSPARK_PYTHON"] = python
    os.environ["PYSPARK_DRIVER_PYTHON"] = python


def _spark_python():
    exe = Path(sys.executable)
    if " " not in str(exe):
        return str(exe)
    link = TMP / "venv"
    target = exe.parent.parent  # .../.venv
    if not link.exists():
        subprocess.check_call(["cmd", "/c", "mklink", "/J", str(link), str(target)])
    return str(link / "Scripts" / "python.exe")


def spark_path(path) -> str:
    """Path string Spark can write. Junction around folders whose path contains spaces."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if " " not in str(path):
        return path.resolve().as_posix()
    link_parent = TMP / "links"
    link_parent.mkdir(parents=True, exist_ok=True)
    link = link_parent / path.name
    if not link.exists():
        subprocess.check_call(["cmd", "/c", "mklink", "/J", str(link), str(path.resolve())])
    return link.resolve().as_posix()


def get_spark(app_name, cores="local[*]"):
    """Build (or reuse) a local SparkSession.

    cores is a Spark master string, e.g. "local[2]" or "local[*]".
    """
    _prepare_dirs()
    return (
        SparkSession.builder.appName(app_name)
        .master(cores)
        .config("spark.local.dir", str(SPARK_LOCAL_DIR))
        .config("spark.sql.warehouse.dir", SPARK_WAREHOUSE.as_posix())
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.extraJavaOptions", f"-Djava.io.tmpdir={TMP.as_posix()}")
        .getOrCreate()
    )


def stop_spark():
    """Stop the active session so a later get_spark can change the master."""
    active = SparkSession.getActiveSession()
    if active is not None:
        active.stop()
