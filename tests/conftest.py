"""Skip @pytest.mark.data / kafka tests when the matching dependency is absent."""
import pytest

from awc.config import EC3D_DIR, MMFIT_DIR


def _data_ready():
    mmfit = MMFIT_DIR.is_dir() and any(MMFIT_DIR.glob("w*"))
    ec3d = (EC3D_DIR / "data_3D.pickle").is_file()
    return mmfit and ec3d


def _kafka_ready():
    try:
        from awc.streaming.kafka_io import broker_up
        return broker_up()
    except Exception:  # noqa: BLE001
        return False


def pytest_collection_modifyitems(config, items):
    skip_data = None if _data_ready() else pytest.mark.skip(
        reason="data/raw/mm-fit or data/raw/ec3d/data_3D.pickle is missing"
    )
    skip_kafka = None if _kafka_ready() else pytest.mark.skip(reason="Kafka broker not reachable")
    for item in items:
        if skip_data is not None and "data" in item.keywords:
            item.add_marker(skip_data)
        if skip_kafka is not None and "kafka" in item.keywords:
            item.add_marker(skip_kafka)
