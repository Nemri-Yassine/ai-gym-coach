"""Skip @pytest.mark.data tests when the raw datasets are not on disk."""
import pytest

from awc.config import EC3D_DIR, MMFIT_DIR


def _data_ready():
    mmfit = MMFIT_DIR.is_dir() and any(MMFIT_DIR.glob("w*"))
    ec3d = (EC3D_DIR / "data_3D.pickle").is_file()
    return mmfit and ec3d


def pytest_collection_modifyitems(config, items):
    if _data_ready():
        return
    skip = pytest.mark.skip(reason="data/raw/mm-fit or data/raw/ec3d/data_3D.pickle is missing")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)
