"""Sanity checks on the raw datasets. Run: python -m pytest tests"""
import pytest

from awc.data import ec3d, mmfit


@pytest.mark.data
def test_mmfit_sessions():
    s = mmfit.sessions()
    assert len(s) == 21
    assert len(mmfit.load_labels("w00")) == 30


@pytest.mark.data
def test_mmfit_heart_rate_missing_where_expected():
    missing = [s for s in mmfit.sessions() if mmfit.load_stream(s, "sw_l_hr") is None]
    assert missing == ["w05", "w10", "w18"]


@pytest.mark.data
def test_ec3d_shapes_and_reps():
    d = ec3d.load()
    assert d["poses"].shape == (29789, 3, 25)
    assert len(ec3d.reps(d)) == 371
