"""Gym station: iPhone Safari camera over Wi-Fi + watch HR → laptop own_recordings."""

__all__ = ["create_app"]


def create_app(*args, **kwargs):
    from awc.station.app import create_app as _create

    return _create(*args, **kwargs)
