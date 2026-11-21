"""Rebuild and print the own-recordings manifest.

    python scripts/own_manifest.py
    python scripts/own_manifest.py --print
"""
from __future__ import annotations

import argparse

from awc.config import OWN_MANIFEST
from awc.data.own_manifest import build_rows, print_table, write_manifest


def main():
    p = argparse.ArgumentParser(description="Build data/own_manifest.csv from own_recordings")
    p.add_argument("--print", action="store_true", help="rebuild and print the table")
    p.add_argument("--out", type=str, default=None, help="output CSV path (default: data/own_manifest.csv)")
    args = p.parse_args()
    path = write_manifest(path=args.out)
    rows = build_rows(print_overrides=False)
    print(f"wrote {path} ({len(rows)} rows)")
    if args.print:
        print_table(rows)


if __name__ == "__main__":
    main()
