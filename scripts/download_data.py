#!/usr/bin/env python3
"""
Download the real Lending Club sample used by this project.

Source (cleaned sample):
  https://github.com/gedeck/practical-statistics-for-data-scientists
  (derived from public Lending Club loan data)

Usage (from project root):
  python scripts/download_data.py
"""

from __future__ import annotations

import gzip
import shutil
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
GZ_URL = (
    "https://github.com/gedeck/practical-statistics-for-data-scientists/"
    "raw/master/data/loan_data.csv.gz"
)
GZ_PATH = RAW_DIR / "loan_data_lc.csv.gz"
CSV_PATH = RAW_DIR / "loan_data_lc.csv"


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 1_000_000:
        print(f"Already present: {CSV_PATH} ({CSV_PATH.stat().st_size:,} bytes)")
        return 0

    print(f"Downloading:\n  {GZ_URL}")
    try:
        urllib.request.urlretrieve(GZ_URL, GZ_PATH)
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        print(
            "\nManual fallback:\n"
            "  1. Open the URL above in a browser\n"
            "  2. Save the file as data/raw/loan_data_lc.csv.gz\n"
            "  3. Re-run this script (or gunzip it yourself)",
            file=sys.stderr,
        )
        return 1

    print(f"Saved compressed file → {GZ_PATH}")
    print("Decompressing...")
    with gzip.open(GZ_PATH, "rb") as f_in:
        with open(CSV_PATH, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f"Ready: {CSV_PATH} ({CSV_PATH.stat().st_size:,} bytes)")
    print("You can now run:  python -m src.models.train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
