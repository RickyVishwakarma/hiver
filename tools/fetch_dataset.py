"""Stage A.1 - acquire the Customer Support on Twitter dataset.

Two paths, because Kaggle auth is a common point of failure and we do not want
the pipeline blocked on it:

  1. API:    python tools/fetch_dataset.py
             Needs a KGAT_ token in ~/.kaggle/access_token (see .env.example).

  2. Manual: python tools/fetch_dataset.py --zip ~/Downloads/archive.zip
             Download the dataset in a browser from
             https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
             and point this at the zip. No account automation required.

Idempotent: if .tmp/raw/twcs.csv already exists and looks sane, does nothing.
"""
from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path

from common import RAW, TWCS_CSV, die, info, ok, warn

DATASET = "thoughtvector/customer-support-on-twitter"
EXPECTED_COLUMNS = {
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
}
# The real file is ~500MB / ~2.8M rows. Anything much smaller means we grabbed
# a partial download or the wrong file.
MIN_BYTES = 100_000_000

CRED_HELP = """
    kaggle.com -> avatar -> Settings -> API -> Generate New Token
    then: mkdir -p ~/.kaggle && echo <KGAT_token> > ~/.kaggle/access_token
    (the legacy KAGGLE_USERNAME + KAGGLE_KEY pair still works too)
    or skip auth entirely:
      python tools/fetch_dataset.py --zip <path-to-archive.zip>"""


def _verify(path: Path) -> bool:
    """Cheap sanity check: size + header columns. Avoids reading 500MB."""
    if not path.exists():
        return False
    size = path.stat().st_size
    if size < MIN_BYTES:
        warn(f"{path.name} is only {size/1e6:.1f}MB - expected >{MIN_BYTES/1e6:.0f}MB")
        return False
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().strip().lstrip("﻿")
    cols = {c.strip().strip('"') for c in header.split(",")}
    missing = EXPECTED_COLUMNS - cols
    if missing:
        warn(f"{path.name} header missing expected columns: {sorted(missing)}")
        return False
    info(f"verified {path.name}: {size/1e6:.0f}MB, columns OK")
    return True


def _extract(zip_path: Path) -> None:
    info(f"extracting {zip_path.name} -> {RAW}")
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not members:
            die(f"no .csv inside {zip_path}")
        # The archive contains a single twcs.csv (sometimes nested one level).
        target = max(members, key=lambda m: zf.getinfo(m).file_size)
        info(f"found {target} ({zf.getinfo(target).file_size/1e6:.0f}MB uncompressed)")
        with zf.open(target) as src, TWCS_CSV.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)


def _download_via_api() -> Path:
    """Use the kaggle CLI package. Credentials come from env or ~/.kaggle."""
    # Kaggle now issues a single KGAT_ token; the old username+key pair is the
    # legacy path. Accept any of the four places credentials can live.
    kaggle_dir = Path.home() / ".kaggle"
    has_creds = (
        os.environ.get("KAGGLE_API_TOKEN")
        or (kaggle_dir / "access_token").exists()
        or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
        or (kaggle_dir / "kaggle.json").exists()
    )
    if not has_creds:
        die(
            "No Kaggle credentials found. Current flow:"
            + CRED_HELP
        )

    try:
        # Imported lazily: the reproduction path must not require this package.
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        die("kaggle package not installed. Run: pip install kaggle")

    api = KaggleApi()
    api.authenticate()
    info(f"downloading {DATASET} via Kaggle API (~180MB zip)...")
    api.dataset_download_files(DATASET, path=str(RAW), unzip=False, quiet=False)
    zips = sorted(RAW.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        die(f"Kaggle reported success but no .zip landed in {RAW}")
    return zips[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, help="Path to a manually downloaded archive.zip")
    ap.add_argument("--force", action="store_true", help="Re-extract even if twcs.csv exists")
    args = ap.parse_args()

    if not args.force and _verify(TWCS_CSV):
        ok(f"dataset already present at {TWCS_CSV} - nothing to do")
        return

    zip_path = args.zip.expanduser() if args.zip else _download_via_api()
    if not zip_path.exists():
        die(f"zip not found: {zip_path}")

    _extract(zip_path)

    if not _verify(TWCS_CSV):
        die("extraction finished but verification failed - inspect .tmp/raw/")
    ok(f"dataset ready: {TWCS_CSV}")
    info("next: python tools/profile_brands.py")


if __name__ == "__main__":
    main()
