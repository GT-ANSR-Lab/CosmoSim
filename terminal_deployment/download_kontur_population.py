#!/usr/bin/env python3
"""Download Kontur population GeoPackage files for CosmoSim."""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DEST_ROOT = ROOT / "terminal_deployment" / "shp_files"

BLOB_BASE = (
    "https://storage.googleapis.com/kontur-datasets/kontur_population_20220630/"
)

DATASETS = {
    "southafrica": {
        "blob": "kontur_population_zaf_20220630.gpkg.gz",
        "target": DEST_ROOT / "southafrica" / "southafrica.gpkg",
    },
    "ghana": {
        "blob": "kontur_population_gha_20220630.gpkg.gz",
        "target": DEST_ROOT / "ghana" / "ghana.gpkg",
    },
    "britain": {
        "blob": "kontur_population_gbr_20220630.gpkg.gz",
        "target": DEST_ROOT / "britain" / "britain.gpkg",
    },
    "haiti": {
        "blob": "kontur_population_hti_20220630.gpkg.gz",
        "target": DEST_ROOT / "haiti" / "haiti.gpkg",
    },
    "lithuania": {
        "blob": "kontur_population_ltu_20220630.gpkg.gz",
        "target": DEST_ROOT / "lithuania" / "lithuania.gpkg",
    },
    "tonga": {
        "blob": "kontur_population_ton_20220630.gpkg.gz",
        "target": DEST_ROOT / "tonga" / "tonga.gpkg",
    },
    "malaysia": {
        "blob": "kontur_population_mys_20220630.gpkg.gz",
        "target": DEST_ROOT / "malaysia" / "malaysia.gpkg",
    },
}

def download_and_extract(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_archive = destination.with_suffix(destination.suffix + ".download")

    print(f"Downloading {url}")
    with urlopen(url) as response, open(tmp_archive, "wb") as fh:
        shutil.copyfileobj(response, fh)

    print(f"Extracting to {destination}")
    with gzip.open(tmp_archive, "rb") as src, open(destination, "wb") as dst:
        shutil.copyfileobj(src, dst)

    tmp_archive.unlink(missing_ok=True)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Kontur population GeoPackages used by CosmoSim."
    )
    parser.add_argument(
        "keys",
        nargs="*",
        choices=sorted(DATASETS.keys()),
        help="Optional dataset keys to download (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .gpkg files if present.",
    )
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    keys = args.keys or sorted(DATASETS.keys())

    for key in keys:
        dataset = DATASETS[key]
        target = dataset["target"]
        url = BLOB_BASE + dataset["blob"]
        if target.exists() and not args.force:
            print(f"Skipping {key}: {target} already exists (use --force to overwrite).")
            continue
        download_and_extract(url, target)

    print("Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
