#!/usr/bin/env python3
"""Standalone Generator for Satellite Beam Allocation Table across Channel Slots from CosmoSim PKL beam assignment files."""

import pickle
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd
import sys 

CURRENT_DIR = Path(__file__).resolve().parent
COSMOSIM_ROOT = CURRENT_DIR.parent
if str(COSMOSIM_ROOT) not in sys.path:
    sys.path.insert(0, str(COSMOSIM_ROOT))
BASE_DATA_DIR = COSMOSIM_ROOT / "data"

def generate_satellite_beam_table_standalone(pkl_path: Path):
    """Loads a PKL beam assignment file, aggregates active satellite beams per channel slot, and prints a tabular matrix."""
    print("=" * 80)
    print(f"      SATELLITE BEAM ALLOCATION TABLE GENERATOR")
    print(f"      Path: {pkl_path}")
    print("=" * 80)

    if not pkl_path.exists():
        print(f"[ERROR] Specified PKL file does not exist: {pkl_path}")
        return

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    active_transmitters = defaultdict(list)
    for k, v in raw.items():
        cell, slot_str = k.rsplit("_", 1)
        slot = int(slot_str)
        reuse, sat, satbeam = v.split("_")
        active_transmitters[slot].append({
            "cell_id": cell,
            "sat_id": int(sat),
            "reuse": int(reuse)
        })
    table_data = []
    for slot, tx_list in sorted(active_transmitters.items()):
        sat_counts = defaultdict(int)
        for tx in tx_list:
            sat_counts[tx["sat_id"]] += 1
        for sat_id, count in sat_counts.items():
            table_data.append({"Channel Slot": slot, "Satellite ID": sat_id, "Active Beams": count})

    if not table_data:
        print("  [INFO] No active transmitters found in the dataset.")
        return

    df = pd.DataFrame(table_data)
    
    pivot_df = df.pivot(index="Satellite ID", columns="Channel Slot", values="Active Beams").fillna(0).astype(int)
    pivot_df["Total Beams"] = pivot_df.sum(axis=1)
    
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.width', 1000)

    print("\n[OUTPUT] Satellite Beam Allocation Matrix (Rows: Satellites | Columns: Channel Slots):\n")
    print(pivot_df.to_string())
    print("\n" + "=" * 80)
    
    return pivot_df


if __name__ == "__main__":
    country = "haiti"
    pop = 1000
    ut_dist = "population"
    routing = "greedy-coordinated"
    time_sec = 0
    time_nsec = time_sec * 1_000_000_000
    
    folder = f"starlink_5shells_ground_stations_starlink_cells_{country}_0_{pop}_{ut_dist}_{routing}"
    pkl_file = BASE_DATA_DIR / folder / f"beam_assignments_{time_nsec}.pkl"

    generate_satellite_beam_table_standalone(pkl_file)