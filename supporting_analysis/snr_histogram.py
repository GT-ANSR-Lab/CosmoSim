#!/usr/bin/env python3
import sys
import pickle
import re
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sgp4.api import Satrec, jday
from astropy.coordinates import TEME, ITRS
from astropy.time import Time
import astropy.units as u

CURRENT_DIR = Path(__file__).resolve().parent
COSMOSIM_ROOT = CURRENT_DIR.parent
if str(COSMOSIM_ROOT) not in sys.path:
    sys.path.insert(0, str(COSMOSIM_ROOT))
BASE_DATA_DIR = COSMOSIM_ROOT / "data"
TLE_FILE = COSMOSIM_ROOT / "constellation_configurations" / "configs" / "starlink_5shells" / "tles.txt"

C = 299792458.0
FREQ = 10.7e9
T_SYS = 200.0
k = 1.38064852e-23
B = 250e6

N0_WATT = k * T_SYS * B
N0_DB = 10 * np.log10(N0_WATT)

D_UT = 0.6
D_ST = 0.863
lam = C / FREQ
D_LAMBDA_RX = D_UT / lam
D_LAMBDA_TX = D_ST / lam

G_SAT_MAX = 20 * np.log10(D_LAMBDA_TX) + 7.7
G_UT_MAX = 20 * np.log10(D_LAMBDA_RX) + 7.7
EIRP_DENSITY_MAX = -51.1
P_TX_DENSITY = EIRP_DENSITY_MAX - G_SAT_MAX

EPOCH_BASE = datetime(2022, 1, 1)

def ecef(lat, lon, alt=0.0):
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2 * f - f ** 2
    
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    
    N = a / np.sqrt(1.0 - e2 * (sin_lat ** 2))
    
    x = (N + alt) * cos_lat * np.cos(lon_rad)
    y = (N + alt) * cos_lat * np.sin(lon_rad)
    z = (N * (1.0 - e2) + alt) * sin_lat
    return np.array([x, y, z])

def get_satellite_positions(tle_path, time_offset_seconds):
    sats = {}
    if not tle_path.exists():
        raise FileNotFoundError(f"TLE File missing at {tle_path}")
    with open(tle_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    while i < len(lines) - 2:
        m = re.search(r"Starlink 5-Shells (\d+)", lines[i])
        if m:
            sat = int(m.group(1))
            sats[sat] = Satrec.twoline2rv(lines[i+1], lines[i+2])
            i += 3
        else:
            i += 1
            
    positions = {}
    t = EPOCH_BASE + timedelta(seconds=float(time_offset_seconds))
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond / 1e6)
    
    obstime = Time(t, scale="utc")
    for sid, rec in tqdm(sats.items(), desc="Propagating satellites"):
        e, r, v = rec.sgp4(jd, fr)
        if e == 0:
            teme = TEME(x=r[0]*u.km, y=r[1]*u.km, z=r[2]*u.km, obstime=obstime)
            itrs = teme.transform_to(ITRS(obstime=obstime))
            positions[sid] = np.array([itrs.x.to(u.m).value, itrs.y.to(u.m).value, itrs.z.to(u.m).value])
    return positions

def calculate_single_beam_snr(u_tgt_pos, sat_tgt_pos):
    d = np.linalg.norm(u_tgt_pos - sat_tgt_pos)
    eirp_dbw = P_TX_DENSITY + G_SAT_MAX + 10 * np.log10(B)
    fspl_db = 20 * np.log10(d) + 20 * np.log10(FREQ) - 20 * np.log10(C / (4 * np.pi))
    S_dbw = eirp_dbw + G_UT_MAX - fspl_db
    snr_db = S_dbw - N0_DB
    return snr_db, d

def main():
    country = "haiti"
    pop = 1000
    ut_dist = "population"
    routing = "greedy-coordinated"
    time_seconds = 0
    time_nanoseconds = time_seconds * 1_000_000_000
    
    folder_name = f"starlink_5shells_ground_stations_starlink_cells_{country}_0_{pop}_{ut_dist}_{routing}"
    target_pkl_path = BASE_DATA_DIR / folder_name / f"beam_assignments_{time_nanoseconds}.pkl"
    
    if not target_pkl_path.exists():
        return

    sat_positions = get_satellite_positions(TLE_FILE, time_seconds)
    
    with target_pkl_path.open("rb") as f:
        raw_data = pickle.load(f)
        
    import h3
    
    records = []
    for key, val in tqdm(raw_data.items(), desc="Processing beam assignments"):
        cell, slot_str = key.rsplit("_", 1)
        slot = int(slot_str)
        reuse, sat_id_str, _ = val.split("_")
        sat_id = int(sat_id_str)
        
        if sat_id not in sat_positions:
            continue
            
        lat, lon = h3.cell_to_latlng(cell)
        u_ecef = ecef(lat, lon, alt=0.0)
        sat_ecef = sat_positions[sat_id]
        
        snr, slant_range_m = calculate_single_beam_snr(u_ecef, sat_ecef)
        
        records.append({
            "Cell ID": cell,
            "Channel Slot": f"Channel {slot}",
            "Serving Sat ID": sat_id,
            "Slant Range (km)": slant_range_m / 1000.0,
            "SNR (dB)": snr
        })

    df_snr = pd.DataFrame(records)
    if df_snr.empty:
        return

    plt.figure(figsize=(9, 4))
    
    snr_values = df_snr["SNR (dB)"]
    bin_edges = np.arange(np.floor(snr_values.min()), np.ceil(snr_values.max()) + 0.25, 0.25)

    plt.hist(
        snr_values, 
        bins=bin_edges, 
        alpha=0.7, 
        color="#1f77b4", 
        edgecolor="black",
        label=f"Cell Links (N={len(df_snr)})"
    )

    median_snr = snr_values.median()
    plt.axvline(
        median_snr, 
        color="red", 
        linestyle="--", 
        linewidth=2, 
        label=f"Median SNR: {median_snr:.2f} dB"
    )

    plt.title(f" SNR (dB) Distribution Histogram (t = {time_seconds}s)", fontsize=12, fontweight="bold")
    plt.xlabel("SNR (dB)", fontsize=11)
    plt.ylabel("Frequency (Number of Links)", fontsize=11)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()