#!/usr/bin/env python3
"""One Cell INR Inspector for CosmoSim PKL Datasets with Interactive 2D Satellite Plotting."""

import pickle
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
import sys
import numpy as np
import pandas as pd
import h3

import matplotlib.pyplot as plt
import mplcursors  # Required for interactive hover tooltips

from sgp4.api import Satrec, jday
from astropy.coordinates import TEME, ITRS
from astropy.time import Time
import astropy.units as u

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
EIRP_DENSITY_MAX = -51.1
P_TX_DENSITY = EIRP_DENSITY_MAX - G_SAT_MAX

EPOCH_BASE = datetime(2022, 1, 1)
INR_THRESHOLD = -6.0

CURRENT_DIR = Path(__file__).resolve().parent
COSMOSIM_ROOT = CURRENT_DIR.parent
if str(COSMOSIM_ROOT) not in sys.path:
    sys.path.insert(0, str(COSMOSIM_ROOT))
BASE_DATA_DIR = COSMOSIM_ROOT / "data"
TLE_FILE = COSMOSIM_ROOT / "constellation_configurations" / "configs" / "starlink_5shells" / "tles.txt"

def norm(v):
    return v / (np.linalg.norm(v) + 1e-15)

def angle(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(norm(a), norm(b)), -1, 1)))

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

def ecef_to_latlon(xyz):
    x, y, z = xyz
    hypot = np.sqrt(x**2 + y**2)
    lat = np.degrees(np.arctan2(z, hypot))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon

def itu_s1528_tx(psi, D):
    psi = max(psi, 1e-9)
    G = 20 * np.log10(D) + 7.7
    pb = np.sqrt(1200) / D
    Y = 1.5 * pb
    Z = Y * 10**(0.04 * (G - 6.75))
    if psi <= Y:
        return G - 3 * (psi / pb)**2
    if psi <= Z:
        return G - 6.75 - 25 * np.log10(psi / Y)
    return 0

def itu_s1428_rx(phi, D):
    phi = max(phi, 1e-9)
    D = np.clip(D, 20, 25)
    Gmax = 20 * np.log10(D) + 7.7
    G1 = 29 - 25 * np.log10(95 / D)
    phim = np.sqrt((Gmax - G1) / 2.5e-3) / D
    phir = 95 / D
    if phi < phim:
        return Gmax - 2.5e-3 * (D * phi)**2
    if phi < phir:
        return G1
    if phi < 33.1:
        return 29 - 25 * np.log10(phi)
    if phi < 80:
        return -9
    return -5

def calculate_received_interference_watt(u_tgt_pos, u_inf_serving_pos, sat_tgt_pos, sat_inf_pos):
    d = np.linalg.norm(u_tgt_pos - sat_inf_pos)
    vec_tx_main = u_inf_serving_pos - sat_inf_pos
    vec_tx_to_tgt = u_tgt_pos - sat_inf_pos
    psi_tx = angle(vec_tx_to_tgt, vec_tx_main)
    
    vec_rx_main = sat_tgt_pos - u_tgt_pos
    vec_rx_from_inf = sat_inf_pos - u_tgt_pos
    phi_rx = angle(vec_rx_main, vec_rx_from_inf)
    
    gtx = itu_s1528_tx(psi_tx, D_LAMBDA_TX)
    grx = itu_s1428_rx(phi_rx, D_LAMBDA_RX)
    
    eirp = P_TX_DENSITY + gtx + 10 * np.log10(B)
    fspl = 20 * np.log10(d) + 20 * np.log10(FREQ) - 20 * np.log10(C / (4 * np.pi))
    I_dbw = eirp + grx - fspl
    return 10**(I_dbw / 10.0), psi_tx, phi_rx, gtx, grx, fspl

def get_satellite_positions(tle_path, time_offset_seconds):
    sats = {}
    if not tle_path.exists():
        return None
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
    
    for sid, rec in sats.items():
        e, r, v = rec.sgp4(jd, fr)
        if e == 0:
            obstime = Time(t, scale="utc")
            teme = TEME(x=r[0]*u.km, y=r[1]*u.km, z=r[2]*u.km, obstime=obstime)
            itrs = teme.transform_to(ITRS(obstime=obstime))
            positions[sid] = np.array([itrs.x.to(u.m).value, itrs.y.to(u.m).value, itrs.z.to(u.m).value])
    return positions


def plot_interfering_satellites_2d(c21_channels, active_transmitters, sat_positions, tgt_ecef, cell21_latlon):
    """Generates a 2D spatial scatter plot with hover text containing satellite numbers and active channel beam counts."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    c21_lat, c21_lon = cell21_latlon

    # Pre-calculate active channel beam counts per satellite across all active transmitters
    sat_channel_beam_counts = defaultdict(lambda: defaultdict(int))
    for slot, tx_list in active_transmitters.items():
        for tx in tx_list:
            sat_channel_beam_counts[slot][tx["sat_id"]] += 1

    arrow_len = 0.5

    for slot, target_data in c21_channels.items():
        target_reuse = target_data["reuse"]
        target_sat_id = target_data["sat"]

        if target_sat_id not in sat_positions:
            continue

        sat_tgt_vec = sat_positions[target_sat_id]
        tgt_sat_lat, tgt_sub_lon = ecef_to_latlon(sat_tgt_vec)

        # Count beams for serving satellite on this specific channel slot
        serving_beam_count = sat_channel_beam_counts[slot].get(target_sat_id, 1)

        # Plot serving satellite with metadata attributes attached
        sc_serv = ax.scatter(
            [tgt_sub_lon], [tgt_sat_lat], 
            color='dodgerblue', marker='*', s=120, zorder=5
        )
        # Store custom hover metadata on the path collection object
        sc_serv.sat_info = f"Serving Sat ID: {target_sat_id}\nChannel: {slot}\nActive Beams: {serving_beam_count}"

        ax.text(tgt_sub_lon + 0.2, tgt_sat_lat, f"CH {slot} (Serv: {target_sat_id})", fontsize=9, color='blue')

        # Boresight direction vector pointing from satellite sub-point toward target cell
        dlat = c21_lat - tgt_sat_lat
        dlon = c21_lon - tgt_sub_lon
        norm_factor = np.sqrt(dlat**2 + dlon**2)
        if norm_factor > 0:
            ax.arrow(tgt_sub_lon, tgt_sat_lat, (dlon / norm_factor) * arrow_len, (dlat / norm_factor) * arrow_len,
                     head_width=0.1, head_length=0.15, fc='dodgerblue', ec='dodgerblue', zorder=6)

        co_channel_txs = active_transmitters.get(slot, [])
        evaluated_beams = set()

        for tx in co_channel_txs:
            if tx["cell_id"] == target_data and tx["reuse"] == target_reuse:
                continue

            beam_key = (tx["sat_id"], tx["reuse"])
            if beam_key in evaluated_beams:
                continue
            evaluated_beams.add(beam_key)

            inf_sat_id = tx["sat_id"]
            if inf_sat_id not in sat_positions:
                continue

            inf_sat_vec = sat_positions[inf_sat_id]
            inf_lat, inf_lon = ecef_to_latlon(inf_sat_vec)

            p_watts, _, _, _, _, _ = calculate_received_interference_watt(
                u_tgt_pos=tgt_ecef,
                u_inf_serving_pos=tx["ecef"],
                sat_tgt_pos=sat_tgt_vec,
                sat_inf_pos=inf_sat_vec
            )
            
            inr_single = (10 * np.log10(p_watts) - N0_DB) if p_watts > 0 else -100.0

            color = 'crimson' if inr_single > INR_THRESHOLD else 'forestgreen'
            marker = 'X' if inr_single > INR_THRESHOLD else 'o'

            # Interfering beam count for this satellite on this channel slot
            inf_beam_count = sat_channel_beam_counts[slot].get(inf_sat_id, 1)

            sc_inf = ax.scatter(
                [inf_lon], [inf_lat], 
                color=color, marker=marker, s=50, alpha=0.8, zorder=5
            )
            sc_inf.sat_info = (
                f"Interfering Sat ID: {inf_sat_id}\n"
                f"Channel: {slot}\n"
                f"Active Beams: {inf_beam_count}\n"
                f"Single INR: {inr_single:.2f} dB"
            )

            # Boresight direction vector pointing from interferer satellite toward its served user/cell ECEF
            inf_cell_lat, inf_cell_lon = ecef_to_latlon(tx["ecef"])
            idlat = inf_cell_lat - inf_lat
            idlon = inf_cell_lon - inf_lon
            inorm_factor = np.sqrt(idlat**2 + idlon**2)
            if inorm_factor > 0:
                ax.arrow(inf_lon, inf_lat, (idlon / inorm_factor) * arrow_len, (idlat / inorm_factor) * arrow_len,
                         head_width=0.08, head_length=0.12, fc=color, ec=color, zorder=6)

    ax.scatter([c21_lon], [c21_lat], color='black', marker='P', s=100, zorder=6)
    ax.text(c21_lon + 0.2, c21_lat, "Cell 21 Target", fontsize=10, fontweight='bold', color='black')

    ax.set_xlabel("Sub-Satellite Longitude (Degrees)", fontsize=10)
    ax.set_ylabel("Sub-Satellite Latitude (Degrees)", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Enable interactive hover tooltips via mplcursors
    cursor = mplcursors.cursor(ax.collections, hover=True)
    @cursor.connect("add")
    def on_add(selection):
        artist = selection.artist
        if hasattr(artist, "sat_info"):
            selection.annotation.set_text(artist.sat_info)
        else:
            selection.annotation.set_text("Cell 21 Target" if selection.target[0] == c21_lon else "Satellite Point")
        selection.annotation.get_bbox_patch().set(fc="white", alpha=0.9, edgecolor="gray")

    plt.tight_layout()
    
    print("\n[OUTPUT] Opening interactive plot window with hover tooltips enabled...")
    plt.show()


def run_cell21_audit(pkl_path: Path, time_seconds: int = 0):
    print("=" * 80)
    print(f"      AUDITING CELL 21 ANOMALY ON PKL FILE")
    print(f"      Path: {pkl_path}")
    print("=" * 80)

    if not pkl_path.exists():
        print(f"[ERROR] Specified PKL file does not exist: {pkl_path}")
        return

    sat_positions = get_satellite_positions(TLE_FILE, time_seconds)
    if not sat_positions:
        print("[ERROR] Failed to calculate satellite positions from TLE.")
        return

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    cell_map = defaultdict(dict)
    cells = set()
    for k, v in raw.items():
        cell, slot_str = k.rsplit("_", 1)
        slot = int(slot_str)
        reuse, sat, satbeam = v.split("_")
        reuse_idx = int(reuse)
        cell_map[cell][slot] = {"reuse": reuse_idx, "sat": int(sat)}
        cells.add(cell)

    sorted_cells = sorted(list(cells))
    cell_labels = {cell: f"Cell {idx + 1}" for idx, cell in enumerate(sorted_cells)}
    
    cell21_h3 = None
    for h3_hash, label in cell_labels.items():
        if label == "Cell 21":
            cell21_h3 = h3_hash
            break

    if not cell21_h3:
        print(f"[CRITICAL] 'Cell 21' label not found! Total cells in PKL: {len(sorted_cells)}")
        return

    print(f"\n[1] Key Mapping:")
    print(f"  Target Label : Cell 21")
    print(f"  H3 Hash Key  : {cell21_h3}")

    cell_metadata = {}
    for cell in sorted_cells:
        lat, lon = h3.cell_to_latlng(cell)
        cell_metadata[cell] = {
            "latlng": [lat, lon],
            "ecef": ecef(lat, lon)
        }

    c21_channels = cell_map[cell21_h3]
    print(f"\n[2] Cell 21 Channel Allocations:")
    if not c21_channels:
        print("  [EXPLANATION] Cell 21 HAS NO ACTIVE CHANNELS ASSIGNED in this snapshot!")
        return

    for slot, data in c21_channels.items():
        print(f"  --> Hardware Channel {slot}: Assigned Sat ID = {data['sat']}, Reuse = {data['reuse']}")

    active_transmitters = defaultdict(list)
    for cell_id in sorted_cells:
        for slot, target_data in cell_map[cell_id].items():
            sat_id = target_data["sat"]
            r_idx = target_data["reuse"]
            if sat_id in sat_positions:
                active_transmitters[slot].append({
                    "cell_id": cell_id,
                    "cell_label": cell_labels[cell_id],
                    "sat_id": sat_id,
                    "reuse": r_idx,
                    "ecef": cell_metadata[cell_id]["ecef"]
                })

    tgt_ecef = cell_metadata[cell21_h3]["ecef"]
    print(f"\n[3] Physical Co-Channel RF Interference Evaluation for Cell 21:")

    for slot, target_data in c21_channels.items():
        target_reuse = target_data["reuse"]
        target_sat_id = target_data["sat"]

        print(f"\n Hardware Channel {slot}")
        if target_sat_id not in sat_positions:
            print(f"  [CRITICAL BUG] Serving Sat ID {target_sat_id} missing from SGP4 positions!")
            continue

        sat_tgt_vec = sat_positions[target_sat_id]
        dist_to_serving = np.linalg.norm(sat_tgt_vec - tgt_ecef) / 1000.0
        print(f"  Serving Satellite  : {target_sat_id}")
        print(f"  Slant Range (km)   : {dist_to_serving:.2f} km")

        co_channel_txs = active_transmitters.get(slot, [])
        print(f"  Total Co-Channel Transmitters on Slot {slot}: {len(co_channel_txs)}")

        evaluated_beams = set()
        total_i_watts = 0.0
        interferer_count = 0

        for tx in co_channel_txs:
            if tx["cell_id"] == cell21_h3 and tx["reuse"] == target_reuse:
                continue

            beam_key = (tx["sat_id"], tx["reuse"])
            if beam_key in evaluated_beams:
                continue
            evaluated_beams.add(beam_key)

            p_watts, psi_tx, phi_rx, gtx, grx, fspl = calculate_received_interference_watt(
                u_tgt_pos=tgt_ecef,
                u_inf_serving_pos=tx["ecef"],
                sat_tgt_pos=sat_tgt_vec,
                sat_inf_pos=sat_positions[tx["sat_id"]]
            )
            total_i_watts += p_watts
            interferer_count += 1

            print(f"\n    * Interfering Beam #{interferer_count} (Sat {tx['sat_id']} -> {tx['cell_label']}):")
            print(f"      - Tx Off-Axis Angle (psi) : {psi_tx:.4f}°  --> Tx Gain : {gtx:.2f} dBi")
            print(f"      - Rx Off-Axis Angle (phi) : {phi_rx:.4f}°  --> Rx Gain : {grx:.2f} dBi")
            print(f"      - FSPL                    : {fspl:.2f} dB")
            print(f"      - Received Power          : {10*np.log10(p_watts if p_watts > 0 else 1e-20):.2f} dBW ({p_watts:.2e} Watts)")

        if total_i_watts > 0:
            inr_db = (10 * np.log10(total_i_watts)) - N0_DB
        else:
            inr_db = -100.0

        print(f"\n _______________________________________________________")
        print(f"  Aggregated Interference Watts : {total_i_watts:.2e} W")
        print(f"  Computed INR for Channel {slot}    : {inr_db:.2f} dB")
        
        if inr_db == -100.0:
            print(f"  --> [ANOMALY DETECTED] Total Interference is EXACTLY ZERO Watts!")
        elif inr_db <= INR_THRESHOLD:
            print(f"  --> Status: SAFE (INR <= {INR_THRESHOLD} dB) [Outlier behavior]")
        else:
            print(f"  --> Status: CRITICAL INTERFERENCE (INR > {INR_THRESHOLD} dB) [Bunched high exceedance]")

    print(f"\n[4] Plotting 2D Satellite Distribution...")
    plot_interfering_satellites_2d(c21_channels, active_transmitters, sat_positions, tgt_ecef, cell_metadata[cell21_h3]["latlng"])
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    country = "haiti"
    pop = 1000
    ut_dist = "population"
    routing = "greedy-coordinated"
    time_sec = 0
    time_nsec = time_sec * 1_000_000_000
    
    folder = f"starlink_5shells_ground_stations_starlink_cells_{country}_0_{pop}_{ut_dist}_{routing}"
    pkl_file = BASE_DATA_DIR / folder / f"beam_assignments_{time_nsec}.pkl"

    run_cell21_audit(pkl_file, time_seconds=time_sec)
