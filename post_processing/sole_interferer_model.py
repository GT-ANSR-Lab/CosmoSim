#!/usr/bin/env python3

"""
This script provides an interactive Streamlit application for analyzing co-channel interference and neighbor relationships within simulation data sets. It loads time snapshot mapping files and propagates satellite ephemeris to calculate isolated link single interferer interference-to-noise ratio (INR) values across neighboring hexagonal ground cells. The user interface includes sidebar scenario selectors, an interactive map visualization displaying cell polygons and color coded neighbor conflict lines, simulation metrics, global statistical summaries, and expandable detailed conflict logs.
"""

import os
import pickle
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
import sys
import numpy as np
import pandas as pd
import h3
import folium
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(layout="wide", page_title="Co-Channel Neighbor Analyzer with INR")
st.title("Co-Channel Neighbor Analyzer with INR")

CURRENT_DIR = Path(__file__).resolve().parent
COSMOSIM_ROOT = CURRENT_DIR.parent
if str(COSMOSIM_ROOT) not in sys.path:
    sys.path.insert(0, str(COSMOSIM_ROOT))

from spectrum_management.interference_rf import (
    C,
    F_START,
    F_END,
    BW as B,
    T_SYS,
    N0_DB as N0,
    D_UT,
    D_ST,
    EIRP_DENSITY_MAX,
    EPOCH_BASE,
    get_channel_frequency,
    norm,
    angle,
    ecef,
    itu_s1528_tx,
    itu_s1428_rx,
    get_satellite_positions,
)

BASE_DATA_DIR = COSMOSIM_ROOT / "data"
TLE_FILE = COSMOSIM_ROOT / "constellation_configurations" / "configs" / "starlink_5shells" / "tles.txt"

CONST = "starlink_5shells"
GS = "ground_stations"

INR_THRESHOLD = -12.0 #-6.0 


def calculate_single_inr(user_target, user_interferer, sat_target_pos, sat_interferer_pos, channel_idx=0):
    """Calculates isolated INR at user_target caused by downlink beam serving user_interferer on channel_idx."""
    freq = get_channel_frequency(channel_idx)
    lam = C / freq

    d_lambda_rx = D_UT / lam
    d_lambda_tx = D_ST / lam
    g_sat_max = 20 * np.log10(d_lambda_tx) + 7.7
    p_tx_density = EIRP_DENSITY_MAX - g_sat_max

    d = np.linalg.norm(user_target - sat_interferer_pos)
    
    # Off-axis angle at transmitting satellite serving the neighboring cell
    phi_tx = angle(user_target - sat_interferer_pos, user_interferer - sat_interferer_pos)
    # Off-axis angle at target user pointing towards target sat vs. interfering sat
    phi_rx = angle(sat_target_pos - user_target, sat_interferer_pos - user_target)
    
    gtx = itu_s1528_tx(phi_tx, d_lambda_tx)
    grx = itu_s1428_rx(phi_rx, d_lambda_rx)
    
    eirp = p_tx_density + gtx + 10 * np.log10(B)
    fspl = 20 * np.log10(d) + 20 * np.log10(freq) - 20 * np.log10(C / (4 * np.pi))
    I = eirp + grx - fspl
    return I - N0


st.sidebar.header("Data Query Parameters")
country_input = st.sidebar.selectbox("Country Target", options=["haiti"], format_func=lambda x: x.title())
pop_input = st.sidebar.selectbox("Number of Terminals", options=[1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000])
ut_dist_input = st.sidebar.selectbox("User Terminal Distribution Policy", options=["population", "gcb_no_cap", "gcb_1000", "gcb_10000", "gcb_100000"], format_func=lambda x: "Population" if x == "population" else x.replace("_", " ").upper())
ku_capacity_input = st.sidebar.selectbox("Ku-Band Capacity Limit Factor", options=[0.956, 1.28, 2.5])

time_input = st.sidebar.selectbox("Time Snapshot (Seconds)", options=[0, 1000000000, 2000000000, 3000000000, 4000000000, 5000000000, 6000000000, 7000000000, 8000000000, 9000000000, 10000000000, 11000000000, 12000000000, 13000000000, 14000000000])

routing_input = st.sidebar.selectbox(
    "Routing Policy", 
    options=["greedy-coordinated", "greedy-uncoordinated", "greedy-coordinated-inr-aware", "greedy-uncoordinated-inr-aware"], 
    format_func=lambda x: x.replace("-", " ").title()
)

if ut_dist_input in ["population", "uniform"]:
    folder_name = f"{CONST}_ground_stations_starlink_cells_{country_input}_0_{pop_input}_{ut_dist_input}_{routing_input}"
else:
    cap_part = "gcb_no_cap" if ut_dist_input == "gcb_no_cap" else f"gcb_cap_{ut_dist_input.split('_')[1]}"
    folder_name = f"{CONST}_ground_stations_starlink_cells_{country_input}_0_{pop_input}_{cap_part}_{ku_capacity_input}_{routing_input}"
    
TARGET_PKL_PATH = BASE_DATA_DIR / folder_name / f"beam_assignments_{time_input}.pkl"
st.sidebar.info(f"**Target Data Path:**\n`{TARGET_PKL_PATH}`")

if not TARGET_PKL_PATH.exists():
    st.error(f"Snapshot folder or file not found. Ensure directory exists at:\n`{TARGET_PKL_PATH}`")
else:
    if st.sidebar.button("Run Simulation", type="primary"):
        time_seconds = time_input / 1_000_000_000
        sat_positions = get_satellite_positions(TLE_FILE, time_seconds)
        if not sat_positions:
            st.error(f"TLE source file could not be parsed at: `{TLE_FILE}`")
            st.stop()

        with st.spinner("Processing spatial maps and computing localized INR link budgets..."):
            with TARGET_PKL_PATH.open("rb") as f:
                raw = pickle.load(f)

            cell_map = defaultdict(dict)
            cells = set()
            active_slots = set()  

            for k, v in raw.items():
                cell, slot_str = k.rsplit("_", 1)
                slot_id = int(slot_str)
                reuse, sat, satbeam = v.split("_")
                cell_map[cell][slot_id] = {"reuse": int(reuse), "sat": int(sat)}
                cells.add(cell)
                active_slots.add(slot_id)

            sorted_slots = sorted(list(active_slots))
            sorted_unique_cells = sorted(list(cells))
            cell_labels = {cell: f"Cell {idx + 1}" for idx, cell in enumerate(sorted_unique_cells)}
            total_cells = len(cell_labels)

            # Discover adjacent neighboring cells
            adjacent_pairs = set()
            for c in sorted_unique_cells:
                try:
                    nbrs = h3.grid_disk(c, 1)
                except:
                    continue
                for n in nbrs:
                    if n != c and n in cells:
                        adjacent_pairs.add(tuple(sorted([c, n])))

            sample_lat, sample_lon = h3.cell_to_latlng(sorted_unique_cells[0])
            m = folium.Map(location=[sample_lat, sample_lon], zoom_start=11, tiles="CartoDB positron")

            for cell in sorted_unique_cells:
                lat, lon = h3.cell_to_latlng(cell)
                folium.Polygon(
                    locations=h3.cell_to_boundary(cell), color="blue", weight=2,
                    fill=True, fill_color="cyan", fill_opacity=0.12,
                    tooltip=f"<b>{cell_labels[cell]}</b><br>Hex: {cell}"
                ).add_to(m)
                
                folium.map.Marker(
                    [lat, lon],
                    icon=folium.DivIcon(
                        icon_size=(150, 36), icon_anchor=(75, 18),
                        html=f'<div style="font-size: 9pt; font-weight: bold; color: #1c2833; text-align: center;">{cell_labels[cell]}</div>'
                    )
                ).add_to(m)

            co_channel_records = []
            total_matches_found = 0
            all_calculated_inrs = []
            passed_edges_count = 0  

            for cell_a, cell_b in sorted(adjacent_pairs):
                lat_a, lon_a = h3.cell_to_latlng(cell_a)
                lat_b, lon_b = h3.cell_to_latlng(cell_b)
                
                user_a_ecef = ecef(lat_a, lon_a)
                user_b_ecef = ecef(lat_b, lon_b)
                
                pair_results = []
                edge_exceeds_threshold = False  

                for slot in sorted_slots:
                    # Check if both adjacent neighbors have active transmission profiles on this SAME slot
                    if slot not in cell_map[cell_a] or slot not in cell_map[cell_b]:
                        continue
                        
                    a = cell_map[cell_a][slot]
                    b = cell_map[cell_b][slot]

                    sat_a_id, sat_b_id = a["sat"], b["sat"]
                    reuse_a, reuse_b = a["reuse"], b["reuse"]
                    
                    inr_a, inr_b = np.nan, np.nan
                    if sat_a_id in sat_positions and sat_b_id in sat_positions:
                        pos_sat_a = sat_positions[sat_a_id]
                        pos_sat_b = sat_positions[sat_b_id]
                        
                        inr_a = calculate_single_inr(user_a_ecef, user_b_ecef, pos_sat_a, pos_sat_b, channel_idx=slot)
                        inr_b = calculate_single_inr(user_b_ecef, user_a_ecef, pos_sat_b, pos_sat_a, channel_idx=slot)
                        
                        all_calculated_inrs.extend([inr_a, inr_b])
                        
                        if inr_a > INR_THRESHOLD or inr_b > INR_THRESHOLD:
                            edge_exceeds_threshold = True

                    pair_results.append({
                        "slot": slot, 
                        "freq_ghz": get_channel_frequency(slot) / 1e9,
                        "reuse_a": reuse_a,
                        "reuse_b": reuse_b,
                        "sat_a": sat_a_id, 
                        "sat_b": sat_b_id,
                        "inr_a": inr_a, 
                        "inr_b": inr_b
                    })
                    total_matches_found += 1

                folium.PolyLine(locations=[[lat_a, lon_a], [lat_b, lon_b]], color="#a6acaf", weight=1.5, opacity=0.5).add_to(m)
                
                if pair_results:
                    co_channel_records.append({
                        "cell_a": cell_a, 
                        "cell_b": cell_b, 
                        "matches": pair_results, 
                        "exceeds_threshold": edge_exceeds_threshold
                    })
                    
                    conflict_details = "<br>".join([
                        f"Slot {r['slot']} ({r['freq_ghz']:.3f} GHz): Sat {r['sat_a']} (R{r['reuse_a']}) ↔ Sat {r['sat_b']} (R{r['reuse_b']}) | "
                        f"INR: A={r['inr_a']:.1f}dB, B={r['inr_b']:.1f}dB" if not np.isnan(r['inr_a']) else f"Slot {r['slot']}: No Ephemeris Data"
                        for r in pair_results
                    ])
                    
                    if edge_exceeds_threshold:
                        line_color = "red"        
                        prefix_title = "Critical Conflict (> -12.0 dB INR)"
                    else:
                        line_color = "blue"       
                        prefix_title = "Co-Channel Active Connection (≤ -12.0 dB INR)"
                        passed_edges_count += 1

                    popup_txt = f"<b>{prefix_title}</b><br>{cell_labels[cell_a]} ↔ {cell_labels[cell_b]}<br>{conflict_details}"
                    
                    folium.PolyLine(
                        locations=[[lat_a, lon_a], [lat_b, lon_b]],
                        color=line_color, weight=4, opacity=0.9,
                        popup=folium.Popup(popup_txt, max_width=350)
                    ).add_to(m)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Co Channel plot")
                st_folium(m, width=850, height=550, returned_objects=[])
                st.caption("**Map Legend:** **Gray** links indicate clean normal neighboring edges. **Blue** lines mark active co-channel overlaps bounded at or below $-12\\text{ dB}$. **Red** lines highlight co-channel links exceeding the $-12\\text{ dB}$ protection threshold.")

            with col2:
                st.subheader("Simulation Analytics Summary")
                st.metric("Total Unique Cells Indexed", f"{total_cells}")
                st.metric("Edges with Co-Channel Conflicts", f"{len(co_channel_records)}")
                st.metric("Passed Co-Channel Edges", f"{passed_edges_count}")
                st.metric("Total Active Co-Channel Events", f"{total_matches_found}")
                
                st.markdown("---")
                st.markdown("#### Global Spatial INR Breakdown")
                if all_calculated_inrs:
                    st.metric("Minimum Observed INR", f"{np.min(all_calculated_inrs):.2f} dB")
                    st.metric("Median Observed INR", f"{np.median(all_calculated_inrs):.2f} dB")
                    st.metric("Maximum Observed INR", f"{np.max(all_calculated_inrs):.2f} dB")
                else:
                    st.caption("No valid spatial link budgets could be resolved for INR evaluation.")

            if co_channel_records:
                st.subheader("Co-Channel Conflicts Detail log")
                for record in co_channel_records:
                    c_a, c_b = record["cell_a"], record["cell_b"]
                    header_tag = "🔴 [EXCEEDS -12.0 dB]" if record["exceeds_threshold"] else "🔵 [UNDER -12.0 dB]"
                    
                    with st.expander(f"{header_tag} Conflict Edge: {cell_labels[c_a]} to {cell_labels[c_b]} ({len(record['matches'])} co channel conflicts)"):
                        for match in record["matches"]:
                            inr_string = (
                                f"<br>&nbsp;&nbsp;&nbsp;&nbsp; **INR Cell A (Victim):** `{match['inr_a']:.2f} dB`"
                                f" | **INR Cell B (Victim):** `{match['inr_b']:.2f} dB`"
                            ) if not np.isnan(match['inr_a']) else "<br>&nbsp;&nbsp;&nbsp;&nbsp;*INR Calculation missing TLE data*"
                            
                            st.write(
                                f"**Slot {match['slot']} ({match['freq_ghz']:.3f} GHz)** | "
                                f"**{cell_labels[c_a]}**: Sat `{match['sat_a']}` (Reuse `{match['reuse_a']}`) ↔ "
                                f"**{cell_labels[c_b]}**: Sat `{match['sat_b']}` (Reuse `{match['reuse_b']}`)"
                                f"{inr_string}",
                                unsafe_allow_html=True
                            )
