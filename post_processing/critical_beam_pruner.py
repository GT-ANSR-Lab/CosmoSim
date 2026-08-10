#!/usr/bin/env python3
import os
import pickle
import re
import random
from pathlib import Path
from collections import defaultdict
import sys

import numpy as np
import pandas as pd
import h3
import folium
import streamlit as st
from streamlit_folium import st_folium

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
    N0_WATT,
    N0_DB,
    D_UT,
    D_ST,
    EIRP_DENSITY_MAX,
    EPOCH_BASE,
    get_channel_frequency,
    norm,
    angle,
    ecef,
    calculate_received_interference_watt,
    get_satellite_positions,
)

st.set_page_config(layout="wide", page_title="INR Adaptive Optimizer")
st.title("INR-Aware Adaptive Beam Optimizer")
st.markdown(
    "This system iterates through the baseline scenario's beam assignments in order. "
    "If an assignment causes any cell's INR for that channel to exceed the $-12\\text{ dB}$ threshold, "
    "it is immediately dropped before moving to the next beam."
)

BASE_DATA_DIR = COSMOSIM_ROOT / "data"
TLE_FILE = COSMOSIM_ROOT / "constellation_configurations" / "configs" / "starlink_5shells" / "tles.txt"

CONST = "starlink_5shells"
GS = "ground_stations"

INR_THRESHOLD = -12.0
MIN_ELEVATION_DEG = 25.0
R_EARTH = 6378137.0


def compute_network_interference(sorted_unique_cells, cell_map, cell_metadata, sat_positions):
    cell_channel_metrics = defaultdict(dict)
    cell_worst_channel_inr = {}
    global_safe_cells_count = 0
    
    active_transmitters = defaultdict(list)
    for cell_id in sorted_unique_cells:
        for slot, target_data in cell_map[cell_id].items():
            sat_id = target_data["sat"]
            if sat_id in sat_positions:
                active_transmitters[slot].append({
                    "cell_id": cell_id,
                    "sat_id": sat_id,
                    "reuse": target_data["reuse"],
                    "ecef": cell_metadata[cell_id]["ecef"]
                })

    for target_cell in sorted_unique_cells:
        meta_target = cell_metadata[target_cell]
        tgt_ecef = meta_target["ecef"]
        
        for slot, target_data in cell_map[target_cell].items():
            target_sat_id = target_data["sat"]
            if target_sat_id not in sat_positions: continue
            
            sat_tgt_vec = sat_positions[target_sat_id]
            channel_interference_watts = 0.0
            evaluated_beams = set()
            
            for tx in active_transmitters.get(slot, []):
                if tx["cell_id"] == target_cell and tx["reuse"] == target_data["reuse"]:
                    continue
                
                beam_key = (tx["sat_id"], tx["reuse"])
                if beam_key in evaluated_beams: continue
                evaluated_beams.add(beam_key)
                
                channel_interference_watts += calculate_received_interference_watt(
                    tgt_ecef, tx["ecef"], sat_tgt_vec, sat_positions[tx["sat_id"]], channel_idx=slot
                )

            channel_inr_db = (10 * np.log10(channel_interference_watts) - N0_DB) if channel_interference_watts > 0 else -100.0
            cell_channel_metrics[target_cell][slot] = channel_inr_db

        worst_inr = max(cell_channel_metrics[target_cell].values()) if cell_channel_metrics[target_cell] else -100.0
        cell_worst_channel_inr[target_cell] = worst_inr
        if worst_inr <= INR_THRESHOLD:
            global_safe_cells_count += 1
            
    return dict(cell_channel_metrics), dict(cell_worst_channel_inr), global_safe_cells_count


@st.cache_data(show_spinner="Iterating through beam assignments and pruning high INR links...")
def get_optimized_and_pruned_mapping(raw_original, sat_positions, cell_metadata):
    dropped_beams = []
    accepted_assignments = []

    def check_max_inr_for_channel(channel, current_assignments):
        # Extract all active transmitters using this specific channel
        channel_txs = [a for a in current_assignments if a["channel"] == channel]
        if len(channel_txs) <= 1:
            return -100.0

        max_inr = -100.0

        # Evaluate received INR at every cell using this channel
        for target in channel_txs:
            tgt_cell = target["cell"]
            tgt_sat_id = target["sat"]
            
            if tgt_sat_id not in sat_positions:
                continue

            tgt_ecef = cell_metadata[tgt_cell]["ecef"]
            sat_tgt_vec = sat_positions[tgt_sat_id]

            channel_interference_watts = 0.0
            evaluated_beams = set()

            for tx in channel_txs:
                if tx["cell"] == tgt_cell and tx["reuse"] == target["reuse"]:
                    continue

                beam_key = (tx["sat"], tx["reuse"])
                if beam_key in evaluated_beams:
                    continue
                evaluated_beams.add(beam_key)

                if tx["sat"] in sat_positions:
                    channel_interference_watts += calculate_received_interference_watt(
                        tgt_ecef, cell_metadata[tx["cell"]]["ecef"], sat_tgt_vec, sat_positions[tx["sat"]], channel_idx=channel
                    )

            inr_db = (10 * np.log10(channel_interference_watts) - N0_DB) if channel_interference_watts > 0 else -100.0
            if inr_db > max_inr:
                max_inr = inr_db

        return max_inr

    # Iterate through each beam assignment in the order defined by the scenario
    for raw_key, raw_val in raw_original.items():
        cell, slot_str = raw_key.rsplit("_", 1)
        reuse, sat, _ = raw_val.split("_")
        sat_id = int(sat)
        channel = int(slot_str)
        reuse_idx = int(reuse)

        if sat_id not in sat_positions:
            dropped_beams.append({
                "cell": cell, "channel": channel, "sat": sat_id, "reason": "Satellite Pos Unavailable"
            })
            continue

        candidate_asn = {
            "cell": cell,
            "channel": channel,
            "reuse": reuse_idx,
            "sat": sat_id,
            "raw_key": raw_key,
            "raw_val": raw_val
        }

        # Temporarily add the candidate assignment
        test_state = accepted_assignments + [candidate_asn]

        # Evaluate if this assignment causes any cell's INR on this channel to exceed -12.0 dB
        peak_inr = check_max_inr_for_channel(channel, test_state)

        if peak_inr > INR_THRESHOLD:
            # Exceeds threshold: Drop the assignment and check the next beam
            dropped_beams.append({
                "cell": cell,
                "channel": channel,
                "sat": sat_id,
                "reason": f"INR Exceeded Threshold ({peak_inr:.2f} dB > -12.0 dB)"
            })
        else:
            # Within limit: Accept assignment and keep it in the active set
            accepted_assignments.append(candidate_asn)

    # Reconstruct optimized mapping dictionary
    optimized_mapping = {}
    for a in accepted_assignments:
        optimized_mapping[f"{a['cell']}_{a['channel']}"] = f"{a['reuse']}_{a['sat']}_{a['channel']}"

    return optimized_mapping, dropped_beams


st.sidebar.header("Parameters")
country_input = st.sidebar.selectbox("Country", options=["haiti"], format_func=lambda x: x.title())
pop_input = st.sidebar.selectbox("Number of Terminals", options=[1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000])

ut_dist_input = st.sidebar.selectbox(
    "User Terminal Distribution Policy", 
    options=["population", "uniform", "gcb_no_cap", "gcb_1000", "gcb_10000", "gcb_100000"], 
    format_func=lambda x: x.title() if x in ["population", "uniform"] else x.replace("_", " ").upper()
)

ku_capacity_input = st.sidebar.selectbox("Ku-Band Capacity Limit Factor", options=[0.956, 1.28, 2.5])
time_input_seconds = st.sidebar.selectbox("Time Snapshot (seconds)", options=list(range(0, 15)))
time_input_nanoseconds = time_input_seconds * 1_000_000_000
routing_input = st.sidebar.selectbox("Routing Protocol Strategy", options=["greedy-coordinated", "greedy-uncoordinated", "greedy-coordinated-inr-aware", "greedy-uncoordinated-inr-aware"])

if ut_dist_input in ["population", "uniform"]:
    folder_name = f"{CONST}_ground_stations_starlink_cells_{country_input}_0_{pop_input}_{ut_dist_input}_{routing_input}"
else:
    cap_part = "gcb_no_cap" if ut_dist_input == "gcb_no_cap" else f"gcb_cap_{ut_dist_input.split('_')[1]}"
    folder_name = f"{CONST}_ground_stations_starlink_cells_{country_input}_0_{pop_input}_{cap_part}_{ku_capacity_input}_{routing_input}"
    
TARGET_PKL_PATH = BASE_DATA_DIR / folder_name / f"beam_assignments_{time_input_nanoseconds}.pkl"

if not TARGET_PKL_PATH.exists():
    st.error(f"Snapshot data file not found at:\n`{TARGET_PKL_PATH}`")
else:
    sat_positions = get_satellite_positions(TLE_FILE, time_input_seconds)
    if not sat_positions:
        st.error("TLE source file structure error or missing data nodes.")
        st.stop()

    with TARGET_PKL_PATH.open("rb") as f:
        raw_original = pickle.load(f)

    cells = set(k.rsplit("_", 1)[0] for k in raw_original.keys())
    sorted_unique_cells = sorted(list(cells))

    cell_metadata = {}
    for cell in sorted_unique_cells:
        lat, lon = h3.cell_to_latlng(cell)
        raw_boundary = h3.cell_to_boundary(cell)
        cell_metadata[cell] = {
            "latlng": [lat, lon],
            "ecef": ecef(lat, lon),
            "boundary": [[coords[0], coords[1]] for coords in raw_boundary]
        }

    run_mode = st.sidebar.radio("Analysis Spectrum Selection", ["Show Original (Baseline)", "Show Optimized Mapping"])

    dropped_beams = []
    if run_mode == "Show Optimized Mapping":
        raw, dropped_beams = get_optimized_and_pruned_mapping(raw_original, sat_positions, cell_metadata)
    else:
        raw = raw_original

    orig_cell_map = defaultdict(dict)
    for k, v in raw_original.items():
        cell, slot_str = k.rsplit("_", 1)
        orig_cell_map[cell][int(slot_str)] = True

    cell_map = defaultdict(dict)
    matrix_occupancy_map = defaultdict(lambda: defaultdict(list))

    for k, v in raw.items():
        cell, slot_str = k.rsplit("_", 1)
        slot = int(slot_str)
        reuse, sat, satbeam = v.split("_")
        reuse_idx = int(reuse)
        cell_map[cell][slot] = {"reuse": reuse_idx, "sat": int(sat)}

    cell_labels = {cell: f"Cell {idx + 1}" for idx, cell in enumerate(sorted_unique_cells)}
    total_cells = len(cell_labels)

    for cell in sorted_unique_cells:
        for slot, data in cell_map[cell].items():
            r_idx = data["reuse"]
            matrix_occupancy_map[slot][r_idx].append(cell)

    cell_channel_metrics, cell_worst_channel_inr, global_safe_cells_count = compute_network_interference(
        sorted_unique_cells, cell_map, cell_metadata, sat_positions
    )

    st.markdown("### Beam Filter")
    matrix_options = ["Show All Profiles"]
    for ch in range(8):
        freq_ghz = get_channel_frequency(ch) / 1e9
        matrix_options.append(f"Channel {ch} ({freq_ghz:.3f} GHz)")
            
    selected_profile = st.selectbox(
        "Select a physical hardware slot configuration to isolate on the plot:",
        options=matrix_options
    )

    highlight_active = selected_profile != "Show All Profiles"
    highlighted_cells = set()
    scenario_passed_count = 0
    scenario_total_count = 0
    
    total_original_beams = len(raw_original)
    total_active_beams = len(raw)

    if highlight_active:
        sel_channel = int(selected_profile.split(" ")[1])
        for ru in range(4):
            highlighted_cells.update(matrix_occupancy_map[sel_channel][ru])
        scenario_total_count = len(highlighted_cells)
        
        scenario_original_count = sum(
            1 for cell in sorted_unique_cells if sel_channel in orig_cell_map[cell]
        )
        
        for h_cell in highlighted_cells:
            if cell_channel_metrics[h_cell].get(sel_channel, -100.0) <= INR_THRESHOLD:
                scenario_passed_count += 1
                
    col_map, col_metrics = st.columns([3, 2])

    with col_map:
        st.subheader("Plot")
        sample_lat, sample_lon = cell_metadata[sorted_unique_cells[0]]["latlng"]
        m = folium.Map(location=[sample_lat, sample_lon], zoom_start=9, tiles="CartoDB positron")

        for cell in sorted_unique_cells:
            meta = cell_metadata[cell]
            label = cell_labels[cell]
            
            is_active_cell = (cell in cell_worst_channel_inr and cell_worst_channel_inr[cell] > -100)

            if not is_active_cell:
                fill_color = "#333333"
                fill_opacity = 0.15
                weight = 1.0
                tooltip_str = f"{label} (All Beams Dropped)"
            elif highlight_active:
                if cell in highlighted_cells:
                    specific_inr = cell_channel_metrics[cell].get(sel_channel, -100.0)
                    fill_color = "red" if specific_inr > INR_THRESHOLD else "blue"
                    fill_opacity = 0.65
                    weight = 2.5
                    tooltip_str = f"{label}: Channel {sel_channel} INR = {specific_inr:.2f} dB"
                else:
                    fill_color = "#D3D3D3"
                    fill_opacity = 0.1
                    weight = 0.5
                    tooltip_str = f"{label} (Not active on this channel)"
            else:
                worst_inr = cell_worst_channel_inr[cell]
                fill_color = "red" if worst_inr > INR_THRESHOLD else "blue"
                fill_opacity = 0.4
                weight = 1.5
                tooltip_str = f"{label}: Peak INR = {worst_inr:.2f} dB"

            popup_html = f"<b>{label}</b>"
            if not is_active_cell:
                popup_html += "<br><span style='color:red;'>Dropped - Fails Criteria</span>"
            elif highlight_active and cell in highlighted_cells:
                popup_html += f"<br>Layer Channel {sel_channel} INR: {cell_channel_metrics[cell].get(sel_channel, -100.0):.2f} dB"
            else:
                popup_html += f"<br>Worst-Case Channel: {cell_worst_channel_inr[cell]:.2f} dB"

            folium.Polygon(
                locations=meta["boundary"],
                color=fill_color if not highlight_active or cell in highlighted_cells else "#A9A9A9",
                weight=weight,
                fill=True,
                fill_color=fill_color,
                fill_opacity=fill_opacity,
                tooltip=tooltip_str,
                popup=folium.Popup(popup_html, max_width=200)
            ).add_to(m)

        st_folium(m, width=700, height=520, returned_objects=[], key=f"map_{time_input_seconds}_{selected_profile}_{run_mode}")

    with col_metrics:
        st.subheader("Scenario Metrics")
        if highlight_active:
            st.info(f"Showing metrics for **{selected_profile}**")
            sm1, sm2 = st.columns(2)
            with sm1:
                st.metric("Active Beams (Current / Original)", f"{scenario_total_count} / {scenario_original_count}")
            with sm2:
                st.metric("Scenario Passed (≤ -12 dB)", f"{scenario_passed_count}")
        else:
            st.info("Showing metrics across **All Beam Profiles**")
            gm1, gm2 = st.columns(2)
            with gm1:
                st.metric("Global Active Beams (Current / Original)", f"{total_active_beams} / {total_original_beams}")
            with gm2:
                st.metric("Global Passed Cells", f"{global_safe_cells_count}")
                
        st.markdown("---")
        st.markdown("#### Scenario INR Analytics")

        if highlight_active:
            scenario_inr_values = [
                cell_channel_metrics[cell].get(sel_channel, -100.0)
                for cell in highlighted_cells
            ]
            analytics_label = f"Channel {sel_channel}"
        else:
            scenario_inr_values = [
                inr_value
                for cell_metrics in cell_channel_metrics.values()
                for inr_value in cell_metrics.values()
                if np.isfinite(inr_value) and inr_value > -100.0
            ]
            analytics_label = "All Active Channels"

        if scenario_inr_values:
            scenario_min_inr = np.min(scenario_inr_values)
            scenario_median_inr = np.median(scenario_inr_values)
            scenario_max_inr = np.max(scenario_inr_values)

            st.caption(f"INR statistics for: **{analytics_label}**")
            st.metric("Minimum INR", f"{scenario_min_inr:.2f} dB")
            st.metric("Median INR", f"{scenario_median_inr:.2f} dB")
            st.metric("Maximum INR", f"{scenario_max_inr:.2f} dB")
        else:
            st.info("No active beams in this scenario to analyze.")

    st.markdown("---")
    st.subheader("Dropped Beams Log")
    
    if run_mode == "Show Original (Baseline)":
        st.caption("Baseline execution Pruning is disabled. Turn on 'Show Optimized Mapping' to view dropped beams.")
    elif dropped_beams:
        st.warning(f"**{len(dropped_beams)}** beam mappings were dropped to enforce INR thresholds.")
        
        dropped_records = []
        for d in dropped_beams:
            freq_ghz = get_channel_frequency(d["channel"]) / 1e9
            dropped_records.append({
                "Cell Identifier": d["cell"],
                "Friendly Name": cell_labels.get(d["cell"], "N/A"),
                "Channel Index": d["channel"],
                "Frequency (GHz)": f"{freq_ghz:.3f}",
                "Serving Satellite ID": d["sat"],
                "Drop Reason Flag": d["reason"]
            })
            
        dropped_df = pd.DataFrame(dropped_records)
        st.dataframe(
            dropped_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Drop Reason Flag": st.column_config.TextColumn(help="Reason for dropping this specific assignment")
            }
        )
    else:
        st.success("All assignments successfully processed! No beams were required to be dropped.")