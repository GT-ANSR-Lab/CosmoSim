#!/usr/bin/env python3
import os
import sys
import pickle
from pathlib import Path
from collections import defaultdict
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
    N0_DB,
    ecef,
    calculate_received_interference_watt,
    get_satellite_positions,
    get_channel_frequency,
)

st.set_page_config(layout="wide", page_title="INR Analyzer")
st.title("INR Analysis")

BASE_DATA_DIR = COSMOSIM_ROOT / "data"
TLE_FILE = COSMOSIM_ROOT / "constellation_configurations" / "configs" / "starlink_5shells" / "tles.txt"

CONST = "starlink_5shells"
GS = "ground_stations"
INR_THRESHOLD = -12.0 #-6.0


@st.cache_resource(show_spinner="Propagating constellation geometry from TLEs...")
def cached_satellite_positions(tle_path, time_offset_seconds):
    return get_satellite_positions(tle_path, time_offset_seconds)


@st.cache_data(show_spinner="Running system co-channel RF analytics...")
def compute_network_interference(sorted_unique_cells, cell_map, cell_metadata, sat_positions):
    cell_channel_metrics = defaultdict(dict)
    cell_worst_channel_inr = {}
    global_safe_cells_count = 0
    
    active_transmitters = defaultdict(list)
    for cell_id in sorted_unique_cells:
        for slot, target_data in cell_map[cell_id].items():
            sat_id = int(target_data["sat"])
            r_idx = int(target_data["reuse"])
            if sat_id in sat_positions:
                active_transmitters[slot].append({
                    "cell_id": str(cell_id),
                    "sat_id": sat_id,
                    "reuse": r_idx,
                    "ecef": cell_metadata[cell_id]["ecef"]
                })

    for target_cell in sorted_unique_cells:
        meta_target = cell_metadata[target_cell]
        tgt_ecef = meta_target["ecef"]
        
        for slot, target_data in cell_map[target_cell].items():
            target_reuse = int(target_data["reuse"])
            target_sat_id = int(target_data["sat"])
            
            if target_sat_id not in sat_positions:
                continue
            
            sat_tgt_vec = sat_positions[target_sat_id]
            channel_interference_watts = 0.0
            
            co_channel_transmitters = active_transmitters.get(slot, [])
            evaluated_beams = set()
            
            for tx in co_channel_transmitters:
                if (
                    tx["cell_id"] == str(target_cell)
                    and tx["sat_id"] == target_sat_id
                    and tx["reuse"] == target_reuse
                ):
                    continue
                
                beam_key = (tx["sat_id"], tx["reuse"])
                if beam_key in evaluated_beams:
                    continue
                evaluated_beams.add(beam_key)

                p_watts = calculate_received_interference_watt(
                    u_tgt_pos=tgt_ecef,
                    u_inf_serving_pos=tx["ecef"], 
                    sat_tgt_pos=sat_tgt_vec,
                    sat_inf_pos=sat_positions[tx["sat_id"]],
                    channel_idx=int(slot)
                )
                channel_interference_watts += p_watts

            if channel_interference_watts > 0:
                channel_inr_db = (10 * np.log10(channel_interference_watts)) - N0_DB
            else:
                channel_inr_db = -100.0
                
            cell_channel_metrics[target_cell][slot] = channel_inr_db

        if cell_channel_metrics[target_cell]:
            worst_inr = max(cell_channel_metrics[target_cell].values())
            cell_worst_channel_inr[target_cell] = worst_inr
        else:
            cell_worst_channel_inr[target_cell] = -100.0

        if cell_worst_channel_inr[target_cell] <= INR_THRESHOLD:
            global_safe_cells_count += 1
            
    return dict(cell_channel_metrics), dict(cell_worst_channel_inr), global_safe_cells_count


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
st.sidebar.info(f"**Target Data Path:**\n`{TARGET_PKL_PATH}`")

if not TARGET_PKL_PATH.exists():
    st.error(f"Snapshot data file not found at:\n`{TARGET_PKL_PATH}`")
else:
    sat_positions = cached_satellite_positions(TLE_FILE, time_input_seconds)
    if not sat_positions:
        st.error("TLE source file structure error or missing data nodes.")
        st.stop()

    with TARGET_PKL_PATH.open("rb") as f:
        raw = pickle.load(f)

    cell_map = defaultdict(dict)
    cells = set()
    matrix_occupancy_map = defaultdict(lambda: defaultdict(list))

    for k, v in raw.items():
        cell, slot_str = k.rsplit("_", 1)
        slot = int(slot_str)
        reuse, sat, satbeam = v.split("_")
        reuse_idx = int(reuse)
        
        cell_map[cell][slot] = {"reuse": reuse_idx, "sat": int(sat)}
        cells.add(cell)

    sorted_unique_cells = sorted(list(cells))
    cell_labels = {cell: f"Cell {idx + 1}" for idx, cell in enumerate(sorted_unique_cells)}
    total_cells = len(cell_labels)

    for cell in sorted_unique_cells:
        for slot, data in cell_map[cell].items():
            r_idx = data["reuse"]
            matrix_occupancy_map[slot][r_idx].append(cell)

    cell_metadata = {}
    for cell in sorted_unique_cells:
        lat, lon = h3.cell_to_latlng(cell)
        raw_boundary = h3.cell_to_boundary(cell)
        cell_metadata[cell] = {
            "latlng": [lat, lon],
            "ecef": ecef(lat, lon),
            "boundary": [[coords[0], coords[1]] for coords in raw_boundary]
        }

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
    
    if highlight_active:
        sel_channel = int(selected_profile.split(" ")[1])
        for ru in range(4):
            highlighted_cells.update(matrix_occupancy_map[sel_channel][ru])
        scenario_total_count = len(highlighted_cells)
        
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
            
            if highlight_active:
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
                    tooltip_str = f"{label} (Not active on layer)"
            else:
                worst_inr = cell_worst_channel_inr[cell]
                fill_color = "red" if worst_inr > INR_THRESHOLD else "blue"
                fill_opacity = 0.4
                weight = 1.5
                tooltip_str = f"{label}: Peak {worst_inr:.2f} dB"

            popup_html = f"<b>{label}</b>"
            if highlight_active and cell in highlighted_cells:
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

        st_folium(m, width=700, height=520, returned_objects=[], key=f"map_{time_input_seconds}_{selected_profile}")

    with col_metrics:
        st.subheader("Scenarios")
        if highlight_active:
            st.info(f"Showing metrics for **{selected_profile}**")
            sm1, sm2 = st.columns(2)
            with sm1:
                st.metric("Total Cells in Scenario", f"{scenario_total_count}")
            with sm2:
                st.metric("Scenario Passed Cells", f"{scenario_passed_count}")
        else:
            st.info("Showing metrics across **All Beam Profiles**")
            gm1, gm2 = st.columns(2)
            with gm1:
                st.metric("Total Constellation Cells", f"{total_cells}")
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
                if np.isfinite(inr_value)
            ]
            analytics_label = "All Cell × Channel Profiles"

        if scenario_inr_values:
            scenario_min_inr = np.min(scenario_inr_values)
            scenario_median_inr = np.median(scenario_inr_values)
            scenario_max_inr = np.max(scenario_inr_values)

            st.caption(f"INR statistics for: **{analytics_label}**")
            
            st.metric("Minimum INR", f"{scenario_min_inr:.2f} dB")
            st.metric("Median INR", f"{scenario_median_inr:.2f} dB")
            st.metric("Maximum INR", f"{scenario_max_inr:.2f} dB")
        else:
            st.info("No INR values available for the selected scenario.")

    st.markdown("---")
    st.subheader("Beam Configurations")

    long_matrix_records = []
    for raw_channel in range(8):
        freq_ghz = get_channel_frequency(raw_channel) / 1e9
        occupants_hashes = []
        for raw_reuse in range(4):
            occupants_hashes.extend(matrix_occupancy_map[raw_channel][raw_reuse])
            
        unique_occupants = sorted(list(set(occupants_hashes)))
        occupants_labels = [cell_labels[chash] for chash in unique_occupants]
        num_cells = len(occupants_labels)
        occupants_str = ", ".join(occupants_labels) if num_cells > 0 else "—"

        long_matrix_records.append({
            "Hardware Channel": f"Channel {raw_channel} ({freq_ghz:.3f} GHz)",
            "Number of Cells": num_cells,
            "Occupant Ground Cells": occupants_str
        })

    long_matrix_df = pd.DataFrame(long_matrix_records)
    st.dataframe(
        long_matrix_df, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Number of Cells": st.column_config.NumberColumn(help="Number of Cells sharing this specific physical hardware channel slot"),
            "Occupant Ground Cells": st.column_config.TextColumn(help="List of cells sharing this specific physical hardware channel slot")
        }
    )
    st.markdown("---")
    st.subheader("Per-Cell Spectrum Layer Analysis")
    
    selected_cell = st.selectbox(
        "Select a coordinate ground cell unit to view its complete physical hardware channel layout map:",
        options=sorted_unique_cells,
        format_func=lambda x: f"{cell_labels[x]} ({x})"
    )

    if selected_cell:
        st.markdown(f"#### Spectrum Profile Allocation Matrix for **{cell_labels[selected_cell]}**")
        
        channels_active = cell_channel_metrics[selected_cell]
        if channels_active:
            records = []
            for ch_id, inr_val in sorted(channels_active.items()):
                freq_ghz = get_channel_frequency(ch_id) / 1e9
                serving_sat = cell_map[selected_cell][ch_id]["sat"]
                reuse_grp = cell_map[selected_cell][ch_id]["reuse"]
                status = " CRITICAL" if inr_val > INR_THRESHOLD else " SAFE"
                records.append({
                    "Hardware Channel": f"Channel {ch_id} ({freq_ghz:.3f} GHz)",
                    "Assigned Sat ID": serving_sat,
                    "Reuse Factor Group": f"Reuse {reuse_grp}",
                    "Received INR (dB)": f"{inr_val:.2f} dB",
                    "Safety Status": status
                })
            
            df = pd.DataFrame(records)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No active channels mapped or assigned to this target cell location.")