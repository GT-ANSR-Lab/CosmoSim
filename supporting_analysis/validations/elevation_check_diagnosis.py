#!/usr/bin/env python3

"""
Checks all generated graphs across all time snapshots for 25 deg min elevation violations and gives a detailed diagnosis of elevation violating edges
"""

import os
import re
import pickle
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import networkx as nx
import h3
import ephem

BASE_DIR = Path(__file__).resolve().parents[2]

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

try:
    from utils.tles import read_tles
except ImportError:
    read_tles = None

def load_pyephem_satellites(tle_path):
    """Loads TLEs directly into PyEphem EarthSatellite objects mapped by their ID."""
    sats = {}
    with open(tle_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    while i < len(lines) - 2:
        m = re.search(r"Starlink 5-Shells (\d+)", lines[i])
        if m:
            sid = int(m.group(1))
            sats[sid] = ephem.readtle(lines[i], lines[i+1], lines[i+2])
            i += 3
        else:
            i += 1
    return sats

def get_tle_info(tle_path):
    """Fallback manual parser if utils.tles.read_tles is unavailable."""
    sat_count = 0
    epoch_datetime = datetime(2022, 1, 1, 0, 0, 0)
    with open(tle_path, 'r') as f:
        for line in f:
            if line.startswith('1 ') and len(line) >= 32:
                sat_count += 1
                if sat_count == 1:
                    try:
                        epoch_str = line[18:32].strip()
                        year_short = int(epoch_str[:2])
                        year = 2000 + year_short if year_short < 57 else 1900 + year_short
                        day_of_year = float(epoch_str[2:])
                        base_time = datetime(year, 1, 1)
                        epoch_datetime = base_time + timedelta(days=day_of_year - 1)
                    except Exception:
                        pass
    return epoch_datetime, sat_count

def sanitize_to_pydatetime(dt_val):
    if isinstance(dt_val, datetime):
        return dt_val
    if isinstance(dt_val, np.datetime64):
        unix_epoch = np.datetime64('1970-01-01T00:00:00')
        seconds = (dt_val - unix_epoch) / np.timedelta64(1, 's')
        return datetime.utcfromtimestamp(float(seconds))
    if hasattr(dt_val, 'to_pydatetime'):
        return dt_val.to_pydatetime()
    if isinstance(dt_val, str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
            try:
                return datetime.strptime(dt_val, fmt)
            except ValueError:
                pass
    raise ValueError(f"Unable to convert epoch object of type {type(dt_val)} to standard datetime.")

def load_allowed_haiti_cells(cells_file_path):
    allowed_cells = set()
    if not os.path.exists(cells_file_path):
        print(f"[!] Warning: Strict cell restriction file not found at {cells_file_path}")
        return allowed_cells
        
    with open(cells_file_path, 'r') as f:
        for line in f:
            cleaned = line.strip()
            if '"cell":' in cleaned:
                m = re.search(r'"cell"\s*:\s*"([a-fA-F0-9]+)"', cleaned)
                if m:
                    allowed_cells.add(m.group(1))
            elif cleaned and not cleaned.startswith('#'):
                tokens = re.split(r'[\s,]+', cleaned)
                for t in tokens:
                    if len(t) >= 12 and h3.is_valid_cell(t):
                        allowed_cells.add(t)
    return allowed_cells

def inspect_generated_graphs(tle_path, graphs_dir, cells_file_path, min_elevation_deg=25.0, max_debug_violations=5):
    print("[-] Extracting simulation parameters...")
    
    if read_tles is not None:
        try:
            tle_data = read_tles(tle_path)
            raw_epoch = tle_data.get("epoch") if isinstance(tle_data, dict) else getattr(tle_data, "epoch", None)
            num_satellites = len(tle_data.get("satellites", [])) if isinstance(tle_data, dict) else len(load_pyephem_satellites(tle_path))
            if raw_epoch is None:
                raise ValueError("Epoch not found in read_tles output.")
            epoch_base = sanitize_to_pydatetime(raw_epoch)
            print("[+] Successfully synchronized epoch via utils.tles.read_tles")
        except Exception as e:
            print(f"[!] Target pipeline reader failed ({e}). Falling back to manual parser.")
            epoch_base, num_satellites = get_tle_info(tle_path)
    else:
        epoch_base, num_satellites = get_tle_info(tle_path)
    
    satellites = load_pyephem_satellites(tle_path)
    haiti_cells_filter = load_allowed_haiti_cells(cells_file_path)
    
    print(f"[+] Loaded {len(haiti_cells_filter)} strict target cells from constraint file.")
    print(f"[+] Found TLE Base Epoch: {epoch_base}")
    print(f"[+] Total Constellation Satellites: {num_satellites}")
    print(f"[+] Target Minimum Elevation Threshold: {min_elevation_deg}°\n")

    if not os.path.exists(graphs_dir):
        print(f"[!] Directory path error: {graphs_dir}")
        return

    graph_files = [f for f in os.listdir(graphs_dir) if f.startswith("graph_") and f.endswith(".txt")]
    if not graph_files:
        print(f"[!] No valid graph snapshot files discovered in target dir.")
        return
        
    graph_files.sort(key=lambda f: int(re.findall(r'\d+', f)[0]))

    print(f"{'Snapshot File Source':<25} | {'Active Sats':<11} | {'Active Cells':<12} | {'GSLs Checked':<12} | {'Elevation Angle Violations (<25°)':<31} | {'Violation Values (Deg Deficit)':<35} | {'Critical (Below Horizon <0°)':<28}")
    print("-" * 167)
    
    total_edges_checked = 0
    total_elevation_violations = 0
    total_critical_violations = 0
    all_violation_degrees = []
    debug_violations_printed = 0
    
    cached_cell_coords = {}
    failing_edges_sample = []

    observer = ephem.Observer()
    observer.elevation = 0.0

    for graph_file in graph_files:
        timestamp_ns = int(re.findall(r'\d+', graph_file)[0])
        graph_path = os.path.join(graphs_dir, graph_file)
        
        elevation_violations_in_file = 0
        critical_violations_in_file = 0
        file_violations = []
        edges_in_file = 0
        
        active_sats_in_file = set()
        active_cells_in_file = set()
        
        try:
            current_time = epoch_base + timedelta(seconds=timestamp_ns / 1e9)
            ephem_time_str = current_time.strftime("%Y/%m/%d %H:%M:%S.%f")
            
            with open(graph_path, 'rb') as f:
                graph = pickle.load(f)
            
            if not isinstance(graph, (nx.Graph, nx.DiGraph)):
                continue

            def is_satellite(node):
                try:
                    val = int(node)
                    return 0 <= val < num_satellites
                except (ValueError, TypeError):
                    return False

            for source, target in graph.edges():
                src_is_sat = is_satellite(source)
                tgt_is_sat = is_satellite(target)
                
                if src_is_sat != tgt_is_sat:
                    sat_id = int(source) if src_is_sat else int(target)
                    ground_node = target if src_is_sat else source
                    
                    if sat_id not in satellites or ground_node not in haiti_cells_filter:
                        continue

                    if ground_node not in cached_cell_coords:
                        try:
                            if h3.is_valid_cell(ground_node):
                                cached_cell_coords[ground_node] = h3.cell_to_latlng(ground_node)
                            else:
                                continue
                        except Exception:
                            continue
                    
                    active_sats_in_file.add(sat_id)
                    active_cells_in_file.add(ground_node)
                    edges_in_file += 1

                    lat, lon = cached_cell_coords[ground_node]
                    
                    observer.date = ephem_time_str
                    observer.lat = str(lat)
                    observer.lon = str(lon)
                    
                    sat = satellites[sat_id]
                    sat.compute(observer)
                    
                    current_elevation = np.degrees(float(sat.alt))
                    
                    if current_elevation < min_elevation_deg:
                        elevation_violations_in_file += 1
                        deficit = min_elevation_deg - current_elevation
                        file_violations.append(round(deficit, 6))
                        
                        if debug_violations_printed < max_debug_violations:
                            edge_data = graph[source][target]
                            edge_weight = edge_data.get('weight', edge_data.get('distance', 'N/A'))
                            failing_edges_sample.append({
                                "file": graph_file,
                                "cell": ground_node,
                                "sat": sat_id,
                                "elev": current_elevation,
                                "weight": edge_weight,
                                "slant_range_m": sat.range
                            })
                            debug_violations_printed += 1
                    
                    if current_elevation < 0.0:
                        critical_violations_in_file += 1
                        
        except Exception as e:
            print(f"\n[!] Read exception inside file {graph_file}: {e}")
            continue
        
        violations_str = ", ".join(f"{v}°" for v in file_violations) if file_violations else "None"
        
        print(f"{graph_file:<25} | {len(active_sats_in_file):<11} | {len(active_cells_in_file):<12} | {edges_in_file:<12} | {elevation_violations_in_file:<31} | {violations_str:<35} | {critical_violations_in_file:<28}")
        
        total_edges_checked += edges_in_file
        total_elevation_violations += elevation_violations_in_file
        total_critical_violations += critical_violations_in_file
        all_violation_degrees.extend(file_violations)

    total_violations_str = ", ".join(f"{v}°" for v in all_violation_degrees) if all_violation_degrees else "None"

    print("-" * 167)
    print(f"{'TOTAL MATRIX RUN SUMMARY':<25} | {'-':<11} | {'-':<12} | {total_edges_checked:<12} | {total_elevation_violations:<31} | {total_violations_str:<35} | {total_critical_violations:<28}")
    print("-" * 167)
    
    if all_violation_degrees:
        max_violation = max(all_violation_degrees)
        print(f"\n[+] Highest Violation Angle Difference (Deficit): {max_violation}°")
    else:
        print("\n[+] Highest Violation Angle Difference (Deficit): None (No violations found)")
    
    if failing_edges_sample:
        print("\n" + "="*45 + " EDGE VIOLATION ANALYSER " + "="*45)
        print(f"{'Snapshot File':<20} | {'H3 Cell ID':<15} | {'Sat ID':<8} | {'Elevation':<10} | {'Graph Weight':<15} | {'PyEphem Slant Range (m)':<22}")
        print("-" * 115)
        for sample in failing_edges_sample:
            weight_str = f"{sample['weight']:.2f}" if isinstance(sample['weight'], (int, float)) else str(sample['weight'])
            print(f"{sample['file']:<20} | {sample['cell']:<15} | {sample['sat']:<8} | {sample['elev']:>8.2f}° | {weight_str:<15} | {sample['slant_range_m']:>22.2f}")
        print("=" * 115)


if __name__ == "__main__":
    TLE_PATH = BASE_DIR / "constellation_configurations" / "configs" / "starlink_5shells" / "tles.txt"
    GRAPHS_DIR = BASE_DIR / "graph_generation" / "graphs" / "starlink_5shells" / "haiti"
    CELLS_PATH = BASE_DIR / "inputs" / "cells" / "haiti.txt"
    
    inspect_generated_graphs(TLE_PATH, GRAPHS_DIR, CELLS_PATH, min_elevation_deg=25.0)
