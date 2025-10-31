# The MIT License (MIT)
#
# Copyright (c) 2020 ETH Zurich
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from .graph_tools import *
from utils.isls import *
from utils.ground_stations import read_ground_stations_basic
from utils.tles import *
import exputil
import networkx as nx
import json
from astropy import units as u
from utils.distance_tools import *
from utils.cells import *
import utils.global_variables as global_vars
import sys, pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONSTELLATION_SEARCH_PATHS = [
    PROJECT_ROOT / "constellation_configs" / "gen_data",
    PROJECT_ROOT / "constellation_configurations" / "gen_data",
    PROJECT_ROOT / "constellation_configurations" / "configs",
]
INPUTS_DIR = PROJECT_ROOT / "inputs"

def generate_satellite_shell_index(num_satellites, num_orbits, num_sats_per_orbit):
    satellites_shell_idx = {}
    idx = 0
    sats_so_far = 0
    for i in range(num_satellites):
        if i == (num_orbits[idx] * num_sats_per_orbit[idx]) + sats_so_far:
            idx += 1
        
        satellites_shell_idx[i] = idx

    return satellites_shell_idx

def precompute_potential_satellites(cell_id, satellites, epoch_str, time_str, bounding_dist_m):
    potential_satellites = []
    
    for sid, satellite in enumerate(satellites):
        if distance_m_ground_station_to_cell(cell_id, satellite, epoch_str, time_str) < bounding_dist_m:
            potential_satellites.append(sid)
    
    return potential_satellites

def resolve_constellation_dir(constellation_config: str) -> Path:
    candidate = Path(constellation_config)
    if candidate.is_absolute():
        if candidate.is_dir():
            return candidate
    else:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    for root in CONSTELLATION_SEARCH_PATHS:
        if not root.exists():
            continue
        rooted = root / constellation_config
        if rooted.exists() and rooted.is_dir():
            return rooted
    raise FileNotFoundError(
        f"Constellation configuration '{constellation_config}' not found in expected locations."
    )


def resolve_cells_path(constellation_config: str, country: str, constellation_dir: Path) -> Path:
    candidates = []
    cells_dir = INPUTS_DIR / "cells"
    base_name = f"{country}.txt"
    candidates.append(cells_dir / base_name)
    if "_" in constellation_config:
        parts = constellation_config.split("_cells_", 1)
        if len(parts) > 1:
            suffix = parts[1].replace("/", "_")
            candidates.append(cells_dir / f"{country}_{suffix}.txt")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find cell allocation for '{country}'. Tried {[str(p) for p in candidates]}."
    )


def resolve_starlink_cells_path(constellation_dir: Path) -> Path:
    inputs_starlink = INPUTS_DIR / "cells" / "starlink_cells.txt"
    if inputs_starlink.exists():
        return inputs_starlink
    raise FileNotFoundError(
        "Starlink cell population file not found alongside constellation configuration "
        "or under inputs/cells/starlink_cells.txt."
    )


GROUND_STATION_FILE_MAP = {
    "ground_stations_starlink": "ground_stations_starlink.basic.txt",
    "ground_stations_top_100": "ground_stations_cities_sorted_by_estimated_2025_pop_top_100.basic.txt",
    "ground_stations_top_1000": "ground_stations_cities_sorted_by_estimated_2025_pop_top_1000.basic.txt",
    "ground_stations_atlanta": "ground_stations_atlanta.basic.txt",
    "ground_stations_sydney": "ground_stations_sydney.basic.txt",
    "ground_stations_fiji": "ground_stations_fiji.basic.txt",
    "three_ground_stations": "ground_stations_starlink.basic.txt",  # default subset maps to starlink list
}


def extract_ground_station_token(constellation_config: str) -> str:
    if "_cells_" not in constellation_config:
        return "ground_stations_starlink"
    prefix = constellation_config.split("_cells_", 1)[0]
    if "_isls_" not in prefix:
        return "ground_stations_starlink"
    return prefix.split("_isls_", 1)[1]


def resolve_ground_station_file(constellation_config: str) -> Path:
    token = extract_ground_station_token(constellation_config)
    filename = GROUND_STATION_FILE_MAP.get(token, "ground_stations_starlink.basic.txt")
    path = INPUTS_DIR / "groundstations" / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Ground station catalogue '{filename}' not found under inputs/groundstations."
        )
    return path


def load_ground_stations(constellation_config: str):
    gs_file = resolve_ground_station_file(constellation_config)
    basics = read_ground_stations_basic(str(gs_file))
    extended = []
    for entry in basics:
        cartesian = geodetic2cartesian(
            float(entry["latitude_degrees_str"]),
            float(entry["longitude_degrees_str"]),
            entry["elevation_m_float"],
        )
        extended.append({
            "gid": entry["gid"],
            "name": entry["name"],
            "latitude_degrees_str": entry["latitude_degrees_str"],
            "longitude_degrees_str": entry["longitude_degrees_str"],
            "elevation_m_float": entry["elevation_m_float"],
            "cartesian_x": cartesian[0],
            "cartesian_y": cartesian[1],
            "cartesian_z": cartesian[2],
        })
    return extended

def compute_graph_subdir(constellation_config: str) -> str:
    name = Path(constellation_config).name
    parts = name.split("_")
    if len(parts) > 2:
        prefix = parts[:-2]
        if prefix:
            return "_".join(prefix)
    return name


def generate_all_graphs(graph_path, constellation_config, country,
                         dynamic_state_update_interval_ms, simulation_start_time_s, simulation_end_time_s):

    print(f"Start time: {simulation_start_time_s}s, End time:{simulation_end_time_s}s")

    graph_path = Path(graph_path)
    graph_path.mkdir(parents=True, exist_ok=True)

    constellation_dir = resolve_constellation_dir(constellation_config)

    isls_path = constellation_dir / "isls.txt"
    tles_path = constellation_dir / "tles.txt"
    description_path = constellation_dir / "description.txt"
    for path, label in [
        (isls_path, "ISL list"),
        (tles_path, "TLE file"),
        (description_path, "description file"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required {label} missing at {path}.")

    cells_path = resolve_cells_path(constellation_config, country, constellation_dir)
    starlink_cells_path = resolve_starlink_cells_path(constellation_dir)

    # Variables (load in for each thread such that they don't interfere)
    ground_stations = load_ground_stations(constellation_config)
    tles = read_tles(str(tles_path))
    satellites = tles["satellites"]
    list_isls = read_isls(str(isls_path), len(satellites))
    epoch = tles["epoch"]
    cells = read_cells(str(cells_path))
    starlink_cells = read_cells_starlink(str(starlink_cells_path))
    description = exputil.PropertiesConfig(str(description_path))

    # Derivatives
    simulation_start_time_ns = simulation_start_time_s * 1000 * 1000 * 1000
    simulation_end_time_ns = simulation_end_time_s * 1000 * 1000 * 1000
    dynamic_state_update_interval_ns = dynamic_state_update_interval_ms * 1000 * 1000
    n_shells = exputil.parse_positive_int(description.get_property_or_fail("num_shells"))
    if n_shells == 1:
        max_gsl_length_m = exputil.parse_positive_float(description.get_property_or_fail("max_gsl_length_m"))
        max_isl_length_m = exputil.parse_positive_float(description.get_property_or_fail("max_isl_length_m"))
    else:
        num_orbits = json.loads(description.get_property_or_fail("num_orbits"))
        num_sats_per_orbit = json.loads(description.get_property_or_fail("num_sats_per_orbit"))
        max_gsl_length_m = json.loads(description.get_property_or_fail("max_gsl_length_m"))
        max_isl_length_m = json.loads(description.get_property_or_fail("max_isl_length_m"))
        satellites_shell_idx = generate_satellite_shell_index(len(satellites), num_orbits, num_sats_per_orbit)
        print(num_orbits, num_sats_per_orbit, max_gsl_length_m, max_isl_length_m, len(satellites))

    # Precompute potential satellites for each UT
    # TODO: Consider other bounding distances such as max GSL length * 2.
    # Or since we know the simulation length and satellite speed, we can compute the max distance a satellite can travel as the bounding distance.
    bounding_dist_m = 5000000 # 5000 km
    initial_time = epoch + simulation_start_time_ns * u.ns
    
    # cell_potential_satellites = {}
    # for cell in cells:
    #     if cell["cell"] not in cell_potential_satellites:
    #         cell_potential_satellites[cell["cell"]] = precompute_potential_satellites(cell["cell"], satellites, str(epoch), str(initial_time), bounding_dist_m)
    
    # for cell in starlink_cells:
    #     if cell["cell"] not in cell_potential_satellites:
    #         cell_potential_satellites[cell["cell"]] = precompute_potential_satellites(cell["cell"], satellites, str(epoch), str(initial_time), bounding_dist_m)
        
    # potential_satellites = precompute_potential_satellites(user_terminals, satellites, str(epoch), str(initial_time), bounding_dist_m)

    for t in range(simulation_start_time_ns, simulation_end_time_ns, dynamic_state_update_interval_ns):
        print(t)
        graph_path_filename = graph_path / f"graph_{t}.txt"
        # Time
        time = epoch + t * u.ns

        # Graph
        graph = nx.DiGraph()
        sat_capacity = [0] * len(satellites)

        # Satellite to GS GSLs
        # sat_potential_ground_stations = {}
        # for sid in range(len(satellites)):
        #     sat_potential_ground_stations[sid] = {}

        for ground_station in ground_stations:
            # Find satellites in range
            for sid in range(len(satellites)):
                if n_shells == 1:
                    max_length = max_gsl_length_m
                else:                    
                    max_length = max_gsl_length_m[satellites_shell_idx[sid]]
                
                distance_m = distance_m_ground_station_to_satellite(ground_station, satellites[sid], str(epoch), str(time))
                # print(f"GS {ground_station['gid']} to satellite {sid} distance: {distance_m} with max length {max_length}")
                if distance_m <= max_length:
                    # print(f"GS {ground_station['gid']} to satellite {sid} distance: {distance_m} with max length {max_length}")
                    # sat_potential_ground_stations[sid][ground_station["gid"]] = distance_m
                    graph.add_edge(len(satellites) + ground_station["gid"], sid, weight=round(distance_m), capacity=0)

        # ground_station_capacity_used = {}
        # for gs_index in range(len(ground_stations)):
        #     ground_station_capacity_used[gs_index] = 0

        
        # sat_capacity = {}
        # for sat in range(len(satellites)):
        #     sat_capacity[sat] = 0

        # for gs in range(len(ground_stations)):
        #     gs_id = len(satellites) + gs

        #     sat_neighbors = graph[gs_id]
        #     sat_neighbors = [sat for sat in sat_neighbors if sat < len(satellites)]
        #     # print(sat_neighbors)
            
        #     while ground_station_capacity_used[gs] < global_vars.ground_station_gsl_capacity:
        #         # print(ground_station_capacity_used[gs])
        #         sat_neighbors = [sat for sat in sat_neighbors if sat_capacity[sat] < global_vars.ground_station_sat_capacity]
        #         if len(sat_neighbors) == 0:
        #             break
        #         sat_neighbors = sorted(sat_neighbors, key=lambda x: (sat_capacity[x], x))

        #         for neighbor in sat_neighbors:
        #             # print(gs, neighbor, sat_capacity[neighbor], ground_station_capacity_used[gs])
        #             if ground_station_capacity_used[gs] >= global_vars.ground_station_gsl_capacity:
        #                 break
        #             if sat_capacity[neighbor] < global_vars.ground_station_sat_capacity:
        #                 sat_capacity[neighbor] += global_vars.ka_beam_capacity
        #                 ground_station_capacity_used[gs] += global_vars.ka_beam_capacity
        #                 graph[gs_id][neighbor]['capacity'] += global_vars.ka_beam_capacity

        # for sat in range(len(satellites)):
        #     if sat in graph.nodes:
        #         graph.nodes[sat]['gsl_capacity'] = sat_capacity[sat]
        #     else:
        #         graph.add_node(sat, gsl_capacity=sat_capacity[sat])
        #     sat_capacity[sat] = 0

        # for sat in range(len(satellites)):
        #     # get gs neighbors as the keys of the sat_potential_ground_stations dict
        #     sat_gs_neighbors = list(sat_potential_ground_stations[sat].keys())

        #     if len(sat_gs_neighbors) > 0:
        #         sat_gs_neighbors = sorted(sat_gs_neighbors, key=lambda x: ground_station_capacity_used[x])
        #         # distributed sat_gs_max_capacity among all ground stations but ensuring that they don't exceed ground_station_gsl_capacity
                
        #         for gs in sat_gs_neighbors:
        #             sat_capacity_left = sat_gs_max_capacity - sat_capacity[sat]
        #             gs_capacity = min(ground_station_gsl_capacity - ground_station_capacity_used[gs], sat_capacity_left)
        #             gs_capacity = (gs_capacity // global_vars.ka_beam_capacity) * global_vars.ka_beam_capacity
        #             graph.add_edge(gs + len(satellites), sat, weight=sat_potential_ground_stations[sat][gs], capacity=gs_capacity)
        #             ground_station_capacity_used[gs] += gs_capacity
        #             sat_capacity[sat] += gs_capacity

        #         # sat_neighbor_ids = list(graph.predecessors(sat))
        #         # for neighbor in sat_neighbor_ids:
        #         #     sat_capacity[sat] += graph[neighbor][sat]['capacity']

        # ISLs
        for (a,b) in list_isls:            
            if n_shells == 1:
                max_length = max_isl_length_m
            else:
                max_length = max_isl_length_m[satellites_shell_idx[a]]

            # Only ISLs which are close enough are considered
            sat_distance_m = distance_m_between_satellites(satellites[a], satellites[b], str(epoch), str(time))
            if sat_distance_m <= max_length:
                graph.add_edge(a, b, weight=round(sat_distance_m), capacity=global_vars.isl_capacity)
                graph.add_edge(b, a, weight=round(sat_distance_m), capacity=global_vars.isl_capacity)

                sat_capacity[a] += global_vars.isl_capacity
                sat_capacity[b] += global_vars.isl_capacity

        # Cell to Satellite GSLs        
        for cell in cells:
            cell_id = cell["cell"]
            for sid in range(len(satellites)):            
                if n_shells == 1:
                    max_length = max_gsl_length_m
                else:
                    max_length = max_gsl_length_m[satellites_shell_idx[sid]]

                distance_m = distance_m_ground_station_to_cell(cell_id, satellites[sid], str(epoch), str(time))
                # print(f"Cell {cell_id} to satellite {sid} distance: {distance_m} with max length {max_length}")
                if distance_m < max_length:
                    # print(f"Cell {cell_id} to satellite {sid} distance: {distance_m} with max length {max_length}")
                    graph.add_edge(sid, cell_id, weight=round(distance_m), capacity=global_vars.ku_beam_capacity)

        # Starlink Cell to Satellite GSLs
        for cell in starlink_cells:
            cell_id = cell["cell"]
            for sid in range(len(satellites)):
                if n_shells == 1:
                    max_length = max_gsl_length_m
                else:
                    max_length = max_gsl_length_m[satellites_shell_idx[sid]]

                distance_m = distance_m_ground_station_to_cell(cell_id, satellites[sid], str(epoch), str(time))
                # print(f"Cell {cell_id} to satellite {sid} distance: {distance_m} with max length {max_length}")
                if distance_m < max_length:
                    # print(f"Cell {cell_id} to satellite {sid} distance: {distance_m} with max length {max_length}")
                    graph.add_edge(sid, cell_id, weight=round(distance_m), capacity=global_vars.ku_beam_capacity)

        # for sat in range(len(satellites)):
        #     if sat in graph.nodes:
        #         graph.nodes[sat]['isl_capacity'] = sat_capacity[sat]

        with open(graph_path_filename, "wb") as f:
            pickle.dump(graph, f)

def main():
    args = sys.argv[1:]
    if len(args) == 6:
        graphs_root = Path(args[0])
        country = args[1]
        constellation_config = args[2]
        update_interval_ms = int(args[3])
        simulation_start_time_s = int(args[4])
        simulation_end_time_s = int(args[5])

        graph_subdir = compute_graph_subdir(constellation_config)
        base_output_dir = graphs_root / graph_subdir / f"{update_interval_ms}ms"
        base_output_dir.mkdir(parents=True, exist_ok=True)

        generate_all_graphs(
            base_output_dir,
            constellation_config,
            country,
            update_interval_ms,
            simulation_start_time_s,
            simulation_end_time_s
        )
    else:
        print("Invalid argument selection for generate_directed_graphs.py.")

if __name__ == "__main__":
    main()
