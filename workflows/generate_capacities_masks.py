#!/usr/bin/env python3
"""Generate capacity traces with ground-station masking scenarios."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import exputil
import networkx as nx
import numpy as np
from networkx.algorithms.flow import max_flow_min_cost

import utils.global_variables as global_vars
from utils.ground_stations import read_ground_stations_extended
from utils.tles import read_tles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONSTELLATION_ROOT = PROJECT_ROOT / "constellation_configurations" / "configs"
GROUNDSTATION_ROOT = PROJECT_ROOT / "inputs" / "groundstations"
CELLS_ROOT = PROJECT_ROOT / "inputs" / "cells"
TERMINAL_ROOT = PROJECT_ROOT / "terminal_deployment" / "terminals"


MASK_GS: Dict[str, List[int]] = {
    "ghana": [0, 1],
    # extend with additional country-specific GS IDs as needed
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def constellation_dir(constellation_name: str) -> Path:
    path = CONSTELLATION_ROOT / constellation_name
    if not path.exists():
        raise FileNotFoundError(
            f"Constellation '{constellation_name}' not found at {path}. "
            "Generate it under constellation_configurations/configs first."
        )
    return path


def groundstations_path(name: str) -> Path:
    filename = name if name.endswith(".txt") else f"{name}.txt"
    path = GROUNDSTATION_ROOT / filename
    if not path.exists():
        raise FileNotFoundError(f"Ground station catalogue '{name}' not found at {path}")
    return path


def validate_country(country: str) -> None:
    cells_path = CELLS_ROOT / f"{country}.txt"
    if not cells_path.exists():
        raise FileNotFoundError(
            f"Country '{country}' missing cells file at {cells_path}. "
            "Generate inputs/cells/{country}.txt first."
        )


def resolve_terminals_path(cells_file: Path) -> Path:
    if cells_file.is_absolute() and cells_file.exists():
        return cells_file
    candidate = resolve_path(cells_file)
    if candidate.exists():
        return candidate
    fallback = TERMINAL_ROOT / cells_file.name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Terminal allocation file not found: {cells_file}")


def load_demands(demands_path: Path) -> np.ndarray:
    if not demands_path.exists():
        raise FileNotFoundError(f"Demands file not found: {demands_path}")
    values = np.atleast_1d(np.loadtxt(demands_path, dtype=float))
    if values.ndim != 1:
        raise ValueError(f"Demands file must be 1-D; got shape {values.shape} from {demands_path}")
    return values


def assign_sat_capacity(graph: nx.DiGraph, demands: np.ndarray, start_node: str, num_satellites: int) -> None:
    for sat in range(num_satellites):
        demand = float(demands[sat]) if sat < len(demands) else 0.0
        if demand <= 0:
            continue
        dummy = f"D{sat}"
        graph.add_edge(dummy, start_node, capacity=demand, weight=0)
        graph.add_edge(sat, dummy, capacity=demand, weight=0)


def remove_dummy_nodes(graph: nx.DiGraph, num_satellites: int) -> None:
    for sat in range(num_satellites):
        dummy = f"D{sat}"
        if graph.has_node(dummy):
            graph.remove_node(dummy)


def max_flow_capacities(
    demands: np.ndarray,
    graph: nx.DiGraph,
    super_source: str,
    super_sink: str,
    num_satellites: int,
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    assign_sat_capacity(graph, demands, super_source, num_satellites)
    flow_dict = max_flow_min_cost(graph, super_sink, super_source)
    capacity = sum(flow_dict[super_sink].values()) if super_sink in flow_dict else 0.0
    return capacity, flow_dict


def apply_mask(ground_stations: Sequence[Dict], gs_base_index: int, mask_mode: str, country: str) -> List[int]:
    mask_ids = MASK_GS.get(country.lower(), [])
    if not mask_ids:
        return []
    masked: List[int] = []
    for gs in ground_stations:
        node_id = gs_base_index + gs["gid"]
        if mask_mode == "on" and gs["gid"] not in mask_ids:
            masked.append(node_id)
        if mask_mode == "off" and gs["gid"] in mask_ids:
            masked.append(node_id)
    return masked


def generate_capacities_masks(
    output_dir: Path,
    graph_dir: Path,
    constellation_name: str,
    groundstations_name: str,
    terminals_file: Path,
    country: str,
    flow_time_s: int,
    beam_policy: str,
    ku_band_capacity_gbps: float,
    *,
    mask_mode: str,
    duration_s: int = 15,
    update_interval_ms: int = 1000,
    demands_dir: Path | None = None,
) -> Path:
    output_dir = resolve_path(output_dir)
    graph_dir = resolve_path(graph_dir)
    terminals_path = resolve_terminals_path(terminals_file)
    validate_country(country)
    demands_base = resolve_path(demands_dir) if demands_dir is not None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if beam_policy not in {"greedy-uncoordinated", "greedy-coordinated"}:
        raise ValueError("Beam policy must match CosmoSim beam mapping choices.")
    if mask_mode not in {"on", "off"}:
        raise ValueError("mask_mode must be 'on' or 'off'.")

    constellation_path = constellation_dir(constellation_name)
    groundstations_file = groundstations_path(groundstations_name)

    ground_stations = read_ground_stations_extended(str(groundstations_file))
    tles = read_tles(str(constellation_path / "tles.txt"))
    satellites = tles["satellites"]

    description = exputil.PropertiesConfig(str(constellation_path / "description.txt"))
    num_orbits = json.loads(description.get_property_or_fail("num_orbits"))
    num_sats_per_orbit = json.loads(description.get_property_or_fail("num_sats_per_orbit"))
    total_sats = sum(o * s for o, s in zip(num_orbits, num_sats_per_orbit))
    if total_sats != len(satellites):
        raise ValueError(
            f"Constellation metadata inconsistent: description lists {total_sats} satellites, "
            f"but TLE contains {len(satellites)} entries."
        )

    if duration_s <= 0:
        raise ValueError("Duration must be positive.")
    if update_interval_ms <= 0:
        raise ValueError("Update interval must be positive.")

    start_ns = flow_time_s * 1_000_000_000
    end_ns = (flow_time_s + duration_s) * 1_000_000_000
    interval_ns = update_interval_ms * 1_000_000

    demands_path = demands_base / "demands.txt"
    demands = load_demands(demands_path)

    flow_path = output_dir / f"mask_{mask_mode}_{flow_time_s}.txt"
    flow_path.write_text("")

    gs_base_index = len(satellites)
    ut_base_index = len(satellites) + len(ground_stations)
    masked_nodes = apply_mask(ground_stations, gs_base_index, mask_mode, country)

    subgraph_nodes = [node for node in range(len(satellites) + len(ground_stations)) if node not in masked_nodes]

    for timestamp in range(start_ns, end_ns, interval_ns):
        graph_path = graph_dir / f"graph_{timestamp}.txt"
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph snapshot not found: {graph_path}")
        with graph_path.open("rb") as fh:
            graph_data = pickle.load(fh)

        di_graph = nx.DiGraph(graph_data.subgraph(subgraph_nodes))

        ground_station_capacity_used: Dict[int, float] = {idx + gs_base_index: 0.0 for idx in range(len(ground_stations))}
        sat_capacity: Dict[int, float] = {sat: 0.0 for sat in range(len(satellites))}

        for sat in np.nonzero(demands)[0]:
            predecessors = list(di_graph.predecessors(int(sat)))
            gs_neighbors = [
                node
                for node in predecessors
                if isinstance(node, int) and node >= gs_base_index and node < ut_base_index
            ]
            if not gs_neighbors:
                continue
            gs_neighbors.sort(key=lambda neighbor: ground_station_capacity_used.get(neighbor, 0.0))
            for neighbor in gs_neighbors:
                if neighbor in masked_nodes:
                    continue
                if (
                    sat_capacity[int(sat)] >= global_vars.ground_station_sat_capacity
                    or ground_station_capacity_used[neighbor] >= global_vars.ground_station_gsl_capacity
                ):
                    continue
                ka_beams_required = min(
                    math.floor(demands[int(sat)] / global_vars.ka_beam_capacity),
                    8 - int(sat_capacity[int(sat)] // global_vars.ka_beam_capacity),
                )
                if ka_beams_required <= 0:
                    continue
                gs_capacity = min(
                    global_vars.ground_station_gsl_capacity - ground_station_capacity_used[neighbor],
                    ka_beams_required * global_vars.ka_beam_capacity,
                )
                gs_capacity = (gs_capacity // global_vars.ka_beam_capacity) * global_vars.ka_beam_capacity
                if gs_capacity <= 0:
                    continue
                di_graph[neighbor][int(sat)]["capacity"] += int(gs_capacity)
                ground_station_capacity_used[neighbor] += gs_capacity
                sat_capacity[int(sat)] += gs_capacity

        for sat in range(len(satellites)):
            if sat_capacity[sat] >= global_vars.sat_gs_max_capacity:
                continue
            predecessors = list(di_graph.predecessors(sat))
            gs_neighbors = [
                node
                for node in predecessors
                if isinstance(node, int) and node >= gs_base_index and node < ut_base_index and node not in masked_nodes
            ]
            if not gs_neighbors:
                continue
            gs_neighbors.sort(key=lambda neighbor: ground_station_capacity_used.get(neighbor, 0.0))
            for neighbor in gs_neighbors:
                if sat_capacity[sat] >= global_vars.sat_gs_max_capacity:
                    break
                extra = min(
                    global_vars.sat_gs_max_capacity - sat_capacity[sat],
                    global_vars.ground_station_gsl_capacity - ground_station_capacity_used[neighbor],
                )
                if extra <= 0:
                    continue
                di_graph[neighbor][sat]["capacity"] += int(extra)
                sat_capacity[sat] += extra
                ground_station_capacity_used[neighbor] += extra

        for sat in range(len(satellites)):
            predecessors = list(di_graph.predecessors(sat))
            for neighbor in predecessors:
                if isinstance(neighbor, int) and neighbor >= gs_base_index and neighbor < ut_base_index:
                    if neighbor in masked_nodes or di_graph[neighbor][sat].get("capacity", 0) == 0:
                        if di_graph.has_edge(neighbor, sat):
                            di_graph.remove_edge(neighbor, sat)

        if math.isclose(ku_band_capacity_gbps, 2.5, rel_tol=1e-3):
            for sat in range(len(satellites)):
                isl_neighbors = [
                    node for node in di_graph.predecessors(sat) if isinstance(node, int) and node < gs_base_index
                ]
                for neighbor in isl_neighbors:
                    di_graph[neighbor][sat]["capacity"] = global_vars.isl_capacity * 2

        source = "S"
        sink = "T"
        di_graph.add_node(source)
        di_graph.add_node(sink)
        for gs in ground_stations:
            node_id = len(satellites) + gs["gid"]
            if node_id in masked_nodes:
                continue
            di_graph.add_edge(sink, node_id, capacity=global_vars.ground_station_gsl_capacity, weight=0)

        capacity, flow_dict = max_flow_capacities(demands, di_graph, source, sink, len(satellites))
        remove_dummy_nodes(di_graph, len(satellites))

        with flow_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{timestamp},{capacity}\n")

    return flow_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate capacities with ground-station masks applied.")
    parser.add_argument("output_dir", type=Path, help="Directory to store capacity time-series.")
    parser.add_argument("graph_dir", type=Path, help="Directory containing graph_*.txt snapshots.")
    parser.add_argument("constellation", help="Constellation name under constellation_configurations/configs.")
    parser.add_argument("groundstations", help="Ground station file prefix under inputs/groundstations.")
    parser.add_argument(
        "cells_file",
        type=Path,
        help="Terminal allocation file (relative to repo root or terminal_deployment/terminals).",
    )
    parser.add_argument("country", help="Country name (matches inputs/cells/<country>.txt).")
    parser.add_argument("flow_time_s", type=int, help="Simulation time (seconds) for the snapshot.")
    parser.add_argument(
        "beam_policy",
        choices=["greedy-uncoordinated", "greedy-coordinated"],
        help="Beam-mapping policy used when generating demands.",
    )
    parser.add_argument(
        "ku_band_capacity",
        type=float,
        help="Available KU-band capacity per beam in Gbps (e.g., 1.28).",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["on", "off"],
        default="on",
        help="Apply mask to disable non-designated ground stations ('on') or only the designated ones ('off').",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=15,
        help="Duration (seconds) to cover starting from flow_time_s. Default: 15s.",
    )
    parser.add_argument(
        "--update-interval-ms",
        type=int,
        default=1000,
        help="Graph sampling interval in milliseconds. Default: 1000ms.",
    )
    parser.add_argument(
        "--demands-dir",
        type=Path,
        help="Directory containing demands.txt (defaults to output_dir).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    flow_path = generate_capacities_masks(
        args.output_dir,
        args.graph_dir,
        args.constellation,
        args.groundstations,
        args.cells_file,
        args.country,
        args.flow_time_s,
        args.beam_policy,
        args.ku_band_capacity,
        mask_mode=args.mask_mode,
        duration_s=args.duration,
        update_interval_ms=args.update_interval_ms,
        demands_dir=args.demands_dir,
    )
    print(f"Wrote masked capacity samples to {flow_path}")


if __name__ == "__main__":
    main()
