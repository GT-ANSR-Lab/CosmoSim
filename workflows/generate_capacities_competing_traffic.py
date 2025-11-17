#!/usr/bin/env python3
"""Generate capacity traces with incumbent/emergency traffic prioritization."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h3
import networkx as nx
import numpy as np
from networkx.algorithms.flow import max_flow_min_cost

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import utils.global_variables as global_vars
from traffic_engineering.hot_potato import hot_potato_modifications
from utils.cells import read_cells, read_cells_starlink
from utils.ground_stations import read_ground_stations_extended
from utils.tles import read_tles

CONSTELLATION_ROOT = PROJECT_ROOT / "constellation_configurations" / "configs"
GROUNDSTATION_ROOT = PROJECT_ROOT / "inputs" / "groundstations"
CELLS_ROOT = PROJECT_ROOT / "inputs" / "cells"
TERMINAL_ROOT = PROJECT_ROOT / "terminal_deployment" / "terminals"
STARLINK_CELLS_DEFAULT = PROJECT_ROOT / "inputs" / "cells" / "starlink_cells.txt"


def read_properties(path: Path) -> Dict[str, str]:
    properties: Dict[str, str] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            properties[key.strip()] = value.strip()
    if not properties:
        raise ValueError(f"No properties found in {path}")
    return properties


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


def load_starlink_cells(cells: Sequence[Dict[str, int]], starlink_file: Path) -> Sequence[str]:
    if not starlink_file.exists():
        raise FileNotFoundError(f"Starlink cells file not found: {starlink_file}")
    level3_cells = {h3.cell_to_parent(cell["cell"], 3) for cell in cells}
    starlink_cells = read_cells_starlink(str(starlink_file))
    filtered = [cell["cell"] for cell in starlink_cells if cell["cell"] not in level3_cells]
    return filtered


def assign_sat_capacity(graph: nx.DiGraph, demands: np.ndarray, start_node: str, num_satellites: int) -> nx.DiGraph:
    for sat in range(num_satellites):
        demand = float(demands[sat]) if sat < len(demands) else 0.0
        if demand <= 0:
            continue
        dummy = f"D{sat}"
        graph.add_edge(dummy, start_node, capacity=demand, weight=0)
        graph.add_edge(sat, dummy, capacity=demand, weight=0)
    return graph


def remove_dummy_nodes(graph: nx.DiGraph, num_satellites: int) -> None:
    for sat in range(num_satellites):
        dummy = f"D{sat}"
        if graph.has_node(dummy):
            graph.remove_node(dummy)


def emergency_flow(graph: nx.DiGraph, demands: np.ndarray, source: str, sink: str, num_satellites: int) -> Dict[str, Dict[str, float]]:
    assign_sat_capacity(graph, demands, source, num_satellites)
    return max_flow_min_cost(graph, sink, source)


def incumbent_flow(
    graph: nx.DiGraph,
    incumbent_satellites: Sequence[int],
    incumbent_demand: float,
    source: str,
    sink: str,
    num_satellites: int,
) -> Dict[str, Dict[str, float]]:
    demands = np.zeros(num_satellites)
    if incumbent_satellites:
        demands[np.array(incumbent_satellites, dtype=int)] = incumbent_demand
    assign_sat_capacity(graph, demands, source, num_satellites)
    return max_flow_min_cost(graph, sink, source)


def calculate_emergency_allocations(
    flow_dict: Dict[str, Dict[str, float]],
    demands: np.ndarray,
    source: str,
    num_satellites: int,
) -> List[float]:
    allocations = [-1.0] * num_satellites
    for sat in range(num_satellites):
        if sat < len(demands) and demands[sat] > 0:
            dummy = f"D{sat}"
            allocations[sat] = flow_dict.get(dummy, {}).get(source, 0.0) / demands[sat]
    return allocations


def calculate_incumbent_allocations(
    flow_dict: Dict[str, Dict[str, float]],
    source: str,
    incumbent_satellites: Sequence[int],
    incumbent_demand: float,
) -> Dict[int, float]:
    allocations: Dict[int, float] = {}
    for sat in incumbent_satellites:
        dummy = f"D{sat}"
        allocations[sat] = flow_dict.get(dummy, {}).get(source, 0.0) / incumbent_demand if incumbent_demand > 0 else 0.0
    return allocations


def apply_flow_consumption(graph: nx.DiGraph, flow_dict: Dict[str, Dict[str, float]]) -> None:
    for node_a, neighbors in flow_dict.items():
        for node_b, flow in neighbors.items():
            if graph.has_edge(node_a, node_b):
                graph[node_a][node_b]["capacity"] -= flow


def identify_incumbent_satellites(
    base_graph: nx.DiGraph,
    starlink_cells: Sequence[str],
    num_satellites: int,
) -> List[int]:
    starlink_set = set(starlink_cells)
    incumbents: List[int] = []
    for sat in range(num_satellites):
        if not base_graph.has_node(sat):
            continue
        for neighbor in base_graph.successors(sat):
            if isinstance(neighbor, str) and neighbor in starlink_set:
                incumbents.append(sat)
                break
    return incumbents


def generate_capacities_competing(
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
    priority: str,
    incumbent_demand_gbps: float,
    routing_policy: str = "max_flow",
    duration_s: int = 15,
    update_interval_ms: int = 1000,
    demands_dir: Path | None = None,
    starlink_cells_file: Path = STARLINK_CELLS_DEFAULT,
) -> Tuple[Path, Path, Path, Path]:
    output_dir = resolve_path(output_dir)
    graph_dir = resolve_path(graph_dir)
    terminals_path = resolve_terminals_path(terminals_file)
    validate_country(country)
    demands_base = resolve_path(demands_dir) if demands_dir is not None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if beam_policy not in {"greedy-uncoordinated", "greedy-coordinated"}:
        raise ValueError("Beam policy must match CosmoSim beam mapping choices.")
    if routing_policy not in {"max_flow", "hot_potato"}:
        raise ValueError("Routing policy must be 'max_flow' or 'hot_potato'.")
    if priority not in {"emergency", "incumbent"}:
        raise ValueError("Priority must be either 'emergency' or 'incumbent'.")

    constellation_path = constellation_dir(constellation_name)
    groundstations_file = groundstations_path(groundstations_name)

    ground_stations = read_ground_stations_extended(str(groundstations_file))
    cells = [cell for cell in read_cells(str(terminals_path)) if cell["num_terminals"] > 0]
    starlink_cells = load_starlink_cells(cells, resolve_path(starlink_cells_file))

    tles = read_tles(str(constellation_path / "tles.txt"))
    satellites = tles["satellites"]

    description = read_properties(constellation_path / "description.txt")
    num_orbits = json.loads(description["num_orbits"])
    num_sats_per_orbit = json.loads(description["num_sats_per_orbit"])
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

    incumbent_demand = incumbent_demand_gbps * global_vars.incumbent_demand_multiplier

    inc_tag = str(incumbent_demand_gbps).replace(".", "p")
    file_tag = f"{routing_policy}_{priority}_inc{inc_tag}_t{flow_time_s}"
    flow_path = output_dir / f"competing_flow_{file_tag}.txt"
    fulfillment_path = output_dir / f"competing_fulfillment_{file_tag}.txt"
    first_pass_path = output_dir / f"competing_first_pass_{file_tag}.json"
    second_pass_path = output_dir / f"competing_second_pass_{file_tag}.json"
    flow_path.write_text("")
    fulfillment_path.write_text("")

    gs_base_index = len(satellites)
    ut_base_index = len(satellites) + len(ground_stations)
    subgraph_nodes = list(range(len(satellites) + len(ground_stations)))

    for timestamp in range(start_ns, end_ns, interval_ns):
        graph_path = graph_dir / f"graph_{timestamp}.txt"
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph snapshot not found: {graph_path}")
        with graph_path.open("rb") as fh:
            graph_data = pickle.load(fh)

        di_graph = nx.DiGraph(graph_data.subgraph(subgraph_nodes))
        incumbents = identify_incumbent_satellites(di_graph, starlink_cells, len(satellites))

        gs_capacity_used: Dict[int, float] = {idx + gs_base_index: 0.0 for idx in range(len(ground_stations))}
        sat_capacity: Dict[int, float] = {sat: 0.0 for sat in range(len(satellites))}

        for sat in np.nonzero(demands)[0]:
            if not di_graph.has_node(int(sat)):
                continue
            predecessors = list(di_graph.predecessors(int(sat)))
            gs_neighbors = [
                node
                for node in predecessors
                if isinstance(node, int) and gs_base_index <= node < ut_base_index
            ]
            if not gs_neighbors:
                continue
            gs_neighbors.sort(key=lambda neighbor: gs_capacity_used[neighbor])
            for neighbor in gs_neighbors:
                if (
                    sat_capacity[int(sat)] >= global_vars.ground_station_sat_capacity
                    or gs_capacity_used[neighbor] >= global_vars.ground_station_gsl_capacity
                ):
                    continue
                ka_beams_required = min(
                    math.floor(demands[int(sat)] / global_vars.ka_beam_capacity),
                    8 - int(sat_capacity[int(sat)] // global_vars.ka_beam_capacity),
                )
                if ka_beams_required <= 0:
                    continue
                gs_capacity = min(
                    global_vars.ground_station_gsl_capacity - gs_capacity_used[neighbor],
                    ka_beams_required * global_vars.ka_beam_capacity,
                )
                gs_capacity = (gs_capacity // global_vars.ka_beam_capacity) * global_vars.ka_beam_capacity
                if gs_capacity <= 0:
                    continue
                di_graph[neighbor][int(sat)]["capacity"] += int(gs_capacity)
                gs_capacity_used[neighbor] += gs_capacity
                sat_capacity[int(sat)] += gs_capacity

        shell_priority_order = list(range(1584, len(satellites))) + list(range(1584))
        for sat in shell_priority_order:
            if not di_graph.has_node(sat):
                continue
            if sat_capacity[sat] >= global_vars.sat_gs_max_capacity:
                continue
            predecessors = list(di_graph.predecessors(sat))
            gs_neighbors = [
                node
                for node in predecessors
                if isinstance(node, int) and gs_base_index <= node < ut_base_index
            ]
            if not gs_neighbors:
                continue
            gs_neighbors.sort(key=lambda neighbor: gs_capacity_used[neighbor])
            for neighbor in gs_neighbors:
                if sat_capacity[sat] >= global_vars.sat_gs_max_capacity:
                    break
                extra = min(
                    global_vars.sat_gs_max_capacity - sat_capacity[sat],
                    global_vars.ground_station_gsl_capacity - gs_capacity_used[neighbor],
                )
                if extra <= 0:
                    continue
                di_graph[neighbor][sat]["capacity"] += int(extra)
                sat_capacity[sat] += extra
                gs_capacity_used[neighbor] += extra

        for sat in range(len(satellites)):
            if not di_graph.has_node(sat):
                continue
            predecessors = list(di_graph.predecessors(sat))
            for neighbor in predecessors:
                if isinstance(neighbor, int) and gs_base_index <= neighbor < ut_base_index:
                    if di_graph.has_edge(neighbor, sat) and di_graph[neighbor][sat].get("capacity", 0) == 0:
                        di_graph.remove_edge(neighbor, sat)

        if math.isclose(ku_band_capacity_gbps, 2.5, rel_tol=1e-3):
            for sat in range(len(satellites)):
                if not di_graph.has_node(sat):
                    continue
                isl_neighbors = [
                    node for node in di_graph.predecessors(sat) if isinstance(node, int) and node < gs_base_index
                ]
                for neighbor in isl_neighbors:
                    di_graph[neighbor][sat]["capacity"] = global_vars.isl_capacity * 2

        routing_demands = np.zeros(len(satellites))
        routing_demands[: min(len(demands), len(satellites))] = demands[: len(routing_demands)]
        if incumbent_demand > 0:
            for sat in incumbents:
                routing_demands[sat] = max(routing_demands[sat], incumbent_demand)
        if routing_policy == "hot_potato":
            hot_potato_modifications(di_graph, routing_demands, len(ground_stations), len(satellites))

        source = "S"
        sink = "T"
        di_graph.add_node(source)
        di_graph.add_node(sink)
        for gs in ground_stations:
            node_id = len(satellites) + gs["gid"]
            di_graph.add_edge(sink, node_id, capacity=global_vars.ground_station_gsl_capacity, weight=0)

        order = [priority, "incumbent" if priority == "emergency" else "emergency"]
        max_flow_emergency = 0.0
        max_flow_incumbent = 0.0
        emergency_alloc = [-1.0] * len(satellites)
        incumbent_alloc = [-1.0] * len(satellites)
        first_pass_flows: Dict[str, Dict[str, float]] = {}
        second_pass_flows: Dict[str, Dict[str, float]] = {}

        for pass_idx, pass_type in enumerate(order):
            working_graph = di_graph
            if pass_type == "emergency":
                flow_dict = emergency_flow(working_graph, demands, source, sink, len(satellites))
                emergency_alloc = calculate_emergency_allocations(flow_dict, demands, source, len(satellites))
                total = sum(flow_dict[sink].values()) if sink in flow_dict else 0.0
                if pass_idx == 0:
                    first_pass_flows = flow_dict
                else:
                    second_pass_flows = flow_dict
                if pass_type == priority:
                    first_pass_flows = flow_dict
                else:
                    second_pass_flows = flow_dict
                max_flow_emergency = total
            else:
                flow_dict = incumbent_flow(working_graph, incumbents, incumbent_demand, source, sink, len(satellites))
                allocs = calculate_incumbent_allocations(flow_dict, source, incumbents, incumbent_demand)
                for sat, ratio in allocs.items():
                    incumbent_alloc[sat] = ratio
                total = sum(flow_dict[sink].values()) if sink in flow_dict else 0.0
                if pass_idx == 0:
                    first_pass_flows = flow_dict
                else:
                    second_pass_flows = flow_dict
                max_flow_incumbent = total

            apply_flow_consumption(di_graph, flow_dict)
            remove_dummy_nodes(di_graph, len(satellites))

        with first_pass_path.open("w", encoding="utf-8") as fh:
            json.dump(first_pass_flows, fh)
        with second_pass_path.open("w", encoding="utf-8") as fh:
            json.dump(second_pass_flows, fh)

        with flow_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{timestamp},{max_flow_emergency},{max_flow_incumbent}\n")

        with fulfillment_path.open("a", encoding="utf-8") as fh:
            for sat in range(len(satellites)):
                demand_value = demands[sat] if sat < len(demands) else 0.0
                fh.write(
                    f"{timestamp},{sat},{demand_value},{emergency_alloc[sat]},{incumbent_alloc[sat]}\n"
                )

    return flow_path, fulfillment_path, first_pass_path, second_pass_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate capacities for emergency/incumbent traffic priorities."
    )
    parser.add_argument("output_dir", type=Path, help="Directory to store capacity outputs.")
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
        "--priority",
        choices=["emergency", "incumbent"],
        default="emergency",
        help="Which traffic class receives service first.",
    )
    parser.add_argument(
        "--incumbent-demand",
        type=float,
        default=0.1,
        help="Incumbent demand per satellite in Gbps (before multiplier).",
    )
    parser.add_argument(
        "--routing",
        choices=["max_flow", "hot_potato"],
        default="max_flow",
        help="Routing policy for both traffic classes (default: max_flow).",
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
    parser.add_argument(
        "--starlink-cells",
        type=Path,
        default=STARLINK_CELLS_DEFAULT,
        help="Path to starlink_cells.txt for incumbent detection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    flow_path, fulfillment_path, first_pass_path, second_pass_path = generate_capacities_competing(
        args.output_dir,
        args.graph_dir,
        args.constellation,
        args.groundstations,
        args.cells_file,
        args.country,
        args.flow_time_s,
        args.beam_policy,
        args.ku_band_capacity,
        priority=args.priority,
        incumbent_demand_gbps=args.incumbent_demand,
        routing_policy=args.routing,
        duration_s=args.duration,
        update_interval_ms=args.update_interval_ms,
        demands_dir=args.demands_dir,
        starlink_cells_file=args.starlink_cells,
    )
    print(f"Wrote capacity summary to {flow_path}")
    print(f"Wrote fulfillment log to {fulfillment_path}")
    print(f"Stored first-pass flows in {first_pass_path}")
    print(f"Stored second-pass flows in {second_pass_path}")


if __name__ == "__main__":
    main()
