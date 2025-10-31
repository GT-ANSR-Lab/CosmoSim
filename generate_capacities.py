"""
Batch driver for directed-capacity generation built on the CosmoSim helper modules.

The original tooling shipped these entry points in dedicated repositories.  This
version keeps the orchestration logic while adopting the new terminal deployment
strategy naming used by `terminal_deployment/generate_cell_allocations.py`.
"""

import os
import shlex
import time
from pathlib import Path
from typing import Iterable, Optional

import exputil

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR
DIGRAPHS_DIR = (PROJECT_ROOT / "data" / "digraphs").resolve()
DIGRAPH_DATA_DIR = (
    PROJECT_ROOT / "data" / "traffic_engineering" / "digraph_data_imc"
).resolve()
CONSTELLATION_OUTPUT_DIR = (PROJECT_ROOT / "constellation_configurations" / "gen_data").resolve()

MAX_NUM_PROCESSES = 400
UPDATE_INTERVAL_MS = 1000
CAPACITY_DURATION_S = 15

COUNTRIES = ["southafrica", "ghana", "tonga", "lithuania", "britain", "haiti"]
POPULATION_TARGETS = {
    "southafrica": [1_000, 5_000, 10_000, 50_000, 100_000, 200_000, 500_000],
    "ghana": [1_000, 5_000, 10_000, 50_000, 100_000, 200_000, 500_000],
    "tonga": [100, 500, 1_000, 2_000, 5_000, 10_000, 20_000],
    "lithuania": [1_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000],
    "britain": [1_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000],
    "haiti": [1_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000],
}

CONSTELLATIONS = ["starlink_current_5shells", "starlink_double_7shells", "starlink_all_13shells"]
ISL_VARIANTS = ["isls_three", "isls_plus_grid"]
BEAM_ALLOCATIONS = ["priority", "popwaterfill"]
ROUTING_POLICIES = ["max_flow", "hot_potato"]

KU_CAPACITIES = [0.956, 1.28, 2.5]
GCB_CAP_VALUES = [1_000, 10_000, 100_000]
TERMINAL_STRATEGIES = ("uniform", "population", "gcb_no_cap", "gcb_cap")


def format_float(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def build_constellation_prefix(constellation: str, isl: str, country: str) -> str:
    return f"{constellation}_{isl}_ground_stations_starlink_cells_{country}_0"


def build_constellation_id(
    prefix: str,
    population: int,
    strategy: str,
    *,
    ku_capacity: Optional[float] = None,
    cap_value: Optional[int] = None,
) -> str:
    if strategy in {"uniform", "population"}:
        if ku_capacity is None:
            raise ValueError("Uniform/population capacity sweeps require a Ku-band capacity.")
        return f"{prefix}_{population}_{strategy}_{format_float(ku_capacity)}"
    if strategy == "gcb_no_cap":
        if ku_capacity is None:
            raise ValueError("gcb_no_cap requires a Ku-band capacity value.")
        return f"{prefix}_{population}_gcb_no_cap_{format_float(ku_capacity)}"
    if strategy == "gcb_cap":
        if ku_capacity is None or cap_value is None:
            raise ValueError("gcb_cap requires both Ku-band capacity and cap value.")
        return f"{prefix}_{population}_gcb_cap_{cap_value}_{format_float(ku_capacity)}"
    raise ValueError(f"Unsupported terminal deployment strategy '{strategy}'.")


def iterate_time_windows(duration_s: int) -> Iterable[tuple[int, int]]:
    start_time = 0
    while start_time < duration_s:
        yield start_time, start_time + duration_s
        start_time += duration_s


def ensure_dependencies(constellation_id: str, prefix: str) -> tuple[Path, Path]:
    constellation_dir = CONSTELLATION_OUTPUT_DIR / constellation_id
    if not constellation_dir.exists():
        raise FileNotFoundError(
            f"Constellation output not found: {constellation_dir}. "
            "Generate configurations in `constellation_configurations/gen_data` first."
        )

    graph_dir = DIGRAPHS_DIR / prefix / f"{UPDATE_INTERVAL_MS}ms"
    if not graph_dir.exists():
        raise FileNotFoundError(
            f"Directed graph snapshots not found: {graph_dir}. "
            "Run `graph_generation/generate_digraphs.py` before generating capacities."
        )

    return constellation_dir, graph_dir


def enqueue_capacity_command(
    commands: list[str],
    constellation_id: str,
    prefix: str,
    start_time: int,
    end_time: int,
    beam_allocation: str,
    routing_policy: str,
    ku_capacity: float,
) -> None:
    constellation_dir, graph_dir = ensure_dependencies(constellation_id, prefix)

    log_file = (
        DIGRAPH_DATA_DIR
        / "command_logs"
        / (
            f"capacities_{constellation_id}_t{start_time}_"
            f"{beam_allocation}_{routing_policy}_{format_float(ku_capacity)}.log"
        )
    )

    command = (
        f"cd {shlex.quote(str(PROJECT_ROOT))}; "
        f"python -m traffic_engineering.utils.generate_capacities_digraph "
        f"{shlex.quote(str(DIGRAPH_DATA_DIR))} "
        f"{shlex.quote(str(graph_dir))} "
        f"{shlex.quote(str(constellation_dir))} "
        f"{UPDATE_INTERVAL_MS} {start_time} {end_time} "
        f"{routing_policy} {beam_allocation} {format_float(ku_capacity)} "
        f"> {shlex.quote(str(log_file))} 2>&1"
    )

    if command not in commands:
        commands.append(command)


def main() -> None:
    os.makedirs(DIGRAPH_DATA_DIR, exist_ok=True)
    (DIGRAPH_DATA_DIR / "command_logs").mkdir(exist_ok=True)

    commands_to_run: list[str] = []

    for constellation in CONSTELLATIONS:
        for isl in ISL_VARIANTS:
            for country in COUNTRIES:
                prefix = build_constellation_prefix(constellation, isl, country)
                populations = POPULATION_TARGETS[country]

                for population in populations:
                    for strategy in TERMINAL_STRATEGIES:
                        if strategy in {"uniform", "population"}:
                            for ku_capacity in KU_CAPACITIES:
                                constellation_id = build_constellation_id(
                                    prefix,
                                    population,
                                    strategy,
                                    ku_capacity=ku_capacity,
                                )
                                for start_time, end_time in iterate_time_windows(CAPACITY_DURATION_S):
                                    for beam_allocation in BEAM_ALLOCATIONS:
                                        for routing_policy in ROUTING_POLICIES:
                                            enqueue_capacity_command(
                                                commands_to_run,
                                                constellation_id,
                                                prefix,
                                                start_time,
                                                end_time,
                                                beam_allocation,
                                                routing_policy,
                                                ku_capacity,
                                            )
                            continue

                        if strategy == "gcb_no_cap":
                            for ku_capacity in KU_CAPACITIES:
                                constellation_id = build_constellation_id(
                                    prefix,
                                    population,
                                    strategy,
                                    ku_capacity=ku_capacity,
                                )
                                for start_time, end_time in iterate_time_windows(CAPACITY_DURATION_S):
                                    for beam_allocation in BEAM_ALLOCATIONS:
                                        for routing_policy in ROUTING_POLICIES:
                                            enqueue_capacity_command(
                                                commands_to_run,
                                                constellation_id,
                                                prefix,
                                                start_time,
                                                end_time,
                                                beam_allocation,
                                                routing_policy,
                                                ku_capacity,
                                            )
                            continue

                        # strategy == "gcb_cap"
                        for cap_value in GCB_CAP_VALUES:
                            for ku_capacity in KU_CAPACITIES:
                                constellation_id = build_constellation_id(
                                    prefix,
                                    population,
                                    strategy,
                                    cap_value=cap_value,
                                    ku_capacity=ku_capacity,
                                )
                                for start_time, end_time in iterate_time_windows(CAPACITY_DURATION_S):
                                    for beam_allocation in BEAM_ALLOCATIONS[1:]:
                                        for routing_policy in ROUTING_POLICIES:
                                            enqueue_capacity_command(
                                                commands_to_run,
                                                constellation_id,
                                                prefix,
                                                start_time,
                                                end_time,
                                                beam_allocation,
                                                routing_policy,
                                                ku_capacity,
                                            )

    if not commands_to_run:
        print("No commands scheduled. Check configuration filters.")
        return

    local_shell = exputil.LocalShell()
    print(f"Running commands (at most {MAX_NUM_PROCESSES} in parallel)...")
    for idx, command in enumerate(commands_to_run, start=1):
        print(f"Starting command {idx}/{len(commands_to_run)}: {command}")
        local_shell.detached_exec(command)
        while local_shell.count_screens() >= MAX_NUM_PROCESSES:
            time.sleep(2)

    print(f"Waiting completion of the last {MAX_NUM_PROCESSES}...")
    while local_shell.count_screens() > 0:
        time.sleep(2)
    print("Finished.")


if __name__ == "__main__":
    main()
