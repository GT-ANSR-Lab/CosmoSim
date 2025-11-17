#!/usr/bin/env python3
"""Batch runner for workflows/generate_capacities_competing_traffic.py."""

from __future__ import annotations

import contextlib
import multiprocessing
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TextIO

import exputil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = PROJECT_ROOT / "workflows" / "generate_capacities_competing_traffic.py"
DATA_ROOT = PROJECT_ROOT / "data"
DEMANDS_ROOT = DATA_ROOT
LOG_ROOT = DATA_ROOT / "command_logs"
RUN_LOG_ROOT = LOG_ROOT / "run_generate_capacities_competing"
GRAPHS_ROOT = PROJECT_ROOT / "graph_generation" / "graphs"
CONSTELLATION_CONFIG_ROOT = PROJECT_ROOT / "constellation_configurations" / "configs"
TERMINAL_ROOT = PROJECT_ROOT / "terminal_deployment" / "terminals"

MAX_PARALLEL = max(1, multiprocessing.cpu_count() - 1)
DURATION_S = 15
DEFAULT_FLOW_TIMES = list(range(0, DURATION_S, DURATION_S)) or [0]
GROUNDSTATIONS = ["ground_stations_starlink"]
CONSTELLATIONS = ["starlink_5shells", "starlink_double", "starlink_all"]
KU_BAND_CAPACITIES = [0.956, 1.28, 2.5]
DEFAULT_KU_CAPACITY = 1.28
ROUTING_POLICIES = ["max_flow", "hot_potato"]
PRIORITIES = ["emergency", "incumbent"]
INCUMBENT_DEMANDS = [0.05, 0.1]
UPDATE_INTERVAL_MS = 1000

COUNTRIES = ["southafrica", "ghana", "tonga", "lithuania", "britain", "haiti"]
POPULATIONS: Dict[str, Sequence[int]] = {
    "tonga": [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000],
    "ghana": [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000],
    "southafrica": [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000],
    "lithuania": [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000],
    "britain": [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000],
    "haiti": [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000],
}
BEAM_POLICIES_PRIMARY = ["greedy-uncoordinated", "greedy-coordinated"]
BEAM_POLICIES_VARIANT = ["greedy-coordinated"]
GCB_CAPS = {
    "southafrica": [1000, 10000, 100000],
    "ghana": [1000, 10000, 100000],
    "tonga": [1000, 10000],
    "lithuania": [1000, 10000, 100000],
    "britain": [1000, 10000, 100000],
    "haiti": [1000, 10000, 100000],
}


@dataclass(frozen=True)
class DistributionCase:
    label: str
    beam_policies: Sequence[str]
    requires_gcb_cap: bool = False
    ku_band_capacities: Sequence[float] = (DEFAULT_KU_CAPACITY,)


DISTRIBUTION_CASES: Sequence[DistributionCase] = [
    DistributionCase(label="uniform", beam_policies=tuple(BEAM_POLICIES_PRIMARY)),
    DistributionCase(label="gcb_no_cap", beam_policies=tuple(BEAM_POLICIES_PRIMARY)),
    DistributionCase(
        label="uniform",
        beam_policies=tuple(BEAM_POLICIES_VARIANT),
        ku_band_capacities=tuple(KU_BAND_CAPACITIES),
    ),
    DistributionCase(
        label="gcb_cap",
        beam_policies=tuple(BEAM_POLICIES_VARIANT),
        requires_gcb_cap=True,
        ku_band_capacities=tuple(KU_BAND_CAPACITIES),
    ),
]


@dataclass
class CommandSpec:
    command: str
    description: str


class _StreamTee:
    def __init__(self, primary: TextIO, secondary: TextIO):
        self.primary = primary
        self.secondary = secondary

    def write(self, data: str) -> None:
        self.primary.write(data)
        self.secondary.write(data)

    def flush(self) -> None:
        self.primary.flush()
        self.secondary.flush()


@contextlib.contextmanager
def tee_output(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _StreamTee(original_stdout, log_file)
        sys.stderr = _StreamTee(original_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def ensure_dirs() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_LOG_ROOT.mkdir(parents=True, exist_ok=True)


def float_variants(value: float) -> List[str]:
    return list({f"{value}", f"{value:.2f}", f"{value:.3f}", f"{value:.1f}"})


def scenario_identifier(
    constellation: str,
    groundstations: str,
    country: str,
    pop: int,
    ut_dist: str,
    gcb_cap: Optional[int],
    ku_band_capacity: Optional[float],
) -> str:
    """Return the canonical scenario_id slug used across CosmoSim outputs."""
    parts: List[str] = [
        constellation,
        groundstations,
        "cells",
        country,
        "0",
        str(pop),
        ut_dist,
    ]
    if gcb_cap is not None:
        parts.append(str(gcb_cap))
    if ku_band_capacity is not None:
        parts.append(str(ku_band_capacity))
    return "_".join(parts)


def resolve_cells_file(
    scenario_id: str,
    country: str,
    pop: int,
    ut_dist: str,
    gcb_cap: Optional[int],
    ku_band_capacity: Optional[float],
) -> Optional[Path]:
    candidates: List[Path] = [CONSTELLATION_CONFIG_ROOT / scenario_id / "cells.txt"]
    base = f"cells_{country}_0_{pop}_{ut_dist}"
    if gcb_cap is not None:
        base += f"_{gcb_cap}"

    if ku_band_capacity is not None:
        for variant in float_variants(ku_band_capacity):
            candidates.append(TERMINAL_ROOT / f"{scenario_id}_{variant}.txt")
            candidates.append(TERMINAL_ROOT / f"{base}_{variant}.txt")
    candidates.append(TERMINAL_ROOT / f"{scenario_id}.txt")
    candidates.append(TERMINAL_ROOT / f"{base}.txt")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_graph_dir(
    constellation: str,
    groundstations: str,
    country: str,
) -> Optional[Path]:
    graph_label = f"{constellation}_{groundstations}_cells_{country}_0"
    candidates = [
        GRAPHS_ROOT / graph_label / "1000ms",
        GRAPHS_ROOT / graph_label,
        GRAPHS_ROOT / constellation / country / "1000ms",
        GRAPHS_ROOT / constellation / country,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def demands_dir_for(scenario_label: str, flow_time: int, beam_policy: str) -> Path:
    return DEMANDS_ROOT / f"{scenario_label}_{beam_policy}"


def build_command(
    constellation: str,
    groundstations: str,
    country: str,
    graph_dir: Path,
    cells_file: Path,
    flow_time: int,
    beam_policy: str,
    ku_band_capacity: float,
    scenario_label: str,
    demands_dir: Path,
    routing_policy: str,
    priority: str,
    incumbent_demand: float,
) -> CommandSpec:
    output_dir = demands_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = (
        LOG_ROOT
        / f"{scenario_label}_t{flow_time}_{beam_policy}_{routing_policy}_{priority}_{incumbent_demand}.log"
    )

    args = [
        "python",
        str(WORKFLOW),
        str(output_dir),
        str(graph_dir),
        constellation,
        groundstations,
        str(cells_file),
        country,
        str(flow_time),
        beam_policy,
        str(ku_band_capacity),
        "--routing",
        routing_policy,
        "--priority",
        priority,
        "--incumbent-demand",
        f"{incumbent_demand}",
        "--duration",
        str(DURATION_S),
        "--update-interval-ms",
        str(UPDATE_INTERVAL_MS),
        "--demands-dir",
        str(demands_dir),
    ]
    cmd = f"cd {shlex.quote(str(PROJECT_ROOT))}; " + " ".join(shlex.quote(arg) for arg in args)
    cmd += f" > {shlex.quote(str(log_path))} 2>&1"
    description = (
        f"{scenario_label} (t={flow_time}, beam={beam_policy}, route={routing_policy}, "
        f"priority={priority}, inc={incumbent_demand})"
    )
    return CommandSpec(command=cmd, description=description)


def _enqueue_cases(
    constellation: str,
    groundstations: str,
    country: str,
    population: int,
    ut_dist: str,
    beam_policy: str,
    gcb_cap: Optional[int],
    ku_band_capacity: float,
    specs: List[CommandSpec],
) -> None:
    graph_dir = resolve_graph_dir(constellation, groundstations, country)
    if graph_dir is None:
        print(f"[skip] Missing graphs for {constellation}/{country}")
        return

    include_ku_in_label = gcb_cap is not None or ut_dist.startswith("gcb")
    ku_label_value: Optional[float] = ku_band_capacity if include_ku_in_label else None

    scenario_id = scenario_identifier(
        constellation,
        groundstations,
        country,
        population,
        ut_dist,
        gcb_cap=gcb_cap,
        ku_band_capacity=ku_label_value,
    )
    cells_path = resolve_cells_file(
        scenario_id,
        country,
        population,
        ut_dist,
        gcb_cap=gcb_cap,
        ku_band_capacity=ku_label_value,
    )
    if cells_path is None:
        print(f"[skip] Missing cells file for {scenario_id}")
        return

    for flow_time in DEFAULT_FLOW_TIMES:
        demands_dir = demands_dir_for(scenario_id, flow_time, beam_policy)
        demands_file = demands_dir / "demands.txt"
        if not demands_file.exists():
            print(f"[skip] Missing demands at {demands_file} -- run run_generate_flows first")
            continue
        for routing in ROUTING_POLICIES:
            for priority in PRIORITIES:
                for incumbent_demand in INCUMBENT_DEMANDS:
                    specs.append(
                        build_command(
                            constellation,
                            groundstations,
                            country,
                            graph_dir,
                            cells_path,
                            flow_time,
                            beam_policy,
                            ku_band_capacity,
                            scenario_id,
                            demands_dir,
                            routing,
                            priority,
                            incumbent_demand,
                        )
                    )


def enqueue_all_cases() -> List[CommandSpec]:
    specs: List[CommandSpec] = []
    for constellation in CONSTELLATIONS:
        for groundstations in GROUNDSTATIONS:
            for country in COUNTRIES:
                pop_list = POPULATIONS.get(country, [])
                available_caps = GCB_CAPS.get(country, [])
                for pop in pop_list:
                    for case in DISTRIBUTION_CASES:
                        gcb_caps = available_caps if case.requires_gcb_cap else (None,)
                        for gcb_cap in gcb_caps:
                            for beam_policy in case.beam_policies:
                                for ku_band_capacity in case.ku_band_capacities:
                                    _enqueue_cases(
                                        constellation,
                                        groundstations,
                                        country,
                                        pop,
                                        case.label,
                                        beam_policy,
                                        gcb_cap=gcb_cap,
                                        ku_band_capacity=ku_band_capacity,
                                        specs=specs,
                                    )
    return specs


def run_commands(commands: List[CommandSpec]) -> None:
    if not commands:
        print("No competing-traffic capacity scenarios to run. Ensure demands exist or adjust the configuration.")
        return
    shell = exputil.LocalShell()
    print(f"Running {len(commands)} commands with up to {MAX_PARALLEL} concurrent screens...")
    for idx, spec in enumerate(commands, 1):
        print(f"[{idx}/{len(commands)}] {spec.description}")
        shell.detached_exec(spec.command)
        while shell.count_screens() >= MAX_PARALLEL:
            time.sleep(2)
    print("Waiting for remaining jobs to finish...")
    while shell.count_screens() > 0:
        time.sleep(2)
    print("Done.")


def main() -> None:
    ensure_dirs()
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_log_path = RUN_LOG_ROOT / f"run_generate_capacities_competing_traffic_{timestamp}.log"
    with tee_output(run_log_path):
        print(f"[info] Runner log written to {run_log_path}")
        specs = enqueue_all_cases()
        run_commands(specs)


if __name__ == "__main__":
    main()
