#!/usr/bin/env python3
"""Batch runner for workflows/generate_flows.py covering legacy scenario sets."""

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
WORKFLOW = PROJECT_ROOT / "workflows" / "generate_flows.py"
DATA_ROOT = PROJECT_ROOT / "data"
DEMANDS_ROOT = DATA_ROOT
LOG_ROOT = DATA_ROOT / "command_logs"
RUN_LOG_ROOT = LOG_ROOT / "run_generate_flows"
GRAPHS_ROOT = PROJECT_ROOT / "graph_generation" / "graphs"
CONSTELLATION_CONFIG_ROOT = PROJECT_ROOT / "constellation_configurations" / "configs"
TERMINAL_ROOT = PROJECT_ROOT / "terminal_deployment" / "terminals"
GROUNDSTATIONS_ROOT = PROJECT_ROOT / "inputs" / "groundstations"

MAX_PARALLEL = max(1, multiprocessing.cpu_count() - 1)
DURATION_S = 15
DEFAULT_FLOW_TIMES = list(range(0, DURATION_S, DURATION_S)) or [0]
GROUNDSTATIONS = ["ground_stations_starlink"]
CONSTELLATIONS = ["starlink_5shells", "starlink_double", "starlink_all"]
KU_BAND_CAPACITIES = [0.956, 1.28, 2.5]
DEFAULT_KU_CAPACITY = 1.28

COUNTRIES = ["southafrica", "ghana", "tonga", "lithuania", "britain", "haiti"]
POPULATIONS: Dict[str, Sequence[int]] = {
    "tonga": [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000],
    "ghana": [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000],
    "southafrica": [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000],
    "lithuania": [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000],
    "britain": [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000],
    "haiti": [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000],
}
BEAM_POLICIES = ["greedy-uncoordinated", "greedy-coordinated"]
GCB_CAPS = {
    "southafrica": [1000, 10000, 100000],
    "ghana": [1000, 10000, 100000],
    "tonga": [1000, 10000],
    "lithuania": [1000, 10000, 100000],
    "britain": [1000, 10000, 100000],
    "haiti": [1000, 10000, 100000],
}


@dataclass
class CommandSpec:
    command: str
    description: str
    log_path: Path


@dataclass(frozen=True)
class DistributionConfig:
    label: str
    requires_gcb_cap: bool = False
    include_ku_in_label: bool = False
    ku_band_capacities: Sequence[float] = (DEFAULT_KU_CAPACITY,)


DISTRIBUTIONS: Sequence[DistributionConfig] = [
    DistributionConfig(label="uniform"),
    DistributionConfig(label="population"),
    DistributionConfig(
        label="gcb_no_cap",
        include_ku_in_label=True,
        ku_band_capacities=tuple(KU_BAND_CAPACITIES),
    ),
    DistributionConfig(
        label="gcb_cap",
        requires_gcb_cap=True,
        include_ku_in_label=True,
        ku_band_capacities=tuple(KU_BAND_CAPACITIES),
    ),
]


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
    DEMANDS_ROOT.mkdir(parents=True, exist_ok=True)
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


def demands_output_dir(scenario_label: str, flow_time: int, beam_policy: str) -> Path:
    return (DEMANDS_ROOT / f"{scenario_label}_{beam_policy}").resolve()


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
) -> CommandSpec:
    output_dir = demands_output_dir(scenario_label, flow_time, beam_policy)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{scenario_label}_t{flow_time}_{beam_policy}.log"

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
    ]
    cmd = f"cd {shlex.quote(str(PROJECT_ROOT))}; " + " ".join(shlex.quote(arg) for arg in args)
    cmd += f" > {shlex.quote(str(log_path))} 2>&1"
    description = f"{scenario_label} (t={flow_time}, beam={beam_policy})"
    return CommandSpec(command=cmd, description=description, log_path=log_path)


def enqueue_cases() -> List[CommandSpec]:
    specs: List[CommandSpec] = []
    for constellation in CONSTELLATIONS:
        for groundstations in GROUNDSTATIONS:
            for country in COUNTRIES:
                graph_dir = resolve_graph_dir(constellation, groundstations, country)
                if graph_dir is None:
                    print(f"[skip] Missing graphs for {constellation}/{country}")
                    continue
                pop_list = POPULATIONS.get(country, [])
                for pop in pop_list:
                    for dist_config in DISTRIBUTIONS:
                        caps = GCB_CAPS.get(country, []) if dist_config.requires_gcb_cap else [None]
                        for gcb_cap in caps:
                            for ku_band_capacity in dist_config.ku_band_capacities:
                                scenario_ku = ku_band_capacity if dist_config.include_ku_in_label else None
                                scenario_id = scenario_identifier(
                                    constellation,
                                    groundstations,
                                    country,
                                    pop,
                                    dist_config.label,
                                    gcb_cap=gcb_cap,
                                    ku_band_capacity=scenario_ku,
                                )
                                cells_path = resolve_cells_file(
                                    scenario_id,
                                    country,
                                    pop,
                                    dist_config.label,
                                    gcb_cap=gcb_cap,
                                    ku_band_capacity=scenario_ku,
                                )
                                if cells_path is None:
                                    print(f"[skip] Missing cells file for {scenario_id}")
                                    continue
                                for beam_policy in BEAM_POLICIES:
                                    for flow_time in DEFAULT_FLOW_TIMES:
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
                                            )
                                        )
    print(f"[info] Enqueued {len(specs)} flow generation jobs.")
    return specs


def run_commands(commands: List[CommandSpec]) -> None:
    if not commands:
        print("No scenarios to run. Ensure prerequisite data exists or adjust the configuration.")
        return
    shell = exputil.LocalShell()
    print(f"Running {len(commands)} commands with up to {MAX_PARALLEL} concurrent screens...")
    for idx, spec in enumerate(commands, 1):
        print(f"[{idx}/{len(commands)}] {spec.description} -> {spec.log_path}")
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
    run_log_path = RUN_LOG_ROOT / f"run_generate_flows_{timestamp}.log"
    with tee_output(run_log_path):
        print(f"[info] Runner log written to {run_log_path}")
        specs = enqueue_cases()
        run_commands(specs)


if __name__ == "__main__":
    main()
