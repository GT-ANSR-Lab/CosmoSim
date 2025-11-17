#!/usr/bin/env python3
"""Batch runner for graph_generation/generate_graphs.py across study scenarios."""

from __future__ import annotations

import contextlib
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, TextIO

import exputil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_SCRIPT = PROJECT_ROOT / "graph_generation" / "generate_graphs.py"
GRAPHS_ROOT = PROJECT_ROOT / "graph_generation" / "graphs"
CONSTELLATION_ROOT = PROJECT_ROOT / "constellation_configurations" / "configs"
GROUNDSTATION_ROOT = PROJECT_ROOT / "inputs" / "groundstations"
CELLS_ROOT = PROJECT_ROOT / "inputs" / "cells"
LOG_ROOT = PROJECT_ROOT / "data" / "command_logs"
RUN_LOG_ROOT = LOG_ROOT / "run_generate_graphs"
JOB_LOG_ROOT = LOG_ROOT / "graph_generation_jobs"

def _physical_core_count() -> int | None:
    try:
        output = subprocess.check_output(["lscpu"], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    threads_per_core = None
    total_cpus = None
    for line in output.splitlines():
        if line.startswith("CPU(s):"):
            try:
                total_cpus = int(line.split(":", 1)[1].strip())
            except ValueError:
                total_cpus = None
        elif line.startswith("Thread(s) per core"):
            try:
                threads_per_core = int(line.split(":", 1)[1].strip())
            except ValueError:
                threads_per_core = None
    if total_cpus and threads_per_core:
        return max(total_cpus // threads_per_core, 1)
    return None


logical_cpus = os.cpu_count() or 4
physical_cpus = _physical_core_count() or logical_cpus
DEFAULT_MAX_PARALLEL = max(physical_cpus - 2, 1)
MAX_PARALLEL = int(os.environ.get("GRAPH_RUNNER_MAX_PARALLEL", DEFAULT_MAX_PARALLEL))
START_S = 0
TOTAL_DURATION_S = 15
CHUNK_DURATION_S = 1

CONSTELLATIONS: Sequence[str] = ["starlink_5shells", "starlink_double", "starlink_all"]
GROUNDSTATIONS: Sequence[str] = ["ground_stations_starlink"]
COUNTRIES: Sequence[str] = ["southafrica", "ghana", "tonga", "lithuania", "britain", "haiti"]


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


@dataclass
class CommandSpec:
    command: str
    description: str
    log_path: Path


def ensure_dirs() -> None:
    GRAPHS_ROOT.mkdir(parents=True, exist_ok=True)
    JOB_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_LOG_ROOT.mkdir(parents=True, exist_ok=True)


def constellation_ready(constellation: str) -> bool:
    return (CONSTELLATION_ROOT / constellation).exists()


def groundstations_ready(name: str) -> bool:
    return any(
        (GROUNDSTATION_ROOT / candidate).exists()
        for candidate in (name, f"{name}.txt", f"{name}.csv")
    )


def cells_ready(country: str) -> bool:
    return (CELLS_ROOT / f"{country}.txt").exists()


def scenario_label(constellation: str, groundstations: str, country: str) -> str:
    return f"{constellation}_{groundstations}_cells_{country}_0"


def build_commands(constellation: str, groundstations: str, country: str) -> List[CommandSpec]:
    scenario = scenario_label(constellation, groundstations, country)
    chunk_specs: List[CommandSpec] = []
    for start_s in range(START_S, START_S + TOTAL_DURATION_S, CHUNK_DURATION_S):
        log_path = JOB_LOG_ROOT / f"{scenario}_t{start_s}.log"
        args = [
            "python",
            str(GRAPH_SCRIPT),
            "--constellation-config",
            constellation,
            "--groundstations",
            groundstations,
            "--country",
            country,
            "--graphs-dir",
            str(GRAPHS_ROOT),
            "--start",
            str(start_s),
            "--duration",
            str(CHUNK_DURATION_S),
            "--constellation-root",
            str(CONSTELLATION_ROOT),
            "--log-file",
            str(log_path),
        ]
        cmd = f"cd {shlex.quote(str(PROJECT_ROOT))}; " + " ".join(shlex.quote(arg) for arg in args)
        description = f"{scenario} (t={start_s}s duration={CHUNK_DURATION_S}s)"
        chunk_specs.append(CommandSpec(command=cmd, description=description, log_path=log_path))
    return chunk_specs


def enqueue_jobs() -> List[CommandSpec]:
    specs: List[CommandSpec] = []
    for constellation in CONSTELLATIONS:
        if not constellation_ready(constellation):
            print(f"[skip] Missing constellation data for {constellation}")
            continue
        for groundstations in GROUNDSTATIONS:
            if not groundstations_ready(groundstations):
                print(f"[skip] Missing groundstation catalogue '{groundstations}'")
                continue
            for country in COUNTRIES:
                if not cells_ready(country):
                    print(f"[skip] Missing cells file for country '{country}'")
                    continue
                specs.extend(build_commands(constellation, groundstations, country))
    return specs


def run_commands(commands: List[CommandSpec]) -> None:
    if not commands:
        print("No graph generation jobs scheduled. Provide inputs or adjust scenario lists.")
        return
    shell = exputil.LocalShell()
    total = len(commands)
    print(f"Running {total} graph generation jobs with up to {MAX_PARALLEL} concurrent screens...")
    for idx, spec in enumerate(commands, 1):
        print(f"[{idx}/{total}] {spec.description} -> {spec.log_path}")
        shell.detached_exec(spec.command)
        while shell.count_screens() >= MAX_PARALLEL:
            time.sleep(2)
    print("Waiting for remaining jobs to finish...")
    while shell.count_screens() > 0:
        time.sleep(2)
    print("All graph generation jobs completed.")


def main() -> None:
    ensure_dirs()
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_log_path = RUN_LOG_ROOT / f"run_generate_graphs_{timestamp}.log"
    with tee_output(run_log_path):
        print(f"[info] Runner log written to {run_log_path}")
        commands = enqueue_jobs()
        run_commands(commands)


if __name__ == "__main__":
    main()
