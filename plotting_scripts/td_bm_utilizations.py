#!/usr/bin/env python3
"""Plot link-utilization CDFs for CosmoSim TD/BM scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from plotting_scripts.common import (
    DEFAULT_POP_VALUES,
    load_flow_dict,
    scenario_identifier,
    constellation_satellite_count,
    groundstation_count,
)
import utils.global_variables as global_vars

matplotlib.rcParams["pdf.fonttype"] = 42


def parse_int(node: str) -> int | None:
    try:
        return int(node)
    except ValueError:
        return None


def is_network_node(node: str) -> bool:
    if node in {"S", "T"}:
        return False
    return not (node.startswith("D") or node.startswith("I") or node.startswith("E"))


def isl_utilizations(flow_dict: Dict[str, Dict[str, float]], num_sats: int) -> np.ndarray:
    values: list[float] = []
    for node_a, edges in flow_dict.items():
        if not is_network_node(node_a):
            continue
        a_id = parse_int(node_a)
        if a_id is None or not (0 <= a_id < num_sats):
            continue
        for node_b, flow in edges.items():
            if not is_network_node(node_b):
                continue
            b_id = parse_int(node_b)
            if b_id is None or not (0 <= b_id < num_sats):
                continue
            values.append(flow / global_vars.isl_capacity)
    return np.array(values)


def gsl_utilizations(
    flow_dict: Dict[str, Dict[str, float]],
    num_sats: int,
    num_gs: int,
) -> np.ndarray:
    base = num_sats
    gs_totals = np.zeros(num_gs)
    for node_a, edges in flow_dict.items():
        if not is_network_node(node_a):
            continue
        a_id = parse_int(node_a)
        if a_id is None:
            continue
        if not (base <= a_id < base + num_gs):
            continue
        for node_b, flow in edges.items():
            if not is_network_node(node_b):
                continue
            b_id = parse_int(node_b)
            if b_id is None or not (0 <= b_id < num_sats):
                continue
            gs_totals[a_id - base] += flow
    return gs_totals / global_vars.ground_station_gsl_capacity


def plot_cdf(values: np.ndarray, ylabel: str, title: str, output: Path) -> None:
    plt.figure(figsize=(6.4, 4))
    values = np.clip(values, 0, None)
    counts, bin_edges = np.histogram(values, bins=np.linspace(0, 1, 101))
    cdf = np.cumsum(counts) / counts.sum() if counts.sum() else np.zeros_like(counts)
    plt.plot(bin_edges[1:], cdf, "b-")
    plt.xlabel("Link Utilization", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.grid(linewidth=0.5, linestyle=":")
    plt.title(title, fontsize=14)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output.with_suffix(".pdf"))
    plt.savefig(output.with_suffix(".png"), dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GSL/ISL utilization CDFs from CosmoSim flow_dict outputs.")
    parser.add_argument("country", choices=sorted(DEFAULT_POP_VALUES.keys()))
    parser.add_argument("population", type=int)
    parser.add_argument("ut_distribution")
    parser.add_argument("beam_policy", choices=["greedy-coordinated", "greedy-uncoordinated"])
    parser.add_argument("routing", choices=["max_flow", "hot_potato"])
    parser.add_argument("--constellation", default="starlink_5shells")
    parser.add_argument("--groundstations", default="ground_stations_starlink")
    parser.add_argument("--gcb-cap", type=int)
    parser.add_argument("--ku-band", type=float, default=1.28, dest="ku_band_capacity")
    parser.add_argument("--flow-time", type=int, default=0, dest="flow_time")
    parser.add_argument("--output-dir", type=Path, default=Path("plotting_scripts/out"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    constellation = args.constellation
    groundstations = args.groundstations
    scenario = scenario_identifier(
        constellation,
        groundstations,
        args.country,
        args.population,
        args.ut_distribution,
        gcb_cap=args.gcb_cap,
        ku_band_capacity=(args.ku_band_capacity if args.gcb_cap is not None else None),
    )
    flow_dict = load_flow_dict(scenario, args.beam_policy, args.routing, args.flow_time)
    num_sats = constellation_satellite_count(constellation)
    num_gs = groundstation_count(groundstations)

    isl_vals = isl_utilizations(flow_dict, num_sats)
    gsl_vals = gsl_utilizations(flow_dict, num_sats, num_gs)

    base = args.output_dir / f"{scenario}_{args.routing}_t{args.flow_time}"
    plot_cdf(isl_vals, "CDF", "ISL Utilization", base.with_name(base.name + "_isl"))
    plot_cdf(gsl_vals, "CDF", "GS Utilization", base.with_name(base.name + "_gsl"))


if __name__ == "__main__":
    main()
