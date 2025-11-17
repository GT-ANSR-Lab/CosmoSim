#!/usr/bin/env python3
"""Plot TD vs beam mapping comparisons using CosmoSim outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from plotting_scripts.common import (
    DEFAULT_GCB_LIMITS,
    DEFAULT_POP_VALUES,
    load_capacity_samples,
    scenario_identifier,
)

matplotlib.rcParams["pdf.fonttype"] = 42

TD_BM_UTS: Dict[str, str] = {
    "population": "greedy-coordinated",
    "gcb_no_cap": "greedy-coordinated",
}
VARIANT_UTS: Dict[str, str] = {
    "gcb_cap": "greedy-coordinated",
}


def choose_constellation(ku_band_capacity: float) -> str:
    if ku_band_capacity >= 2.5:
        return "starlink_double"
    return "starlink_5shells"


def read_primary_data(
    constellation: str,
    groundstations: str,
    country: str,
    pop_values: Iterable[int],
    ku_band_capacity: float,
    routing: str,
    flow_time: int,
) -> Dict[str, np.ndarray]:
    data: Dict[str, List[np.ndarray]] = {key: [] for key in TD_BM_UTS}
    for ut_dist, beam_policy in TD_BM_UTS.items():
        for pop in pop_values:
            scenario_id = scenario_identifier(
                constellation,
                groundstations,
                country,
                int(pop),
                ut_dist,
            )
            samples = load_capacity_samples(scenario_id, beam_policy, routing, flow_time)
            data[ut_dist].append(samples)
    return {key: np.array(value, dtype=object) for key, value in data.items()}


def read_variant_data(
    constellation: str,
    groundstations: str,
    country: str,
    pop_values: Iterable[int],
    ku_band_capacity: float,
    routing: str,
    flow_time: int,
) -> Dict[str, np.ndarray]:
    variants: Dict[str, List[np.ndarray]] = {key: [] for key in VARIANT_UTS}
    limits = list(DEFAULT_GCB_LIMITS.get(country, []))
    if not limits:
        return {key: np.array(value, dtype=object) for key, value in variants.items()}
    for ut_dist, beam_policy in VARIANT_UTS.items():
        for pop in pop_values:
            limit_series = []
            for limit in limits:
                scenario_id = scenario_identifier(
                    constellation,
                    groundstations,
                    country,
                    int(pop),
                    ut_dist,
                    gcb_cap=int(limit),
                    ku_band_capacity=ku_band_capacity,
                )
                samples = load_capacity_samples(scenario_id, beam_policy, routing, flow_time)
                limit_series.append(samples)
            variants[ut_dist].append(np.array(limit_series, dtype=object))
    return {key: np.array(value, dtype=object) for key, value in variants.items()}


def plot_td_bm(
    country: str,
    ku_band_capacity: float,
    routing: str,
    flow_time: int,
    output_dir: Path,
) -> None:
    constellation = choose_constellation(ku_band_capacity)
    groundstations = "ground_stations_starlink"
    pop_values = DEFAULT_POP_VALUES[country]

    primary = read_primary_data(
        constellation,
        groundstations,
        country,
        pop_values,
        ku_band_capacity,
        routing,
        flow_time,
    )
    variants = read_variant_data(
        constellation,
        groundstations,
        country,
        pop_values,
        ku_band_capacity,
        routing,
        flow_time,
    )

    plt.figure(figsize=(6.4, 3.2))
    for ut_dist, series in primary.items():
        averages = [np.mean(samples) / 1000 for samples in series]
        label = ut_dist.replace("_", " ").title()
        linestyle = "-" if ut_dist == "population" else "--"
        plt.plot(pop_values // 1000, averages, label=f"{label} ({routing})", linestyle=linestyle, marker="o")

    for ut_dist, matrix in variants.items():
        if matrix.size == 0:
            continue
        limits = DEFAULT_GCB_LIMITS.get(country, [])
        for idx, limit in enumerate(limits):
            averages = [np.mean(samples[idx]) / 1000 for samples in matrix]
            plt.plot(
                pop_values // 1000,
                averages,
                marker="s",
                linestyle="-.",
                label=f"{ut_dist.replace('_', ' ').title()} {limit:,}",
            )

    plt.xlabel("Number of Terminals (1000s)", fontsize=14)
    plt.ylabel("Average Total Capacity (Tbps)", fontsize=14)
    plt.grid(linewidth=0.5, linestyle=":")
    plt.legend(fontsize=10, frameon=False)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"td_bm_{country}_ku{ku_band_capacity:g}"
    plt.savefig(base.with_suffix(".pdf"))
    plt.savefig(base.with_suffix(".png"), dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot TD vs beam-mapping trade-offs using CosmoSim outputs.")
    parser.add_argument("country", choices=sorted(DEFAULT_POP_VALUES.keys()))
    parser.add_argument("ku_band_capacity", type=float, nargs="?", default=1.28)
    parser.add_argument("--routing", default="max_flow", choices=["max_flow", "hot_potato"])
    parser.add_argument("--flow-time", type=int, default=0, dest="flow_time")
    parser.add_argument("--output-dir", type=Path, default=Path("plotting_scripts/out"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_td_bm(
        country=args.country,
        ku_band_capacity=args.ku_band_capacity,
        routing=args.routing,
        flow_time=args.flow_time,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
