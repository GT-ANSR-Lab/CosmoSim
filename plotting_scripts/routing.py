#!/usr/bin/env python3
"""Compare CosmoSim routing strategies for a given scenario."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from plotting_scripts.common import DEFAULT_POP_VALUES, load_capacity_samples, scenario_identifier

matplotlib.rcParams["pdf.fonttype"] = 42

ROUTING_POLICIES = ["max_flow", "hot_potato"]

def choose_constellation(ku_band_capacity: float) -> str:
    if ku_band_capacity >= 2.5:
        return "starlink_double"
    return "starlink_5shells"

def read_data(country: str, ku_band_capacity: float, beam_policy: str, flow_time: int) -> np.ndarray:
    constellation = choose_constellation(ku_band_capacity)
    groundstations = "ground_stations_starlink"
    pop_values = DEFAULT_POP_VALUES[country]
    data = []
    for routing in ROUTING_POLICIES:
        pop_data = []
        for pop in pop_values:
            scenario_id = scenario_identifier(
                constellation,
                groundstations,
                country,
                int(pop),
                "uniform",
            )
            samples = load_capacity_samples(scenario_id, beam_policy, routing, flow_time)
            pop_data.append(samples)
        data.append(np.array(pop_data, dtype=object))
    return pop_values, data

def plot(country: str, ku_band_capacity: float, beam_policy: str, flow_time: int, output_dir: Path) -> None:
    pop_values, data = read_data(country, ku_band_capacity, beam_policy, flow_time)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.4, 3.2))
    for routing, series in zip(ROUTING_POLICIES, data):
        averages = [np.mean(samples) / 1000 for samples in series]
        label = routing.replace("_", " ").title()
        linestyle = "-" if routing == "max_flow" else "--"
        plt.plot(pop_values // 1000, averages, label=label, marker="o", linestyle=linestyle)

    plt.xlabel("Number of Terminals (1000s)", fontsize=14)
    plt.ylabel("Average Total Capacity (Tbps)", fontsize=14)
    plt.grid(linewidth=0.5, linestyle=":")
    plt.legend(fontsize=12, frameon=False)
    plt.tight_layout()

    base = output_dir / f"routing_{country}_ku{ku_band_capacity:g}"
    plt.savefig(base.with_suffix(".pdf"))
    plt.savefig(base.with_suffix(".png"), dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot routing strategy comparison for CosmoSim data.")
    parser.add_argument("country", choices=sorted(DEFAULT_POP_VALUES.keys()))
    parser.add_argument("ku_band_capacity", type=float, nargs="?", default=1.28)
    parser.add_argument("--beam-policy", default="greedy-coordinated", choices=["greedy-coordinated", "greedy-uncoordinated"], dest="beam_policy")
    parser.add_argument("--flow-time", type=int, default=0, dest="flow_time")
    parser.add_argument("--output-dir", type=Path, default=Path("plotting_scripts/out"))
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    plot(
        country=args.country,
        ku_band_capacity=args.ku_band_capacity,
        beam_policy=args.beam_policy,
        flow_time=args.flow_time,
        output_dir=args.output_dir,
    )

if __name__ == "__main__":
    main()
