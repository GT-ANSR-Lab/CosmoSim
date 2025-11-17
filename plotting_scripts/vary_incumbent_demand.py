#!/usr/bin/env python3
"""Plot incumbent-demand sweeps using CosmoSim competing-traffic outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from plotting_scripts.common import (
    DEFAULT_GCB_LIMITS,
    load_competing_capacity,
    scenario_identifier,
)

matplotlib.rcParams["pdf.fonttype"] = 42


def parse_float_list(values: Iterable[str]) -> List[float]:
    return [float(value) for value in values]


def parse_int_list(values: Iterable[str]) -> List[int]:
    return [int(value) for value in values]


def plot(country: str, pops: List[int], caps: List[int], args) -> None:
    if len(pops) != len(caps):
        raise ValueError("populations and gcb-caps lists must align")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    incumbent_demands = args.incumbent_demands
    linestyle_map = ["-", "--", ":", "-."]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plt.figure(figsize=(6.4, 3.2))
    for idx, (pop, cap) in enumerate(zip(pops, caps)):
        scenario = scenario_identifier(
            args.constellation,
            args.groundstations,
            country,
            pop,
            args.ut_distribution,
            gcb_cap=cap,
            ku_band_capacity=args.ku_band_capacity,
        )
        series = []
        for inc in incumbent_demands:
            data = load_competing_capacity(
                scenario,
                args.beam_policy,
                args.routing,
                args.priority,
                inc,
                args.flow_time,
            )
            emergency = data[:, 1]
            incumbent = data[:, 2]
            if args.priority == "emergency":
                series.append(np.mean(emergency))
            else:
                series.append(np.mean(incumbent))
        series = np.array(series)
        series = series / (np.max(series) or 1)
        style = linestyle_map[idx % len(linestyle_map)]
        plt.plot(
            incumbent_demands,
            series,
            linestyle=style,
            marker="o",
            color=colors[idx % len(colors)],
            label=f"pop={pop:,} cap={cap:,}",
        )

    plt.xlabel("Incumbent demand per satellite (Gbps)", fontsize=14)
    plt.ylabel("Normalized Capacity", fontsize=14)
    plt.grid(linewidth=0.5, linestyle=":")
    plt.legend(fontsize=10, frameon=False)
    plt.tight_layout()
    base = output_dir / f"vary_incumbent_{country}_ku{args.ku_band_capacity:g}_{args.priority}"
    plt.savefig(base.with_suffix(".pdf"))
    plt.savefig(base.with_suffix(".png"), dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot incumbent-demand sweeps using competing traffic outputs.")
    parser.add_argument("country")
    parser.add_argument("--constellation", default="starlink_5shells")
    parser.add_argument("--groundstations", default="ground_stations_starlink")
    parser.add_argument("--ut-distribution", default="gcb_cap")
    parser.add_argument("--beam-policy", default="greedy-coordinated", dest="beam_policy")
    parser.add_argument("--routing", default="max_flow")
    parser.add_argument("--priority", default="incumbent", choices=["emergency", "incumbent"])
    parser.add_argument("--ku-band-capacity", type=float, default=1.28, dest="ku_band_capacity")
    parser.add_argument("--flow-time", type=int, default=0, dest="flow_time")
    parser.add_argument("--populations", nargs="*", type=int)
    parser.add_argument("--gcb-caps", nargs="*", type=int)
    parser.add_argument("--incumbent-demands", nargs="*", type=float, default=[0.05, 0.1, 0.2, 0.5, 0.8, 1.0])
    parser.add_argument("--output-dir", type=Path, default=Path("plotting_scripts/out"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_caps = list(DEFAULT_GCB_LIMITS.get(args.country, []))
    pops = args.populations or default_caps
    caps = args.gcb_caps or default_caps
    if not pops or not caps:
        raise ValueError(f"No defaults available for {args.country}; provide --populations and --gcb-caps")
    args.incumbent_demands = args.incumbent_demands if isinstance(args.incumbent_demands, list) else parse_float_list(args.incumbent_demands)
    plot(args.country, pops, caps, args)


if __name__ == "__main__":
    main()
