#!/usr/bin/env python3
"""Compare CosmoSim mask-on/off capacity scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from plotting_scripts.common import (
    DEFAULT_GCB_LIMITS,
    load_capacity_samples,
    load_mask_capacity,
    scenario_identifier,
)

matplotlib.rcParams["pdf.fonttype"] = 42
MASK_MODES = ("on", "off")


def parse_int_list(values: Iterable[str]) -> List[int]:
    return [int(value) for value in values]


def plot(country: str, pops: List[int], limits: List[int], args) -> None:
    if len(pops) != len(limits):
        raise ValueError("populations and gcb-caps lists must align")
    bars = {"max": [], "mask_on": [], "mask_off": []}
    constel = args.constellation
    groundstations = args.groundstations
    for pop, limit in zip(pops, limits):
        scenario = scenario_identifier(
            constel,
            groundstations,
            country,
            pop,
            args.ut_distribution,
            gcb_cap=limit,
            ku_band_capacity=args.ku_band_capacity,
        )
        baseline = load_capacity_samples(
            scenario,
            args.beam_policy,
            args.routing,
            args.flow_time,
        )
        mask_on = load_mask_capacity(
            scenario,
            args.beam_policy,
            "on",
            args.flow_time,
        )
        mask_off = load_mask_capacity(
            scenario,
            args.beam_policy,
            "off",
            args.flow_time,
        )
        max_value = max(np.max(baseline), np.max(mask_on), np.max(mask_off)) or 1
        bars["max"].append(np.mean(baseline) / max_value)
        bars["mask_on"].append(np.mean(mask_on) / max_value)
        bars["mask_off"].append(np.mean(mask_off) / max_value)

    x = np.arange(len(pops))
    width = 0.25
    plt.figure(figsize=(6.4, 3.2))
    plt.bar(x - width, bars["max"], width, label="All Gateways", color="#1f77b4")
    plt.bar(x, bars["mask_on"], width, label="Mask On", color="#d62728")
    plt.bar(x + width, bars["mask_off"], width, label="Mask Off", color="#2ca02c")
    plt.ylabel("Normalized Capacity", fontsize=14)
    labels = [f"pop={pop:,}
cap={limit:,}" for pop, limit in zip(pops, limits)]
    plt.xticks(x, labels, fontsize=10)
    plt.yticks(fontsize=12)
    plt.grid(axis="y", linewidth=0.5, linestyle=":")
    plt.legend(fontsize=10, frameon=False, loc="upper center", ncol=3)
    plt.tight_layout()
    base = args.output_dir / f"mask_{country}_ku{args.ku_band_capacity:g}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(base.with_suffix(".pdf"))
    plt.savefig(base.with_suffix(".png"), dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot effects of ground-station masks using CosmoSim data.")
    parser.add_argument("country")
    parser.add_argument("--constellation", default="starlink_5shells")
    parser.add_argument("--groundstations", default="ground_stations_starlink")
    parser.add_argument("--ut-distribution", default="gcb_cap")
    parser.add_argument("--beam-policy", default="greedy-coordinated", dest="beam_policy")
    parser.add_argument("--routing", default="max_flow")
    parser.add_argument("--ku-band-capacity", type=float, default=1.28, dest="ku_band_capacity")
    parser.add_argument("--flow-time", type=int, default=0, dest="flow_time")
    parser.add_argument("--populations", nargs="*", type=int)
    parser.add_argument("--gcb-caps", nargs="*", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("plotting_scripts/out"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    country = args.country
    default_limits = list(DEFAULT_GCB_LIMITS.get(country, []))
    if args.populations:
        pops = args.populations
    else:
        pops = default_limits[:]
    if args.gcb_caps:
        limits = args.gcb_caps
    else:
        limits = default_limits
    if not pops or not limits:
        raise ValueError(f"No defaults for {country}; specify --populations and --gcb-caps")
    plot(country, pops, limits, args)


if __name__ == "__main__":
    main()
