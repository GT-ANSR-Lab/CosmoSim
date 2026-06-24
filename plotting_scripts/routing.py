#!/usr/bin/env python3
"""Compare CosmoSim routing strategies across the study countries.

Mirrors the IMC ``compare_routing_uniform.py`` figure: a normalized bar chart of
average total capacity per country for each routing policy (max_flow vs
hot_potato), where each country's capacities are normalized to its max_flow
value (so max_flow == 1.0 and hot_potato is shown relative to it).

Like the IMC figure, this defaults to the ``gcb_no_cap`` ("waterfill")
distribution with the ``greedy-coordinated`` ("popwaterfill") beam policy at
Ku-band capacity 1.28; all of these are overridable on the command line.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from plotting_scripts.common import load_capacity_samples, scenario_identifier

matplotlib.rcParams["pdf.fonttype"] = 42

ROUTING_POLICIES = ["max_flow", "hot_potato"]

# One representative population per country, mirroring the IMC routing figure.
DEFAULT_COUNTRY_POPULATIONS = {
    "ghana": 100000,
    "britain": 200000,
    "haiti": 50000,
    "lithuania": 50000,
    "southafrica": 200000,
    "tonga": 1000,
}

COUNTRY_LABELS = {
    "ghana": "Ghana",
    "britain": "Great\nBritain",
    "haiti": "Haiti",
    "lithuania": "Lithuania",
    "southafrica": "South\nAfrica",
    "tonga": "Tonga",
}


def choose_constellation(ku_band_capacity: float) -> str:
    if ku_band_capacity >= 2.5:
        return "starlink_double"
    return "starlink_5shells"


def mean_capacity(
    constellation: str,
    groundstations: str,
    country: str,
    population: int,
    ut_distribution: str,
    beam_policy: str,
    routing: str,
    ku_band_capacity: float,
    flow_time: int,
) -> float:
    # The Ku-band capacity only appears in the scenario_id for GCB distributions.
    ku_label = ku_band_capacity if ut_distribution.startswith("gcb") else None
    scenario_id = scenario_identifier(
        constellation,
        groundstations,
        country,
        int(population),
        ut_distribution,
        ku_band_capacity=ku_label,
    )
    samples = load_capacity_samples(scenario_id, beam_policy, routing, flow_time)
    return float(np.mean(samples))


def plot(countries, populations, args) -> None:
    constellation = args.constellation or choose_constellation(args.ku_band_capacity)
    groundstations = args.groundstations

    capacities = {routing: [] for routing in ROUTING_POLICIES}
    for country in countries:
        for routing in ROUTING_POLICIES:
            capacities[routing].append(
                mean_capacity(
                    constellation,
                    groundstations,
                    country,
                    populations[country],
                    args.ut_distribution,
                    args.beam_policy,
                    routing,
                    args.ku_band_capacity,
                    args.flow_time,
                )
            )

    max_flow = np.array(capacities["max_flow"], dtype=float)
    hot_potato = np.array(capacities["hot_potato"], dtype=float)
    # Normalize each country's capacities to its own max_flow value.
    hot_potato = hot_potato / max_flow
    max_flow = max_flow / max_flow

    x = np.arange(1, len(countries) + 1) * 2  # [2, 4, 6, ...], matching the IMC figure
    bar_width = 0.4

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.4, 3.2))
    plt.bar(x - bar_width / 2, max_flow, bar_width, color="Blue", label="Max Flow")
    plt.bar(x + bar_width / 2, hot_potato, bar_width, color="orange", label="Hot Potato")

    plt.ylabel("Normalized Capacity", fontsize=16)
    plt.xticks(
        ticks=x,
        labels=[COUNTRY_LABELS.get(country, country.title()) for country in countries],
        fontsize=14,
    )
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, frameon=False)
    plt.grid(linewidth=0.5, linestyle=":")
    plt.tight_layout()

    base = args.output_dir / f"routing_{args.ut_distribution}"
    plt.savefig(base.with_suffix(".pdf"))
    plt.savefig(base.with_suffix(".png"), dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot normalized routing-policy capacity across study countries (IMC routing figure)."
    )
    parser.add_argument(
        "--countries",
        nargs="*",
        default=list(DEFAULT_COUNTRY_POPULATIONS.keys()),
        help="Countries to include (default: the six study countries).",
    )
    parser.add_argument(
        "--populations",
        nargs="*",
        type=int,
        help="Per-country populations aligned with --countries (default: IMC values).",
    )
    # The IMC routing figure read the waterfill dataset, i.e. CosmoSim's gcb_no_cap.
    parser.add_argument("--ut-distribution", default="gcb_no_cap", dest="ut_distribution")
    parser.add_argument(
        "--beam-policy",
        default="greedy-coordinated",
        choices=["greedy-coordinated", "greedy-uncoordinated"],
        dest="beam_policy",
    )
    parser.add_argument("--ku-band-capacity", type=float, default=1.28, dest="ku_band_capacity")
    parser.add_argument(
        "--constellation",
        default=None,
        help="Override constellation (default: chosen from --ku-band-capacity).",
    )
    parser.add_argument("--groundstations", default="ground_stations_starlink")
    parser.add_argument("--flow-time", type=int, default=0, dest="flow_time")
    parser.add_argument("--output-dir", type=Path, default=Path("plotting_scripts/out"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    countries = args.countries
    if args.populations:
        if len(args.populations) != len(countries):
            raise ValueError("--populations must align one-to-one with --countries")
        populations = dict(zip(countries, args.populations))
    else:
        missing = [c for c in countries if c not in DEFAULT_COUNTRY_POPULATIONS]
        if missing:
            raise ValueError(f"No default population for {missing}; pass --populations")
        populations = {country: DEFAULT_COUNTRY_POPULATIONS[country] for country in countries}
    plot(countries, populations, args)


if __name__ == "__main__":
    main()
