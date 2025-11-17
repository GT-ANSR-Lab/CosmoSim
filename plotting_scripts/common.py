from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import json
import numpy as np
import exputil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
DEMANDS_ROOT = DATA_ROOT
MASK_CAPACITY_ROOT = DATA_ROOT / "capacities_masks"

DEFAULT_POP_VALUES: Dict[str, np.ndarray] = {
    "tonga": np.array([100, 200, 500, 1000, 2000, 5000, 10000, 20000]),
    "ghana": np.array([1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]),
    "southafrica": np.array([1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]),
    "lithuania": np.array([1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]),
    "britain": np.array([1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]),
    "haiti": np.array([1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]),
}

DEFAULT_GCB_LIMITS: Dict[str, Iterable[int]] = {
    "tonga": [1000, 10000],
    "ghana": [1000, 10000, 100000],
    "southafrica": [1000, 10000, 100000],
    "lithuania": [1000, 10000, 100000],
    "britain": [1000, 10000, 100000],
    "haiti": [1000, 10000, 100000],
}


def scenario_identifier(
    constellation: str,
    groundstations: str,
    country: str,
    population: int,
    ut_distribution: str,
    gcb_cap: Optional[int] = None,
    ku_band_capacity: Optional[float] = None,
) -> str:
    """Return the canonical scenario_id slug used across CosmoSim outputs."""
    parts = [
        constellation,
        groundstations,
        "cells",
        country,
        "0",
        str(population),
        ut_distribution,
    ]
    if gcb_cap is not None:
        parts.append(str(gcb_cap))
    if ku_band_capacity is not None:
        parts.append(str(ku_band_capacity))
    return "_".join(parts)


def demands_dir(
    scenario_id: str,
    beam_policy: str,
    flow_time: int,
) -> Path:
    return DEMANDS_ROOT / f"{scenario_id}_{beam_policy}"


def capacity_dir(
    scenario_id: str,
    beam_policy: str,
    routing: str,
    flow_time: int,
) -> Path:
    return DEMANDS_ROOT / f"{scenario_id}_{beam_policy}"


def competing_capacity_dir(
    scenario_id: str,
    beam_policy: str,
    routing: str,
    priority: str,
    incumbent_demand: float,
    flow_time: int,
) -> Path:
    return DEMANDS_ROOT / f"{scenario_id}_{beam_policy}"


def competing_file_tag(routing: str, priority: str, incumbent_demand: float, flow_time: int) -> str:
    inc_tag = str(incumbent_demand).replace(".", "p")
    return f"{routing}_{priority}_inc{inc_tag}_t{flow_time}"


def capacity_series(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Capacity file not found: {path}")
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.ndim == 1:
        data = data.reshape(-1, 2)
    return data[:, 1]


def load_capacity_samples(
    scenario_id: str,
    beam_policy: str,
    routing: str,
    flow_time: int,
) -> np.ndarray:
    base = capacity_dir(scenario_id, beam_policy, routing, flow_time)
    path = base / f"{routing}_{flow_time}.txt"
    return capacity_series(path)


def load_demands(scenario_id: str, beam_policy: str, flow_time: int) -> np.ndarray:
    directory = demands_dir(scenario_id, beam_policy, flow_time)
    path = directory / "demands.txt"
    if not path.exists():
        raise FileNotFoundError(f"Demands file not found: {path}")
    return np.loadtxt(path, ndmin=1)


def load_flow_dict(
    scenario_id: str,
    beam_policy: str,
    routing: str,
    flow_time: int,
) -> Dict[str, Dict[str, float]]:
    base = capacity_dir(scenario_id, beam_policy, routing, flow_time)
    path = base / f"flow_dict_{routing}_{flow_time}.json"
    if not path.exists():
        raise FileNotFoundError(f"Flow dictionary not found: {path}")
    with path.open() as fh:
        return json.load(fh)


def load_competing_flow_dict(
    scenario_id: str,
    beam_policy: str,
    routing: str,
    priority: str,
    incumbent_demand: float,
    flow_time: int,
    pass_name: str,
) -> Dict[str, Dict[str, float]]:
    base = competing_capacity_dir(
        scenario_id, beam_policy, routing, priority, incumbent_demand, flow_time
    )
    tag = competing_file_tag(routing, priority, incumbent_demand, flow_time)
    suffix = "first_pass" if pass_name == "first" else "second_pass"
    path = base / f"competing_{suffix}_{tag}.json"
    if not path.exists():
        raise FileNotFoundError(f"Competing flow dictionary not found: {path}")
    with path.open() as fh:
        return json.load(fh)


def load_competing_capacity(
    scenario_id: str,
    beam_policy: str,
    routing: str,
    priority: str,
    incumbent_demand: float,
    flow_time: int,
) -> np.ndarray:
    base = competing_capacity_dir(
        scenario_id, beam_policy, routing, priority, incumbent_demand, flow_time
    )
    tag = competing_file_tag(routing, priority, incumbent_demand, flow_time)
    path = base / f"competing_flow_{tag}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Competing capacity file not found: {path}")
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.ndim == 1:
        data = data.reshape(-1, 3)
    return data


def load_mask_capacity(
    scenario_id: str,
    beam_policy: str,
    mask_mode: str,
    flow_time: int,
) -> np.ndarray:
    base = MASK_CAPACITY_ROOT / scenario_id / f"t{flow_time}" / beam_policy / mask_mode
    path = base / f"mask_{mask_mode}_{flow_time}.txt"
    return capacity_series(path)


def constellation_satellite_count(constellation_name: str) -> int:
    description = (
        PROJECT_ROOT
        / "constellation_configurations"
        / "configs"
        / constellation_name
        / "description.txt"
    )
    if not description.exists():
        raise FileNotFoundError(f"Constellation description missing: {description}")
    props = exputil.PropertiesConfig(str(description))
    num_orbits = json.loads(props.get_property_or_fail("num_orbits"))
    num_sats_per_orbit = json.loads(props.get_property_or_fail("num_sats_per_orbit"))
    return int(sum(o * s for o, s in zip(num_orbits, num_sats_per_orbit)))


def groundstation_count(name: str) -> int:
    filename = name if name.endswith(".txt") else f"{name}.txt"
    path = PROJECT_ROOT / "inputs" / "groundstations" / filename
    if not path.exists():
        raise FileNotFoundError(f"Ground station list missing: {path}")
    with path.open() as fh:
        return sum(1 for line in fh if line.strip())
