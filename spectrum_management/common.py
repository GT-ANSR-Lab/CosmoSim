"""Shared helpers for beam-mapping algorithms."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from utils import global_variables as global_vars

from .constants import MAX_CHANNELS_PER_CELL

CellRecord = Mapping[str, object]


def prepare_cells(cells: Sequence[CellRecord]) -> List[Dict[str, int]]:
    """Filter cells with demand and normalize types."""
    prepared: List[Dict[str, int]] = []
    for cell in cells:
        cell_id = str(cell["cell"])
        num_terminals = int(cell.get("num_terminals", 0))  # type: ignore[arg-type]
        if num_terminals <= 0:
            continue
        prepared.append({"cell": cell_id, "num_terminals": num_terminals})
    return prepared


def normalize_satellite_list(satellites: Iterable[int]) -> List[int]:
    """Sanitize the iterable of satellite ids into a sorted list."""
    return sorted({int(sat) for sat in satellites})


def initialize_beam_state(satellites: Sequence[int]) -> Tuple[Set[str], Dict[int, List[str]]]:
    """Return all available beams plus an empty sat->cells tracker."""
    all_beams: Set[str] = set()
    sat_cells_assigned: Dict[int, List[str]] = {}
    for sat in satellites:
        sat_cells_assigned[sat] = []
        for beam_idx in range(MAX_CHANNELS_PER_CELL):
            for freq in range(global_vars.frequency_reuse_factor):
                all_beams.add(f"{freq}_{sat}_{beam_idx}")
    return all_beams, sat_cells_assigned


def candidate_sats(cell: str, cell_satellites: Mapping[str, Sequence[int]]) -> List[int]:
    return [int(sat) for sat in cell_satellites.get(cell, [])]


def priority_from_terminals(prepared_cells: Sequence[Mapping[str, int]]) -> Dict[str, int]:
    max_terminals = max((cell["num_terminals"] for cell in prepared_cells), default=0)
    if max_terminals == 0:
        return {cell["cell"]: 0 for cell in prepared_cells}  # type: ignore[index]

    interval = max(1, math.ceil(max_terminals / MAX_CHANNELS_PER_CELL))
    return {
        cell["cell"]: min(MAX_CHANNELS_PER_CELL, max(1, math.ceil(cell["num_terminals"] / interval)))
        for cell in prepared_cells  # type: ignore[index]
    }


def coordinated_priorities(
    prepared_cells: Sequence[Mapping[str, int]], users_per_channel: int
) -> Dict[str, int]:
    if users_per_channel <= 0:
        raise ValueError("users_per_channel must be a positive integer.")
    return {
        cell["cell"]: min(MAX_CHANNELS_PER_CELL, math.ceil(cell["num_terminals"] / users_per_channel))
        for cell in prepared_cells  # type: ignore[index]
    }
