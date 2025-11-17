"""Greedy, uncoordinated beam allocation."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

from utils import global_variables as global_vars

from .common import (
    candidate_sats,
    initialize_beam_state,
    priority_from_terminals,
)
from .compatibility import check_compatibility
from .constants import MAX_CHANNELS_PER_CELL


def assign_beams(
    prepared_cells: Sequence[Mapping[str, int]],
    satellites: Sequence[int],
    satellite_cells: Mapping[int, Sequence[str]],  # kept for signature parity
    cell_satellites: Mapping[str, Sequence[int]],
) -> Dict[str, str]:
    """Assign beams by repeatedly iterating over cells in priority order."""

    del satellite_cells  # not needed for this heuristic

    if not satellites or not prepared_cells:
        return {}

    cell_priority = priority_from_terminals(prepared_cells)
    beams_available, sat_cells_assigned = initialize_beam_state(satellites)
    mapping: Dict[str, str] = {}
    cell_ids = [cell["cell"] for cell in prepared_cells]  # type: ignore[index]

    for _ in range(MAX_CHANNELS_PER_CELL):
        ordered_cells = sorted(
            cell_ids,
            key=lambda cid: (-cell_priority[cid], len(cell_satellites.get(cid, []))),
        )
        for cell_id in ordered_cells:
            if cell_priority[cell_id] <= 0:
                continue

            candidate = sorted(
                candidate_sats(cell_id, cell_satellites),
                key=lambda sat: len(sat_cells_assigned.get(sat, [])),
            )

            assigned = False
            for channel_idx in range(MAX_CHANNELS_PER_CELL):
                if assigned:
                    break
                dummy_node = f"{cell_id}_{channel_idx}"
                if dummy_node in mapping:
                    continue

                for sat in candidate:
                    sat_cells_assigned.setdefault(sat, [])
                    for freq in range(global_vars.frequency_reuse_factor):
                        beam_id = f"{freq}_{sat}_{channel_idx}"
                        if beam_id in beams_available and check_compatibility(cell_id, mapping, beam_id):
                            mapping[dummy_node] = beam_id
                            beams_available.remove(beam_id)
                            sat_cells_assigned[sat].append(cell_id)
                            cell_priority[cell_id] = max(0, cell_priority[cell_id] - 1)
                            assigned = True
                            break
                    if dummy_node in mapping:
                        break

    return mapping
