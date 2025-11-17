"""Greedy, coordinated beam allocation."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

from utils import global_variables as global_vars

from .common import (
    candidate_sats,
    coordinated_priorities,
    initialize_beam_state,
)
from .compatibility import check_compatibility
from .constants import MAX_CHANNELS_PER_CELL


def assign_beams(
    prepared_cells: Sequence[Mapping[str, int]],
    satellites: Sequence[int],
    satellite_cells: Mapping[int, Sequence[str]],
    cell_satellites: Mapping[str, Sequence[int]],
    config: str,
    shell_satellite_indices: Sequence[Sequence[int]],
    users_per_channel: int,
    cell_population: Mapping[str, int],
) -> Dict[str, str]:
    del satellite_cells  # unused but retained for parity with prior APIs
    del config

    if not satellites or not prepared_cells:
        return {}

    cell_priority = coordinated_priorities(prepared_cells, users_per_channel)
    cell_ids = [cell["cell"] for cell in prepared_cells]  # type: ignore[index]

    beams_available, sat_cells_assigned = initialize_beam_state(satellites)
    mapping: Dict[str, str] = {}
    num_shells = len(shell_satellite_indices)

    ordered_cells = sorted(
        cell_ids,
        key=lambda cid: (
            -cell_priority[cid],
            -cell_population.get(cid, 0),
        ),
    )

    def try_assign(cell_id: str, channel_idx: int) -> bool:
        candidate = _order_satellites(
            candidate_sats(cell_id, cell_satellites),
            shell_satellite_indices,
            sat_cells_assigned,
            num_shells,
        )
        for sat in candidate:
            sat_cells_assigned.setdefault(sat, [])
            for freq in range(global_vars.frequency_reuse_factor):
                beam_id = f"{freq}_{sat}_{channel_idx}"
                if beam_id in beams_available and check_compatibility(cell_id, mapping, beam_id):
                    mapping[f"{cell_id}_{channel_idx}"] = beam_id
                    beams_available.remove(beam_id)
                    sat_cells_assigned[sat].append(cell_id)
                    cell_priority[cell_id] = max(0, cell_priority[cell_id] - 1)
                    return True
        return False

    # First pass prioritizes higher-population cells until their demand is met
    for cell_id in ordered_cells:
        if cell_priority[cell_id] == 0:
            break
        for channel_idx in range(MAX_CHANNELS_PER_CELL):
            if cell_priority[cell_id] == 0:
                break
            try_assign(cell_id, channel_idx)

    # Second pass retries remaining cells even if initial ordering skipped them
    for cell_id in ordered_cells:
        if cell_priority[cell_id] == 0:
            continue
        for channel_idx in range(MAX_CHANNELS_PER_CELL):
            if cell_priority[cell_id] == 0:
                break
            try_assign(cell_id, channel_idx)

    print(
        f"[beam-mapping] policy=greedy-coordinated mapped_slots={len(mapping)}"
    )
    return mapping


def _order_satellites(
    sats: Sequence[int],
    shell_satellite_indices: Sequence[Sequence[int]],
    sat_cells_assigned: Mapping[int, Sequence[str]],
    num_shells: int,
) -> List[int]:
    if not sats:
        return []

    bucket_count = max(num_shells, 1)
    shell_buckets: List[List[int]] = [[] for _ in range(bucket_count)]

    for sat in sats:
        assigned = False
        for idx, bounds in enumerate(shell_satellite_indices):
            start, end = bounds
            if start <= sat < end:
                shell_buckets[idx].append(sat)
                assigned = True
                break
        if not assigned:
            shell_buckets[0].append(sat)

    sat_priority: Dict[int, Tuple[int, int]] = {}
    for idx, bucket in enumerate(shell_buckets):
        if not bucket:
            continue
        threshold = int(0.6 * len(bucket))
        for pos, sat in enumerate(bucket):
            region = idx if pos <= threshold else idx + num_shells
            sat_priority[sat] = (region, len(sat_cells_assigned.get(sat, [])))

    return sorted(
        sats,
        key=lambda sat: (
            sat_priority.get(sat, (0, 0))[0],
            sat_priority.get(sat, (0, 0))[1],
            sat,
        ),
    )
