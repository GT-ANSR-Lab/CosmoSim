"""High-level entrypoint for beam-to-cell mapping."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

from .common import normalize_satellite_list, prepare_cells
from .greedy_coordinated import assign_beams as assign_greedy_coordinated
from .greedy_uncoordinated import assign_beams as assign_greedy_uncoordinated


def beam_mapping(
    policy: str,
    cells: Sequence[Mapping[str, object]],
    satellites: Iterable[int],
    satellite_cells: Mapping[int, Sequence[str]],
    cell_satellites: Mapping[str, Sequence[int]],
    config: str,
    shell_satellite_indices: Sequence[Sequence[int]],
    users_per_channel: int,
    cell_population: Mapping[str, int],
) -> Dict[str, str]:
    """Dispatch beam mapping based on the requested policy."""

    normalized_policy = _normalize_policy(policy)
    prepared_cells = prepare_cells(cells)
    sat_list = normalize_satellite_list(satellites)

    if normalized_policy == "greedy-uncoordinated":
        mapping = assign_greedy_uncoordinated(
            prepared_cells,
            sat_list,
            satellite_cells,
            cell_satellites,
        )
    elif normalized_policy == "greedy-coordinated":
        mapping = assign_greedy_coordinated(
            prepared_cells,
            sat_list,
            satellite_cells,
            cell_satellites,
            config,
            shell_satellite_indices,
            users_per_channel,
            cell_population,
        )
    else:
        raise ValueError(
            "Unsupported beam-mapping policy. Expected 'greedy-uncoordinated' or 'greedy-coordinated'."
        )

    print(
        f"[beam-mapping] policy={normalized_policy} cells={len(prepared_cells)} "
        f"mapped_slots={len(mapping)}"
    )
    return mapping


def _normalize_policy(policy: str) -> str:
    return policy.strip().lower().replace("_", "-")
