"""Greedy Coordinated INR-Aware Beam Allocation Strategy."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

try:
    import h3
except ImportError:
    import h3.api.basic_int as h3

import utils.global_variables as global_vars

from .common import (
    candidate_sats,
    coordinated_priorities,
    initialize_beam_state,
)
from .constants import MAX_CHANNELS_PER_CELL
from .interference_rf import (
    N0_DB,
    calculate_received_interference_watt,
    ecef,
    get_satellite_positions,
)

INR_THRESHOLD = -12.0  


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
    """Greedy coordinated beam allocation with cached aggregate RF INR verification."""
    del satellite_cells

    sat_list = list(satellites)
    if not sat_list or not prepared_cells:
        return {}

    # Dynamic relative path resolution for constellation TLE files
    constellation_name = config.split("_cells_")[0] if "_cells_" in config else "starlink_5shells"
    current_file = Path(__file__).resolve()
    cosmosim_root = current_file.parents[1]  # Resolves up to CosmoSim root directory
    tle_path = cosmosim_root / "constellation_configurations" / "configs" / constellation_name / "tles.txt"

    sim_time = float(getattr(global_vars, "current_simulation_time_s", 0))
    sat_positions = get_satellite_positions(tle_path, sim_time)

    cell_priority = coordinated_priorities(prepared_cells, users_per_channel)
    cell_ids = [str(cell["cell"]) for cell in prepared_cells]

    cell_metadata = {}
    for cid in cell_ids:
        lat, lon = h3.cell_to_latlng(cid)
        cell_metadata[cid] = {"ecef": ecef(lat, lon)}

    beams_available, sat_cells_assigned = initialize_beam_state(sat_list)
    mapping: Dict[str, str] = {}
    num_shells = len(shell_satellite_indices)

    active_assignments: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    max_watts = 10.0 ** ((INR_THRESHOLD + N0_DB) / 10.0)

    ordered_cells = sorted(
        cell_ids,
        key=lambda cid: (
            -cell_priority[cid],
            -cell_population.get(cid, 0),
        ),
    )

    def is_rf_safe(
        cand_cell_id: str, cand_sat_id: int, cand_freq: int, channel_idx: int
    ) -> Tuple[bool, float, List[float]]:
        """O(M) incremental check verifying total aggregate I/N <= INR_THRESHOLD."""
        if not sat_positions or cand_sat_id not in sat_positions:
            return True, 0.0, []

        cand_tgt_ecef = cell_metadata[cand_cell_id]["ecef"]
        cand_sat_ecef = np.array(sat_positions[cand_sat_id])
        co_channel_txs = active_assignments.get(channel_idx, [])

        cand_aggregate_watts = 0.0
        for tx in co_channel_txs:
            tx_sat_id = int(tx["sat_id"])
            if tx_sat_id not in sat_positions:
                continue

            p_watts = calculate_received_interference_watt(
                u_tgt_pos=cand_tgt_ecef,
                u_inf_serving_pos=np.array(tx["ecef"]),
                sat_tgt_pos=cand_sat_ecef,
                sat_inf_pos=np.array(sat_positions[tx_sat_id]),
                channel_idx=channel_idx,
            )
            cand_aggregate_watts += p_watts
            if cand_aggregate_watts > max_watts:
                return False, 0.0, []

        deltas: List[float] = []
        for rx_tx in co_channel_txs:
            rx_sat_id = int(rx_tx["sat_id"])
            if rx_sat_id not in sat_positions:
                deltas.append(0.0)
                continue

            new_link_watts = calculate_received_interference_watt(
                u_tgt_pos=np.array(rx_tx["ecef"]),
                u_inf_serving_pos=cand_tgt_ecef,
                sat_tgt_pos=np.array(sat_positions[rx_sat_id]),
                sat_inf_pos=cand_sat_ecef,
                channel_idx=channel_idx,
            )

            existing_watts = float(rx_tx.get("accumulated_watts", 0.0))
            if existing_watts + new_link_watts > max_watts:
                return False, 0.0, []

            deltas.append(new_link_watts)

        return True, cand_aggregate_watts, deltas

    def try_assign(cell_id: str, channel_idx: int) -> bool:
        candidate = _order_satellites(
            candidate_sats(cell_id, cell_satellites),
            shell_satellite_indices,
            sat_cells_assigned,
            num_shells,
        )
        for sat in candidate:
            sat_cells_assigned.setdefault(sat, [])
            for freq in range(global_vars.spatial_channel_reuse_factor):
                beam_id = f"{freq}_{sat}_{channel_idx}"
                if beam_id in beams_available:
                    is_safe, cand_watts, deltas = is_rf_safe(cell_id, sat, freq, channel_idx)
                    if is_safe:
                        mapping[f"{cell_id}_{channel_idx}"] = beam_id
                        beams_available.remove(beam_id)
                        sat_cells_assigned[sat].append(cell_id)
                        cell_priority[cell_id] = max(0, cell_priority[cell_id] - 1)

                        # Update accumulated interference for existing active nodes
                        co_txs = active_assignments[channel_idx]
                        for idx, d_watts in enumerate(deltas):
                            co_txs[idx]["accumulated_watts"] = (
                                float(co_txs[idx].get("accumulated_watts", 0.0)) + d_watts
                            )

                        # Append candidate with its cached aggregate interference
                        active_assignments[channel_idx].append({
                            "cell_id": cell_id,
                            "sat_id": sat,
                            "reuse": freq,
                            "ecef": cell_metadata[cell_id]["ecef"],
                            "accumulated_watts": cand_watts,
                        })
                        return True
        return False

    for cell_id in ordered_cells:
        if cell_priority[cell_id] == 0:
            continue
        for channel_idx in range(MAX_CHANNELS_PER_CELL):
            if cell_priority[cell_id] == 0:
                break
            try_assign(cell_id, channel_idx)

    for cell_id in ordered_cells:
        if cell_priority[cell_id] == 0:
            continue
        for channel_idx in range(MAX_CHANNELS_PER_CELL):
            if cell_priority[cell_id] == 0:
                break
            try_assign(cell_id, channel_idx)

    print(f"[beam-mapping] policy=greedy-coordinated-inr-aware mapped_slots={len(mapping)}")
    return mapping