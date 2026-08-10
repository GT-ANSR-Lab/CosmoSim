"""Greedy Uncoordinated INR-Aware Beam Allocation Strategy."""

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
    initialize_beam_state,
    priority_from_terminals,
)
from .constants import MAX_CHANNELS_PER_CELL
from .interference_rf import (
    N0_DB,
    calculate_received_interference_watt,
    ecef,
    get_satellite_positions,
)

INR_THRESHOLD = -12.0  


def assign_beams(
    prepared_cells: Sequence[Mapping[str, int]],
    satellites: Sequence[int],
    satellite_cells: Mapping[int, Sequence[str]],
    cell_satellites: Mapping[str, Sequence[int]],
    config: str = "",
    shell_satellite_indices: Sequence[Sequence[int]] = (),
    users_per_channel: int = 1,
    cell_population: Mapping[str, int] | None = None,
    *args,
    **kwargs,
) -> Dict[str, str]:
    """Greedy uncoordinated beam allocation with cached aggregate RF INR verification."""
    del satellite_cells, config, shell_satellite_indices, users_per_channel, cell_population

    sat_list = list(satellites)
    if not sat_list or not prepared_cells:
        return {}

    current_file = Path(__file__).resolve()
    cosmosim_root = current_file.parents[1]  
    config_name = getattr(global_vars, "active_constellation_config", "starlink_5shells")
    tle_path = cosmosim_root / "constellation_configurations" / "configs" / config_name / "tles.txt"

    sim_time = float(getattr(global_vars, "current_simulation_time_s", 0))
    sat_positions = get_satellite_positions(tle_path, sim_time)

    cell_priority = priority_from_terminals(prepared_cells)
    beams_available, sat_cells_assigned = initialize_beam_state(sat_list)
    mapping: Dict[str, str] = {}
    cell_ids = [str(cell["cell"]) for cell in prepared_cells]  

    cell_metadata = {}
    for cid in cell_ids:
        lat, lon = h3.cell_to_latlng(cid)
        cell_metadata[cid] = {"ecef": ecef(lat, lon)}

    active_assignments: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    max_watts = 10.0 ** ((INR_THRESHOLD + N0_DB) / 10.0)

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

    # Multi-pass uncoordinated allocation loop
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
                    for reuse in range(global_vars.spatial_channel_reuse_factor):
                        beam_id = f"{reuse}_{sat}_{channel_idx}"
                        if beam_id in beams_available:
                            is_safe, cand_watts, deltas = is_rf_safe(cell_id, sat, reuse, channel_idx)
                            if is_safe:
                                mapping[dummy_node] = beam_id
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
                                    "reuse": reuse,
                                    "ecef": cell_metadata[cell_id]["ecef"],
                                    "accumulated_watts": cand_watts,
                                })
                                assigned = True
                                break
                    if dummy_node in mapping:
                        break

    print(f"[beam-mapping] policy=greedy-uncoordinated-inr-aware mapped_slots={len(mapping)}")
    return mapping