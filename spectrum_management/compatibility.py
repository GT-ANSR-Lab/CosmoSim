"""Beam compatibility helpers."""

from __future__ import annotations

from typing import Mapping

from utils.h3_compat import k_ring

from .constants import MAX_CHANNELS_PER_CELL


def check_compatibility(cell: str, mappings: Mapping[str, str], proposed_beam: str) -> bool:
    """Return True when the proposed beam assignment does not conflict with neighbors."""
    proposed_sat = int(proposed_beam.split("_")[1])
    proposed_beam_idx = int(proposed_beam.split("_")[2])

    for channel_idx in range(MAX_CHANNELS_PER_CELL):
        dummy = f"{cell}_{channel_idx}"
        if dummy in mappings and proposed_beam_idx == int(mappings[dummy].split("_")[2]):
            return False

    for neighbor in k_ring(cell, 1):
        for channel_idx in range(MAX_CHANNELS_PER_CELL):
            dummy = f"{neighbor}_{channel_idx}"
            if dummy not in mappings:
                continue
            sat_id, beam_idx = mappings[dummy].split("_")[1:]
            if proposed_sat == int(sat_id) and proposed_beam_idx == int(beam_idx):
                return False

    return True
