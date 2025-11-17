"""Compatibility helpers for bridging H3 v3/v4 API differences."""

from __future__ import annotations

from typing import Iterable, Iterator, List, Sequence, Tuple, Union

import h3


def cell_to_latlon(cell_id: str) -> Tuple[float, float]:
    """Return (lat, lon) for a cell working across H3 versions."""
    if hasattr(h3, "h3_to_geo"):
        lat, lon = h3.h3_to_geo(cell_id)
        return float(lat), float(lon)
    if hasattr(h3, "cell_to_latlng"):
        lat, lon = h3.cell_to_latlng(cell_id)
        return float(lat), float(lon)
    raise AttributeError("Installed h3 package is missing cell->lat/lon helpers.")


def cell_to_boundary(cell_id: str, *, geo_json: bool = False) -> List[Tuple[float, float]]:
    """Return polygon boundary points for a cell."""
    if hasattr(h3, "h3_to_geo_boundary"):
        return h3.h3_to_geo_boundary(cell_id, geo_json=geo_json)
    if hasattr(h3, "cell_to_boundary"):
        return h3.cell_to_boundary(cell_id, geo_json=geo_json)
    raise AttributeError("Installed h3 package is missing cell boundary helpers.")


def k_ring(cell_id: str, radius: int) -> List[str]:
    """Return all cells within radius steps of the origin cell across H3 versions."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if hasattr(h3, "k_ring"):
        return list(h3.k_ring(cell_id, radius))
    if hasattr(h3, "grid_disk"):
        disk = h3.grid_disk(cell_id, radius)
        flattened: List[str] = []
        seen = set()
        for entry in disk:
            if isinstance(entry, (list, tuple, set)):
                iterable = entry
            else:
                iterable = (entry,)
            for cell in iterable:
                if cell in seen:
                    continue
                seen.add(cell)
                flattened.append(cell)
        return flattened
    raise AttributeError("Installed h3 package is missing neighborhood helpers.")
