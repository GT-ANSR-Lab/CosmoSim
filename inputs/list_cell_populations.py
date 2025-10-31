#!/usr/bin/env python3
"""List all H3 level-5 cells and their aggregated population for a country.

Usage:
    python list_cells.py <country_name> [--output cells]

The script reads the country's shapefile and population data from
`inputs/shp_files/<country>/` and writes `<country>.txt` (or
`<country>_level<resolution>.txt` when the resolution is overridden) inside the
specified output directory (default: `cells/`). The CSV format is:

    h3_index,population
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import h3

try:
    from shapely.geometry import Polygon, MultiPolygon
except ImportError:  # pragma: no cover
    from shapely.geometry import Polygon, MultiPolygon


SCRIPT_DIR = Path(__file__).resolve().parent
SHAPE_ROOT = SCRIPT_DIR / "shp_files"


def swap_coordinates(geometry):
    """Swap x/y for shapely geometries (accounts for dataset ordering)."""
    if geometry.geom_type == "Polygon":
        coords = list(geometry.exterior.coords)
        if len(coords[0]) == 3:
            coords = [(x, y) for x, y, _ in coords]
        return Polygon([(y, x) for x, y in coords])
    if geometry.geom_type == "MultiPolygon":
        return MultiPolygon([swap_coordinates(poly) for poly in geometry.geoms])
    raise ValueError(f"Unsupported geometry type: {geometry.geom_type}")


def h3_cells_for_polygon(polygon, resolution):
    if polygon.geom_type == "MultiPolygon":
        polys = list(polygon.geoms)
    else:
        polys = [polygon]

    cells = set()
    for poly in polys:
        coords = list(poly.exterior.coords)
        if len(coords[0]) == 3:
            coords = [(x, y) for x, y, _ in coords]
        geojson_poly = {"type": "Polygon", "coordinates": [coords]}
        cells.update(h3.polyfill(geojson_poly, resolution))
    return list(cells)


def aggregate_population(country_name: str, level: int = 5) -> pd.DataFrame:
    shp_dir = SHAPE_ROOT / country_name
    shp_path = shp_dir / f"{country_name}_level0.shp"
    if not shp_path.exists():
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")

    pop_path = shp_dir / f"{country_name}.gpkg"
    if not pop_path.exists():
        raise FileNotFoundError(f"Population dataset not found: {pop_path}")

    country_gdf = gpd.read_file(shp_path)
    if country_gdf.crs is None or country_gdf.crs.to_epsg() != 4326:
        country_gdf = country_gdf.to_crs(epsg=4326)
    country_geom = country_gdf.geometry.apply(swap_coordinates).iloc[0]

    cells = h3_cells_for_polygon(country_geom, level)
    if country_name.lower() == "southafrica":
        cells = remove_lesotho_cells(cells, level)

    pop_gdf = gpd.read_file(pop_path)
    pop_by_cell = {cell: 0 for cell in cells}

    for _, row in pop_gdf.iterrows():
        res8 = row.get("h3")
        population = int(row.get("population", 0))
        if not res8:
            continue
        parent = h3.h3_to_parent(res8, level)
        if parent in pop_by_cell:
            pop_by_cell[parent] += population

    df = pd.DataFrame([
        {"h3_index": cell, "population": pop_by_cell[cell]}
        for cell in sorted(pop_by_cell)
    ])
    return df


def remove_lesotho_cells(cells, level):
    lesotho_path = SHAPE_ROOT / "lesotho" / "lesotho.shp"
    if not lesotho_path.exists():
        return cells

    lesotho_gdf = gpd.read_file(lesotho_path)
    if lesotho_gdf.crs is None or lesotho_gdf.crs.to_epsg() != 4326:
        lesotho_gdf = lesotho_gdf.to_crs(epsg=4326)
    lesotho_geom = lesotho_gdf.geometry.apply(swap_coordinates).iloc[0]
    lesotho_cells = set(h3_cells_for_polygon(lesotho_geom, level))
    return [cell for cell in cells if cell not in lesotho_cells]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("country", help="Country name matching inputs/shp_files/<country>/")
    parser.add_argument(
        "--resolution",
        type=int,
        default=5,
        help="H3 resolution level to aggregate to (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    output_dir = SCRIPT_DIR / "cells"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = aggregate_population(args.country, args.resolution)
    first_cols = ["h3_index", "population"]
    df = df[first_cols]

    if args.resolution == 5:
        filename = f"{args.country}.txt"
    else:
        filename = f"{args.country}_level{args.resolution}.txt"
    out_path = output_dir / filename
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} cells to {out_path}")


if __name__ == "__main__":
    main()
