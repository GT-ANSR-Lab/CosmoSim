# Inputs

Static input assets required by CosmoSim live here.

- `groundstations/` holds curated ground-station catalogs (text/CSV/imagery).
- `cells/` contains country-level cells and their populations. These files are generated via the `graph_generation` pipeline (see `graph_generation/generate_digraphs.py`) after ingesting the shape inputs below; do not hand-edit them.
- `shp_files/` and other GIS downloads should be refreshed from source data when needed.

Treat this directory as the canonical source for slow-changing inputs shared across workflows. When datasets need updating, re-download them from the original sources listed below.

## Shape Sources

- Britain: https://statistics.ukdataservice.ac.uk/dataset/2011-census-geography-boundaries-great-britain/resource/284514b9-3eef-4521-9e0a-f6489e02668c
- Haiti: https://data.humdata.org/dataset/haiti-administrative-boundaries
- Ghana: https://data.humdata.org/dataset/ghana-administrative-boundaries
- South Africa: https://data.humdata.org/dataset/south-africa-administrative-boundaries
- Tonga: https://data.humdata.org/dataset/tonga-administrative-boundaries
- Lithuania: https://data.humdata.org/dataset/lithuania-administrative-boundaries

## Population Data

- Kontur population dataset: https://data.humdata.org/dataset/kontur-population-dataset
