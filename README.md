# CosmoSim

CosmoSim is the an open-source simulator to model satellite network capacity released as part of the
_Assessing LEO Satellite Networks for National Emergency Failover_ paper at IMC 2025.  The code
bundled here is curated for public release: configuration generators, graph
builders, traffic-engineering runners, spectrum-management tooling, and plotting
scripts are organised around a shared data directory so you can reproduce the
pipeline end-to-end without hauling large intermediate artefacts.

If you use this codebase or derivative datasets, please cite:

```
@inproceedings{bhosale2025leo,
  title     = {Assessing LEO Satellite Networks for National Emergency Failover},
  author    = {Vaibhav Bhosale and Ying Zhang and Sameer Kapoor and Robin Kim and Miguel Schlicht and Muskaan Gupta and Ekaterina Tumanova and Zachary Bischof and Fabián E. Bustamante and Alberto Dainotti and Ahmed Saeed},
  booktitle = {Proceedings of the 2025 ACM on Internet Measurement Conference (IMC~2025)},
  year      = {2025}
}
```

## Reproducing the pipeline

The bundled dataset covers the six study countries from the paper (Great Britain,
Ghana, Haiti, Lithuania, South Africa, and Tonga). For any additional geography
you must first obtain the appropriate shapefiles and population rasters,
generate an `inputs/cells/<country>.txt` file, and register the country in
`plotting_scripts/common.py`. Likewise, new constellations must be described
under `constellation_configurations/` (see the existing `starlink_*.yaml`
files) before they can be referenced by the workflows.

### Prerequisites

Install the required native libraries (no ns-3 toolchain needed):

```bash
sudo apt-get update && sudo apt-get install -y libproj-dev proj-data proj-bin libgeos-dev
```

Then install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the following stages from the repository root using the dedicated helper
scripts. Each script spawns the required workflow jobs and logs output under
`data/command_logs/`.

1. **Generate terminal distributions**

   ```bash
   python terminal_deployment/script_cell_allocation.py
   ```

   Adjust the country/population/distribution lists near the top of the script
   (or edit `terminal_deployment/generate_cell_allocations.py` for bespoke
   runs). This stage writes `cells.txt` plus extended scenario assets inside
   `data/scenarios/<scenario_id>/` (where `scenario_id` is a slug such as
   `starlink_5shells_ground_stations_starlink_cells_britain_0_10000_uniform`) and populates
   `data/<scenario_id>_<beam_policy>/demands.txt` with the baseline demand
   snapshots (one directory per beam policy).

2. **Generate graphs**

   ```bash
   python scripts/run_generate_graphs.py
   ```

   Graph snapshots for each constellation/country/time combination are produced
   under `graph_generation/graphs/<constellation>/<country>/1000ms/`.

3. **Generate flows (demand snapshots)**

   ```bash
   python scripts/run_generate_flows.py
   ```

   This invokes `workflows/generate_flows.py` for every scenario, storing
   `demands.txt` files directly under
   `data/<scenario>_<beam_policy>/demands.txt`.

4. **Generate capacities (routing policies)**

   ```bash
   python scripts/run_generate_capacities.py
   ```

   `workflows/generate_capacities.py` converts those demands into routed
   capacities, writing `{routing}_{t}.txt` and `flow_dict_{routing}_{t}.json`
   alongside `demands.txt` in `data/<scenario_id>_<beam_policy>/`.

5. **Generate capacities with competing traffic**

   ```bash
   python scripts/run_generate_capacities_competing_traffic.py
   ```

   This step calls `workflows/generate_capacities_competing_traffic.py` to
   evaluate emergency/incumbent demand priorities. Outputs (max-flow summaries,
   fulfillment logs, first/second-pass flow dictionaries) are stored under
   `data/capacities_competing/<scenario>/t<t>/<beam>/<routing>/<priority>/inc_<demand>/`.

Each workflow shares the same positional arguments (output directory, graph
directory, constellation, ground stations, terminal file, country, flow time,
beam policy, KU-band capacity), so you can invoke them directly if you only need
a single scenario. Let each stage finish before starting the next to ensure all
dependencies are in place.
