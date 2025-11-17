# Workflows

Single-scenario entry points for CosmoSim simulations.

Each script (e.g., `generate_flows.py`, `generate_capacities.py`, the competing-traffic variant) consumes specific inputs and emits outputs inside `data/<scenario_id>_<beam_policy>/`.

Use these directly for debugging or embed them in the batch runners under `scripts/` for large sweeps.
