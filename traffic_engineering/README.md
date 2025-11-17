# Traffic Engineering

Home to the reusable routing logic that CosmoSim workflows import when computing flows and capacities.

## Contents

| Component | Purpose |
| --- | --- |
| `utils/` modules (common, graph_tools, routing/*, etc.) | Shared primitives for manipulating digraph snapshots, building demands, and running flow solvers. |
| `hot_potato.py`, `max_flow.py`, `shortest_path.py` | Reference routing strategies used by `generate_flows.py`/`generate_capacities.py`. |
| Legacy scripts (kept for parity) | Older helpers that remain useful for diagnostics or comparison with legacy pipelines. |

## Routing API

Each routing strategy exposes a function with the signature used in `generate_capacities.py`:

```python
def modify_graph_for_routing(
    di_graph: nx.DiGraph,
    demands: np.ndarray,
    num_ground_stations: int,
    num_sats: int,
) -> None:
    ...
```

or a function returning flow dicts via NetworkX (see `hot_potato_modifications` and `max_flow_min_cost`). Workflows import these utilities directly.

## Adding a New Routing Algorithm

1. Drop a module under `traffic_engineering/routing/` (or a top-level file if it needs broader visibility).
2. Implement the helper(s) that mutate the graph or compute flows. Follow the patterns in `hot_potato.py` or `max_flow.py` for logging and parameter handling.
3. Update workflows or scripts to import the new module when `--routing <name>` is selected.

Keeping the implementations here ensures routing logic stays decoupled from orchestration and can be unit-tested in isolation.
