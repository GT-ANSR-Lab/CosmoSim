# Post-Processing Scripts

This package contains interactive diagnostic and post-processing tools used to analyze, visualize, and verify the physical layer RF performance of beam mapping. It contains:

| Component | Purpose |
| --- | --- |
| `aggregate_interference.py` | Streamlit-based interactive dashboard that loads serialized beam assignments (`.pkl`), propagates constellation TLE states, computes network-wide interference-to-noise ratio ($I/N$) metrics, and renders map overlays and channel breakdown tables. |
| `critical_beam_pruner.py` | Streamlit-based interactive dashboard that sequentially steps through beam assignments dynamically and evaluates aggregate $I/N$ against the $-12\text{ dB}$ threshold, and prunes critical violations to enforce interference safety. |
| `sole_interferer_model.py` | Streamlit-based interactive dashboard that evaluates isolated link-budget interference between adjacent H3 hexagon cells across shared hardware channels. |

## Dashboard Overview and Data Workflow

The post-processing tools act as inspection and optimization engines for beam assignment simulation runs:

* **Dynamic Path Resolution:** Resolves simulation directory names using configurable parameters such as country selection, terminal count, distribution policies, capacity scaling factors, and routing strategies.
* **Cached and Adaptive RF Computation:** Integrates with `spectrum_management.interference_rf` to calculate received co-channel interference power in Watts, derive channel-specific $I/N$ values, and dynamically prune or analyze threshold violations.
* **Interactive Visualizations:** Renders geographic boundary polygons of H3 user cells, color-coded by compliance safety status (safe vs. critical/dropped).
* **Granular Diagnostics:** Provides filtering tools to isolate specific physical hardware channels, review global vs. scenario-specific statistics (minimum, median, maximum $I/N$), and inspect per-cell spectrum allocation layouts and dropped beam logs.

## Running the Dashboards

To run any of the interactive Streamlit dashboards, navigate into the post-processing directory in your terminal and execute the Streamlit command with the desired target script:

```bash 
streamlit run aggregate_interference.py
```

```bash
streamlit run critical_beam_pruner.py
```

```bash
streamlit run sole_interferer_model.py
```
