# Supporting Analysis

This package contains standalone diagnostic and visualization scripts used to analyze beam-assignment snapshots, compute RF metrics (SNR/INR), and generate beam/antenna pattern visualizations. It provides lightweight, script-driven tools for rapid post-simulation inspection and local figure generation.

| Component | Purpose |
| --- | --- |
| `supporting_analysis/beam_metrics.py` | Loads serialized beam-assignment snapshots (`.pkl`) and prints a pivot table of active beams per `Satellite ID` × `Channel Slot`. Useful for quick tabular summaries of per-satellite load. |
| `supporting_analysis/capacityloss_vs_inr.py` | Generates a simple Matplotlib plot showing percent capacity loss vs. Interference-to-Noise Ratio ($I/N$) for a range of baseline SNR values. |
| `supporting_analysis/cell_isolator.py` | Cell auditor: inspects a PKL snapshot, computes per-beam $I/N$ contributions using SGP4/TLE propagation, prints per-interferer diagnostics, and opens an interactive 2D satellite scatter plot with hover tooltips (requires `mplcursors`). |
| `supporting_analysis/snr_histogram.py` | Computes per-link SNR values from beam assignments and constellation TLEs, then plots an SNR distribution histogram. Useful for assessing link-quality statistics across a scenario. |

## Submodules

| Path | Purpose |
| --- | --- |
| `supporting_analysis/Additional_satellite_separation/separation_case_study.py` | Case-study analysis utilities for satellite separation scenarios. |
| `supporting_analysis/Additional_satellite_separation/synthetic_separation.py` | Synthetic separation data generator and analysis helper. |
| `supporting_analysis/beamspot_visualizer/Tx_gain_pattern.py` | Tools for generating and plotting transmit gain (beam) patterns. |
| `supporting_analysis/beamspot_visualizer/graph_non_reuse.py` | Visual comparisons for non-reuse beam scenarios. |
| `supporting_analysis/beamspot_visualizer/sole_interferer_non_reuse.py` | Visual analysis for a single interferer under non-reuse assumptions. |
| `supporting_analysis/beamspot_visualizer/sole_interferer_reuse_exclusion.py` | Visual analysis for single interferer with reuse/exclusion rules. |
| `supporting_analysis/interference_patterns/Rx_pattern.py` | Receive antenna pattern helpers and plotting utilities. |
| `supporting_analysis/interference_patterns/Tx_pattern.py` | Transmit antenna pattern helpers and plotting utilities. |
| `supporting_analysis/validations/elevation_check.py` | Validation utilities such as elevation-angle checks and other sanity tests. |

## Overview and Data Workflow

The supporting analysis tools act as lightweight diagnostic and visualization engines for rapid post-simulation inspection:

* **Dynamic Path Resolution:** Resolves simulation data directories under the repository `data/` tree using scenario naming conventions (country, user terminal counts, distribution, routing policies, etc.).
* **TLE-Based Propagation:** Propagates constellation TLEs via `sgp4` and converts to ECEF/ITRS using `astropy` for accurate slant-range and off-axis gain computations when calculating SNR/$I/N$.
* **Per-Link RF Computations:** Implements link-budget math, free-space path loss (FSPL), antenna off-axis gain models (ITU-like patterns), and noise floor calculations for quick inspection and debugging.
* **Interactive Plotting:** Renders interactive Matplotlib windows providing hover metadata and distribution insights (requires `mplcursors`).

## Dependencies

Common Python dependencies used by the scripts (install into your environment):

```bash
pip install numpy pandas matplotlib sgp4 astropy h3 mplcursors tqdm
