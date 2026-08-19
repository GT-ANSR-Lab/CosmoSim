# Supporting Analysis

This package contains diagnostic and visualization scripts used to analyze beam assignment snapshots, compute RF metrics (SNR/INR), and generate beam/antenna pattern visualizations. These scripts are designed to provide insight into specific beam assignment scenarios and understand the interference influence of making beam assignments.  

| Component | Purpose |
| --- | --- |
| `supporting_analysis/beam_metrics.py` | Loads serialized beam assignment snapshots (`.pkl`) and prints a pivot table of active beams per `Satellite ID` × `Channel Slot`. Useful for quick tabular summaries of per-satellite load. |
| `supporting_analysis/capacityloss_vs_inr.py` | Generates a simple plot showing capacity loss percentage vs. Interference-to-Noise Ratio ($I/N$) for a range of baseline SNR values. |
| `supporting_analysis/cell_isolator.py` | Cell auditor: inspects a PKL snapshot, computes per-beam $I/N$ contributions using SGP4/TLE propagation, prints per-interferer diagnostics, and shows a 2D satellite scatter plot with beam directions and cell load on satellites. |
| `supporting_analysis/snr_histogram.py` | Computes per-link SNR values from beam assignments snapshot, then plots an SNR distribution histogram to show link quality statistics across a scenario. |

## Submodules

| Path | Purpose |
| --- | --- |
| `supporting_analysis/Additional_satellite_separation/synthetic_separation.py` | Evaluates the INR influence of a pair of satellites placed at synthetic distances serving neighboring h3 cells under single interferer case.|
| `supporting_analysis/Additional_satellite_separation/separation_case_study.py` | Evaluates the INR influence of a pair of satellites placed at all possible separation distances permitted by a certain constellation serving neighboring h3 cells under single interferer case. Also studies a particular beam assignment for a pair of neighboring cells from a given scenario and compares this beam assignment with all possible assignments for the same cells |
| `supporting_analysis/beamspot_visualizer/Tx_gain_pattern.py` | Spatial transmitter gain visualization. |
| `supporting_analysis/beamspot_visualizer/graph_non_reuse.py` | Allowed elevation angle plot for single interferer under non-reuse case. |
| `supporting_analysis/beamspot_visualizer/sole_interferer_non_reuse.py` | Visual allowed elevation angle analysis for a single interferer under non-reuse case to satisfy INR threshold. |
| `supporting_analysis/beamspot_visualizer/sole_interferer_reuse_exclusion.py` | Visual INR analysis for single interferer under satellite reuse case. |
| `supporting_analysis/interference_patterns/Rx_pattern.py` | Receive antenna gain pattern vs off-axis angle plot. |
| `supporting_analysis/interference_patterns/Tx_pattern.py` | Transmit antenna gain pattern vs off-axis angle plot. |
| `supporting_analysis/validations/elevation_check.py` | Checks for 25 degree minimum elevation angle violations across all graph snapshots and provides diagnostics on violating edges. |

