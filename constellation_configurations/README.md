# Constellation Configurations

This directory stores the constellation definitions used across CosmoSim.

- `configs/` holds generated constellation state (description files, TLEs, cells, etc.).
- `generate_constellation.py` and template YAML files describe how to synthesize those configs from high-level parameters.

Regenerate the contents whenever constellation parameters change and commit only the configuration templates, not the generated data.
