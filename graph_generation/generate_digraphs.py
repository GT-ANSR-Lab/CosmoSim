import os
import shlex
import time
from pathlib import Path

import exputil

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASE_DIR
DIGRAPHS_DIR = (BASE_DIR / "data" / "digraphs").resolve()
CONSTELLATION_OUTPUT_DIR_CANDIDATES = [
    (BASE_DIR / "constellation_configs" / "gen_data").resolve(),
    (BASE_DIR / "constellation_configurations" / "gen_data").resolve(),
    (BASE_DIR / "constellation_configurations" / "configs").resolve(),
]


def resolve_constellation_dir(constellation_id: str) -> Path:
    for root in CONSTELLATION_OUTPUT_DIR_CANDIDATES:
        if not root.exists():
            continue
        candidate = root / constellation_id
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Constellation output not found for '{constellation_id}'. "
        "Generate configurations first (expected under constellation_configs/gen_data "
        "or constellation_configurations/{gen_data,configs})."
    )


def extract_country(constellation_id: str) -> str:
    marker = "_cells_"
    idx = constellation_id.find(marker)
    if idx == -1:
        raise ValueError(
            f"Unable to infer country from constellation id '{constellation_id}'. "
            "Expected pattern '*_cells_<country>_...'."
        )
    suffix = constellation_id[idx + len(marker):]
    return suffix.split("_", 1)[0]

local_shell = exputil.LocalShell()
# this number should always be lower than the number of cores. A good rule of thumb is to use the number of cores minus 1
max_num_processes = 511

# Create graphs directory if not present already
os.makedirs(DIGRAPHS_DIR, exist_ok=True)

command_logs_dir = DIGRAPHS_DIR / "command_logs"
os.makedirs(command_logs_dir, exist_ok=True)

# Where to store all commands
commands_to_run = []

# comment out the constellations you do not want to generate graphs for
for constellation_id in [
    "starlink_current_5shells_isls_three_ground_stations_starlink_cells_southafrica_0_1000_uniform",
    "starlink_current_5shells_isls_three_ground_stations_starlink_cells_ghana_0_1000_uniform",
    "starlink_current_5shells_isls_three_ground_stations_starlink_cells_tonga_0_1000_uniform",
    "starlink_current_5shells_isls_three_ground_stations_starlink_cells_lithuania_0_1000_uniform",
    "starlink_current_5shells_isls_three_ground_stations_starlink_cells_britain_0_1000_uniform",
    "starlink_current_5shells_isls_three_ground_stations_starlink_cells_haiti_0_1000_uniform",
    "starlink_double_7shells_isls_three_ground_stations_starlink_cells_southafrica_0_1000_population",
    "starlink_double_7shells_isls_three_ground_stations_starlink_cells_ghana_0_1000_population",
    "starlink_double_7shells_isls_three_ground_stations_starlink_cells_tonga_0_1000_population",
    "starlink_double_7shells_isls_three_ground_stations_starlink_cells_lithuania_0_1000_population",
    "starlink_double_7shells_isls_three_ground_stations_starlink_cells_britain_0_1000_population",
    "starlink_double_7shells_isls_three_ground_stations_starlink_cells_haiti_0_1000_population",
    "starlink_all_13shells_isls_three_ground_stations_starlink_cells_southafrica_0_1000_population",
    "starlink_all_13shells_isls_three_ground_stations_starlink_cells_ghana_0_1000_population",
    "starlink_all_13shells_isls_three_ground_stations_starlink_cells_tonga_0_1000_population",
    "starlink_all_13shells_isls_three_ground_stations_starlink_cells_lithuania_0_1000_population",
    "starlink_all_13shells_isls_three_ground_stations_starlink_cells_britain_0_1000_population",
    "starlink_all_13shells_isls_three_ground_stations_starlink_cells_haiti_0_1000_population",
]:
    update_interval_ms = 1000
    duration_s = 15 # 5 minutes

    interval = 1
    # interval = 2
    for start_time in range(0, 0 + duration_s, interval):
        _ = resolve_constellation_dir(constellation_id)
        country = extract_country(constellation_id)

        log_file = (
            command_logs_dir
            / f"generate_graphs_digraphs_{constellation_id}_{update_interval_ms}ms_for_{duration_s}s_{start_time}.log"
        )

        command = (
            f"cd {shlex.quote(str(PROJECT_ROOT))}; "
            f"time python -m graph_generation.helpers.generate_directed_graphs "
            f"{shlex.quote(str(DIGRAPHS_DIR))} "
            f"{shlex.quote(country)} "
            f"{shlex.quote(constellation_id)} "
            f"{update_interval_ms} {start_time} {start_time + interval} "
            f" > {shlex.quote(str(log_file))} "
            f"2>&1"
        )

        print(command)
        commands_to_run.append(command)

# Run the commands
print("Running commands (at most %d in parallel)..." % max_num_processes)
for i in range(len(commands_to_run)):
    print("Starting command %d out of %d: %s" % (i + 1, len(commands_to_run), commands_to_run[i]))
    local_shell.detached_exec(commands_to_run[i])
    while local_shell.count_screens() >= max_num_processes:
        time.sleep(2)

# Awaiting final completion before exiting
print("Waiting completion of the last %d..." % max_num_processes)
while local_shell.count_screens() > 0:
    time.sleep(2)
print("Finished.")
