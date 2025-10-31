import os
import shlex
import time
from pathlib import Path

import exputil

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASE_DIR
DIGRAPHS_DIR = (BASE_DIR / "data" / "digraphs").resolve()
DIGRAPH_DATA_DIR = (BASE_DIR / "data" / "traffic_engineering" / "digraph_data_imc").resolve()
CONSTELLATION_OUTPUT_DIR = (BASE_DIR / "constellation_configs" / "gen_data").resolve()

local_shell = exputil.LocalShell()
max_num_processes = 400

# Create digraph_data_imc directory if not present already
os.makedirs(DIGRAPH_DATA_DIR, exist_ok=True)

command_logs_dir = DIGRAPH_DATA_DIR / "command_logs"
os.makedirs(command_logs_dir, exist_ok=True)


def enqueue_command(
    constellation_config_id: str,
    graph_path: str,
    update_interval_ms: int,
    start_time: int,
    end_time: int,
    beam_allocation: str,
    routing_policy: str,
    comp_priority: str,
    incumbent_demand: float,
):
    """Add a TE comparison command while validating dependencies."""
    constellation_dir = CONSTELLATION_OUTPUT_DIR / constellation_config_id
    if not constellation_dir.exists():
        raise FileNotFoundError(
            f"Constellation output not found: {constellation_dir}. "
            "Generate configurations in `constellation_configs/gen_data` first."
        )

    graph_dir = DIGRAPHS_DIR / graph_path / "1000ms"
    if not graph_dir.exists():
        raise FileNotFoundError(
            f"Directed graph snapshots not found: {graph_dir}. "
            "Run `graph_generation/generate_digraphs.py` before running directed TE comparisons."
        )

    log_file = (
        command_logs_dir
        / f"comparison_{constellation_config_id}_{start_time}_{routing_policy}_{beam_allocation}_{comp_priority}_{incumbent_demand}.log"
    )

    command = (
        f"cd {shlex.quote(str(PROJECT_ROOT))}; "
        f"python -m traffic_engineering.utils.compare_te_priorities_digraph "
        f"{shlex.quote(str(DIGRAPH_DATA_DIR))} {shlex.quote(str(graph_dir))} "
        f"{shlex.quote(str(constellation_dir))} "
        f"{update_interval_ms} {start_time} {end_time} "
        f"{routing_policy} {beam_allocation} {comp_priority} {incumbent_demand} "
        f"> {shlex.quote(str(log_file))} 2>&1"
    )

    commands_to_run.append(command)

# Where to store all commands
commands_to_run = []

# this value should correspond to the number of cores
num_shells = 1 # Ensure this value is thread friendly

countries = ["ghana", "tonga", "lithuania", "britain", "haiti", "southafrica"]
# countries = []

pop_values = {
    "southafrica": [1000,2000,5000,10000,20000,50000,100000,200000,500000],
    "ghana": [1000,2000,5000,10000,20000,50000,100000,200000,500000],
    "tonga": [100,200,500,1000,2000,5000,10000,20000,50000],
    "lithuania": [1000,2000,5000,10000,20000,50000,100000,200000,500000],
    "haiti": [1000,2000,5000,10000,20000,50000,100000,200000,500000],
    "britain": [1000,2000,5000,10000,20000,50000,100000,200000,500000]
}

pop_values = {
    "southafrica": [20000,50000,100000,200000],
    "ghana": [10000,100000],
    "tonga": [500,1000],
    "lithuania": [10000,50000,1000],
    "haiti": [5000,100000],
    "britain": [50000,100000,200000],
}


isls = ["isls_three", "isls_plus_grid"]

# ut_distribution = ["population", "waterfill"]
beam_allocations = ["popwaterfill"]
comparison_priorities = ["equal", "incumbent", "emergency"]
incumbent_demands = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
constellations = ["starlink_current_5shells", "starlink_double_7shells", "starlink_all_13shells"]
routing_policies = ["max_flow", "hot_potato"]

# countries = ["southafrica"]
# countries = ["southafrica", "ghana", "tonga", "lithuania", "britain", "haiti"]
ut_distribution = ["waterfill_variant"]

pop_limits = {
    "southafrica": [10000, 100000],
    "ghana": [10000, 100000],
    "tonga": [1000, 10000],
    "lithuania": [1000, 10000, 100000],
    "britain": [10000, 100000],
    "haiti": [10000, 100000]
}

# pop_limits = {
#     "southafrica": [10000],
#     "ghana": [10000],
#     "tonga": [1000],
#     "lithuania": [1000],
#     "britain": [10000],
#     "haiti": [10000]
# }

for constellation in constellations[:1]:
    for isl in isls[:1]:
        for country in countries:
            for pop in pop_values[country]:
                    for ut_dist in ut_distribution:
                        for pop_limit in pop_limits[country]:
                        
                            constellation_config_id = f"{constellation}_{isl}_ground_stations_starlink_cells_{country}_0_{pop}_{ut_dist}_{pop_limit}_1.28"
                            update_interval_ms = 1000
                            duration_s = 15
                            graph_path = f"{constellation}_{isl}_ground_stations_starlink_cells_{country}_0"

                            total_duration = 15
                    
                            for start_time in range(0, 0 + duration_s, total_duration):
                                for beam_allocation in beam_allocations:
                                    for routing_policy in routing_policies[:1]:
                                        for comp_priority in comparison_priorities[1:]:
                                            for incumbent_demand in incumbent_demands:
                                                enqueue_command(
                                                    constellation_config_id,
                                                    graph_path,
                                                    update_interval_ms,
                                                    start_time,
                                                    start_time + duration_s,
                                                    beam_allocation,
                                                    routing_policy,
                                                    comp_priority,
                                                    incumbent_demand,
                                                )

# commands_to_run = commands_to_run[:1]
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
