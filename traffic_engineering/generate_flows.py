import os
import shlex
import time
from pathlib import Path

import exputil

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASE_DIR
GRAPHS_DIR = (BASE_DIR / "data" / "graphs").resolve()
SIGCOMM_DATA_DIR = (BASE_DIR / "data" / "traffic_engineering" / "sigcomm_data").resolve()
CONSTELLATION_OUTPUT_DIR = (BASE_DIR / "constellation_configs" / "gen_data").resolve()

local_shell = exputil.LocalShell()
max_num_processes = 300

# Create sigcomm_data directory if not present already
os.makedirs(SIGCOMM_DATA_DIR, exist_ok=True)

command_logs_dir = SIGCOMM_DATA_DIR / "command_logs"
os.makedirs(command_logs_dir, exist_ok=True)

# Where to store all commands
commands_to_run = []

# this value should correspond to the number of cores
num_shells = 1 # Ensure this value is thread friendly

countries = ["southafrica", "ghana", "tonga", "lithuania", "britain", "haiti"]
pop_values = {
    "southafrica": [1000,2000,5000,10000,20000,50000,100000,200000,500000],
    "ghana": [1000,2000,5000,10000,20000,50000,100000,200000,500000],
    "tonga": [100,200,500,1000,2000,5000,10000,20000,50000],
    "lithuania": [1000,2000,5000,10000,20000,50000,100000,200000,500000],
    "haiti": [1000,2000,5000,10000,20000,50000,100000,200000,500000],
    "britain": [1000,2000,5000,10000,20000,50000,100000,200000,500000]
}

pop_values = {
    "southafrica": [200000, 500000, 1000000],
    "ghana": [100000, 200000, 500000],
    "tonga": [1000, 2000, 5000],
    "lithuania": [50000, 100000, 200000],
    "britain": [100000, 200000, 500000],
    "haiti": [50000, 100000, 200000],
}

# pop_values = {
#     "southafrica": [200000],
#     "ghana": [100000],
#     "tonga": [1000]
# }



countries = []

isls = ["isls_three", "isls_plus_grid"]

ut_distribution = ["uniform", "population", "waterfill"]
beam_allocations = ["priority", "waterfill"]
constellations = ["starlink_current_5shells", "starlink_double_7shells", "starlink_all_13shells"]

for constellation in constellations:
    for isl in isls[:1]:
        for country in countries:
            for pop in pop_values[country]:
                for ut_dist in ut_distribution:
                    constellation_config_id = f"{constellation}_{isl}_ground_stations_starlink_cells_{country}_0_{pop}_{ut_dist}"
                    duration_s = 15
                    print(constellation_config_id)
                    graph_path = "_".join(constellation_config_id.split("_")[:-2])

                    total_duration = 15
                    # interval = 2
                    for start_time in range(0, 0 + duration_s, total_duration):
                        for beam_allocation in beam_allocations:
                            constellation_dir = CONSTELLATION_OUTPUT_DIR / constellation_config_id
                            if not constellation_dir.exists():
                                raise FileNotFoundError(
                                    f"Constellation output not found: {constellation_dir}. "
                                    "Generate configurations in `constellation_configs/gen_data` first."
                                )

                            graph_dir = GRAPHS_DIR / graph_path / "1000ms"
                            if not graph_dir.exists():
                                raise FileNotFoundError(
                                    f"Graph snapshots not found: {graph_dir}. "
                                    "Run the graph-generation step before generating flows."
                                )

                            log_file = (
                                command_logs_dir
                                / f"demands_{constellation_config_id}_for_{start_time}_{beam_allocation}.log"
                            )

                            commands_to_run.append(
                                "cd {project_root}; "
                                "python -m spectrum_management.utils.generate_flows "
                                "{sigcomm} {graphs} {constellation} {start} {beam_allocation}"
                                " > {log} 2>&1".format(
                                    project_root=shlex.quote(str(PROJECT_ROOT)),
                                    sigcomm=shlex.quote(str(SIGCOMM_DATA_DIR)),
                                    graphs=shlex.quote(str(graph_dir)),
                                    constellation=shlex.quote(str(constellation_dir)),
                                    start=start_time,
                                    beam_allocation=beam_allocation,
                                    log=shlex.quote(str(log_file)),
                                )
                            )

pop_limits = {
    "southafrica": [1000, 10000, 100000],
    "ghana": [1000, 10000, 100000],
    "tonga": [1000, 10000],
    "lithuania": [1000, 10000, 100000],
    "britain": [1000, 10000, 100000],
    "haiti": [1000, 10000, 100000]
}

# pop_limits = {
#     "southafrica": [10000],
#     "ghana": [10000],
#     "tonga": [1000],
#     "lithuania": [1000],
#     "britain": [10000],
#     "haiti": [10000]
# }

countries = ["britain"]
countries = ["southafrica", "ghana", "tonga", "lithuania", "britain", "haiti"]

ut_distribution = ["waterfill_variant", "population_variant"]

for constellation in constellations:
    for isl in isls[:1]:
        for country in countries:
            for pop in pop_values[country]:
                for ut_dist in ut_distribution[:1]:
                    for pop_limit in pop_limits[country]:
                        constellation_config_id = f"{constellation}_{isl}_ground_stations_starlink_cells_{country}_0_{pop}_{ut_dist}_{pop_limit}"
                        duration_s = 15
                        print(constellation_config_id)
                        graph_path = "_".join(constellation_config_id.split("_")[:-4])

                        total_duration = 15
                        # interval = 2
                        for start_time in range(0, 0 + duration_s, total_duration):
                            for beam_allocation in beam_allocations[1:]:
                                constellation_dir = CONSTELLATION_OUTPUT_DIR / constellation_config_id
                                if not constellation_dir.exists():
                                    raise FileNotFoundError(
                                        f"Constellation output not found: {constellation_dir}. "
                                        "Generate configurations in `constellation_configs/gen_data` first."
                                    )

                                graph_dir = GRAPHS_DIR / graph_path / "1000ms"
                                if not graph_dir.exists():
                                    raise FileNotFoundError(
                                        f"Graph snapshots not found: {graph_dir}. "
                                        "Run the graph-generation step before generating flows."
                                    )

                                log_file = (
                                    command_logs_dir
                                    / f"demands_{constellation_config_id}_for_{start_time}_{beam_allocation}.log"
                                )

                                commands_to_run.append(
                                    "cd {project_root}; "
                                    "python -m spectrum_management.utils.generate_flows "
                                    "{sigcomm} {graphs} {constellation} {start} {beam_allocation}"
                                    " > {log} 2>&1".format(
                                        project_root=shlex.quote(str(PROJECT_ROOT)),
                                        sigcomm=shlex.quote(str(SIGCOMM_DATA_DIR)),
                                        graphs=shlex.quote(str(graph_dir)),
                                        constellation=shlex.quote(str(constellation_dir)),
                                        start=start_time,
                                        beam_allocation=beam_allocation,
                                        log=shlex.quote(str(log_file)),
                                    )
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
