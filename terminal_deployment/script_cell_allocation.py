import os
import subprocess
import time
from pathlib import Path

import exputil


SCRIPT_DIR = Path(__file__).resolve().parent
COMMAND_LOG_DIR = SCRIPT_DIR / "command_logs"
COMMAND_LOG_DIR.mkdir(exist_ok=True)


def _physical_core_count():
    try:
        output = subprocess.check_output(["lscpu"], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    threads_per_core = None
    total_cpus = None
    for line in output.splitlines():
        if line.startswith("CPU(s):"):
            try:
                total_cpus = int(line.split(":", 1)[1].strip())
            except ValueError:
                total_cpus = None
        elif line.startswith("Thread(s) per core"):
            try:
                threads_per_core = int(line.split(":", 1)[1].strip())
            except ValueError:
                threads_per_core = None
    if total_cpus and threads_per_core:
        return max(total_cpus // threads_per_core, 1)
    return None


local_shell = exputil.LocalShell()
logical_cpus = os.cpu_count() or 4
physical_cpus = _physical_core_count() or logical_cpus
max_num_processes = max(physical_cpus - 2, 1)

# Where to store all commands
commands_to_run = []

countries = ["southafrica", "ghana", "tonga", "lithuania", "britain", "haiti"]
# countries = ["southafrica", "britain"]
pop_values = {
    "southafrica": [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000],
    "ghana": [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000],
    "tonga": [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000],
    "lithuania": [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000],
    "britain": [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000],
    "haiti": [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000],
}

# Non-GCB distributions supported by the new CLI
standard_distributions = ["uniform", "population"]

# Ku-band capacity values to sweep over
ku_capacities = [0.956, 1.28, 2.5]

# Cap values for GCB cap mode
cap_values = [1000, 10000, 100000]

# Optional detail level override (None uses script default of 0)
detail_override = None  # e.g., set to "1" to run with --detail 1

def build_command(args, log_name):
    detail_flag = f" --detail {detail_override}" if detail_override is not None else ""
    return (
        f"python generate_cell_allocations.py"
        f"{detail_flag} {args} > {COMMAND_LOG_DIR / log_name} 2>&1"
    )


for country in countries:
    for pop in pop_values[country]:
        for dist in standard_distributions:
            log_file = f"{country}_{pop}_{dist}.log"
            cmd_args = f"{country} {pop} {dist}"
            commands_to_run.append(build_command(cmd_args, log_file))

        for ku_capacity in ku_capacities:
            # GCB no-cap: use cap value 0 as a placeholder (ignored by CLI)
            log_file = f"{country}_{pop}_gcb_no_cap_{ku_capacity}.log"
            cmd_args = f"{country} {pop} gcb no_cap 0 {ku_capacity}"
            commands_to_run.append(build_command(cmd_args, log_file))

            for cap in cap_values:
                log_file = f"{country}_{pop}_gcb_cap_{cap}_{ku_capacity}.log"
                cmd_args = f"{country} {pop} gcb cap {cap} {ku_capacity}"
                commands_to_run.append(build_command(cmd_args, log_file))

# Run the commands
print(f"Running commands (at most {max_num_processes} in parallel)...")
for i, command in enumerate(commands_to_run, start=1):
    print(f"Starting command {i} out of {len(commands_to_run)}: {command}")
    local_shell.detached_exec(command)
    while local_shell.count_screens() >= max_num_processes:
        time.sleep(2)

# Awaiting final completion before exiting
print(f"Waiting completion of the last {max_num_processes}...")
while local_shell.count_screens() > 0:
    time.sleep(2)
print("Finished.")
