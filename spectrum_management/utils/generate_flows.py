from .utils import calculate_rtts
from .graph_tools import *
from utils.isls import *
from utils.ground_stations import *
from utils.tles import *
from spectrum_management.utils.beam_mapping import *
from traffic_engineering.utils.routing import *
import utils.global_variables as global_vars
import networkx as nx
from utils.distance_tools import *
from utils.cells import *
import random
import math
import sys
import exputil
import pickle
import json

def generate_flows_digraphs(data_dir, graph_dir, satellite_network_dir, flow_time_s, 
                   beam_allocation, ku_band_capacity, config):
    
    ground_stations = read_ground_stations_extended(satellite_network_dir + "/ground_stations.txt")
    cells = read_cells(satellite_network_dir + "/cells.txt")

    # user_terminals = user_terminals[:num_uts]

    tles = read_tles(satellite_network_dir + "/tles.txt")
    satellites = tles["satellites"]
    epoch = tles["epoch"]
    description = exputil.PropertiesConfig(satellite_network_dir + "/description.txt")
    num_orbits = json.loads(description.get_property_or_fail("num_orbits"))
    num_sats_per_orbit = json.loads(description.get_property_or_fail("num_sats_per_orbit"))


    shell_satellite_indices = []
    count = 0
    for idx in range(len(num_orbits)):
        shell_satellite_indices.append((count, count + num_sats_per_orbit[idx] * num_orbits[idx]))
        count += num_sats_per_orbit[idx] * num_orbits[idx]

    print(shell_satellite_indices)
    shell_satellite_indices = shell_satellite_indices[1:] + [shell_satellite_indices[0]]
    print(shell_satellite_indices)

    demands_filename = data_dir + "/demands_" + str(flow_time_s) + ".txt"
    # rtt_filename = data_dir + "/rtt_" + str(simulation_start_time_s) + ".txt"

    open(demands_filename, 'w').close()
    # open(rtt_filename, 'w').close()

    # last_config_part = config.split("_")[-1]
    # if last_config_part == "1.28" or last_config_part == "0.956":
    #     ku_band_capacity = float(last_config_part)
    # else:
    #     ku_band_capacity = 1.28

    users_per_channel = math.floor(ku_band_capacity * 10)

    gs_base_index = len(satellites)
    ut_base_index = len(satellites) + len(ground_stations)

    cell_ut_mapping = {}
    cell_ch_satellite = {}

    t = flow_time_s * 1000 * 1000 * 10000
    print(f"Time: {flow_time_s}s")
    global_schedule_counter = flow_time_s % global_vars.satellite_handoff_seconds

    # Load graph
    graph_path = graph_dir + "/graph_" + str(t) + ".txt"
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    sat_demands = {}
    for sat in range(len(satellites)):
        sat_demands[sat] = 0

    # sat_gsl_capacities = {}
    # sat_isl_capacities = {}
    # for sat in range(len(satellites)):
    #     if sat in graph.nodes:
    #         sat_gsl_capacities[sat] = graph.nodes[sat]["gsl_capacity"]
    #         sat_isl_capacities[sat] = graph.nodes[sat]["isl_capacity"]

    all_cell_satellites = set()
    cell_satellites = {}
    satellite_cells = {}

    # remove cells with no terminals
    cells = [cell_obj for cell_obj in cells if cell_obj["num_terminals"] > 0]

    for cell_obj in cells:
        cell = cell_obj["cell"]
        current_cell_satellites = []
        # print(graph.predecessors(cell))
        for sat in graph.predecessors(cell):
            sat_neighbor_ids = list(graph.predecessors(sat))
            # print(sat, sat_neighbor_ids)
            
            # only consider satellites that have an edge to either a ground station or another satellite
            if any([isinstance(neighbor_id, int) and neighbor_id < ut_base_index for neighbor_id in sat_neighbor_ids]):
                current_cell_satellites.append(sat)

        # print(cell, current_cell_satellites)
                
        all_cell_satellites = all_cell_satellites.union(current_cell_satellites)

        cell_satellites[cell] = current_cell_satellites
        for sat in current_cell_satellites:
            if sat not in satellite_cells:
                satellite_cells[sat] = []
            satellite_cells[sat].append(cell)

    print("Total number of satellites for region:", len(all_cell_satellites))

    mapping = beam_mapping(beam_allocation, cells, all_cell_satellites, satellite_cells, cell_satellites, config, shell_satellite_indices, users_per_channel)

    for cell_obj in cells:
        cell = cell_obj["cell"]
        cell_terminals = cell_obj["num_terminals"]
        cell_dummy_nodes = [cell + "_" + str(idx) for idx in range(8)]
        cell_dummy_nodes_mapped = [dummy_node for dummy_node in cell_dummy_nodes if dummy_node in mapping]

        if len(cell_dummy_nodes_mapped) == 0:
            continue

        num_channels_mapped = len(cell_dummy_nodes_mapped)

        ut_channel_allocations = [cell_terminals // num_channels_mapped + (1 if i < cell_terminals % num_channels_mapped else 0) for i in range(num_channels_mapped)]
        
        for i, channel in enumerate(cell_dummy_nodes_mapped):
            sat = int(mapping[channel].split("_")[1])
            cell_ch_satellite[channel] = sat
            channel_demand = min(ut_channel_allocations[i] * global_vars.user_terminal_gsl_capacity, ku_band_capacity * 1000)
            sat_demands[sat] += channel_demand

        # for sat in range(len(satellites)):
        #     sat_demands[sat] = min(sat_demands[sat], global_vars.sat_gs_max_capacity)
    
    with open(demands_filename, "a+") as demands_file:
        for sat in sat_demands:
            demands_file.write(str(sat_demands[sat]) + "\n")

def main():
    args = sys.argv[1:]
    if len(args) == 6:
        config = args[2].split("/")[-1]
        base_output_dir = f"{args[0]}/{config}_{args[4]}"
        
        print(base_output_dir, config)
        local_shell = exputil.LocalShell()
        local_shell.make_full_dir(base_output_dir)

        generate_flows_digraphs(base_output_dir,
                        args[1],        # graph_dir
                        args[2],        # satellite_network_dir
                        int(args[3]),   # flow time s
                        args[4],        # beam allocation
                        float(args[5]),  # ku_band_capacity
                        config)
    else:
        print("Invalid argument selection for generate_flows_digraphs.py")

if __name__ == "__main__":
    main()