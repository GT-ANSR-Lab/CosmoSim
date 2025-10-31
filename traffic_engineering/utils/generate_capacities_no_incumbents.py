from .common import calculate_rtts
from .graph_tools import *
from utils.isls import *
from utils.ground_stations import *
from utils.tles import *
from spectrum_management.utils.beam_mapping import *
from traffic_engineering.utils.routing import *
from utils.global_variables import *
import networkx as nx
from utils.distance_tools import *
from utils.user_terminals import *
import random
import sys
import exputil
import pickle
import numpy as np


def assign_sat_capacity(graph, demands, start_node, num_satellites):
    for sat in range(num_satellites):
        dummy_node = "D" + str(sat)
        graph.add_edge(start_node, dummy_node, capacity=demands[sat], weight=0)
        graph.add_edge(dummy_node, sat, capacity=demands[sat], weight=0)

    return graph

def max_flow_calculator(demands, graph, source, sink, num_satellites):
    
    graph = assign_sat_capacity(graph, demands, source, num_satellites)
    max_flow, flow_dict = nx.maximum_flow(graph, source, sink)
    
    return max_flow, flow_dict

def hot_potato_calculator(demands, graph, source, sink, num_satellites, num_groundstations):
    graph = hot_potato_modifications(graph, demands, num_groundstations, num_satellites)
    graph = assign_sat_capacity(graph, demands, source, num_satellites)
    
    max_flow, flow_dict = nx.maximum_flow(graph, source, sink)
    
    return max_flow, flow_dict


def generate_capacities_no_incumbents(data_dir, demands_directory, graph_dir, satellite_network_dir, 
                                    dynamic_state_update_interval_ms, simulation_start_time_s, simulation_end_time_s, routing_policy):
    
    ground_stations = read_ground_stations_extended(satellite_network_dir + "/ground_stations.txt")
    
    tles = read_tles(satellite_network_dir + "/tles.txt")
    satellites = tles["satellites"]
    epoch = tles["epoch"]

    simulation_start_time_ns = simulation_start_time_s * 1000 * 1000 * 1000
    simulation_end_time_ns = simulation_end_time_s * 1000 * 1000 * 1000
    dynamic_state_update_interval_ns = dynamic_state_update_interval_ms * 1000 * 1000

    global_schedule_counter = 0

    demands_filename = f"{demands_directory}/demands_{simulation_start_time_s}.txt"
    flow_filename = f"{data_dir}/{routing_policy}_{simulation_start_time_s}.txt"

    open(flow_filename, 'w').close()

    gs_base_index = len(satellites)
    ut_base_index = len(satellites) + len(ground_stations)

    subgraph_num_nodes = len(ground_stations) + len(satellites)
    subgraph_nodes = list(range(subgraph_num_nodes))

    for t in range(simulation_start_time_ns, simulation_end_time_ns, dynamic_state_update_interval_ns):
        time_seconds = t / 1000 / 1000 / 1000
        print(f"Time: {time_seconds}s")
        global_schedule_counter = time_seconds % satellite_handoff_seconds

        # Load graph
        graph_path = graph_dir + "/graph_" + str(t) + ".txt"
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)

        # Initialize capacity dictionaries
        graph = nx.Graph(graph.subgraph(subgraph_nodes))
        
        ground_station_capacity_used = {}
        for gs_index in range(len(ground_stations)):
            ground_station_capacity_used[gs_index] = 0

        
        sat_capacity = {}
        for sat in range(len(satellites)):
            sat_capacity[sat] = 0

        for sat in range(len(satellites)):
            # get gs neighbors -- id between gs_base_index and ut_base_index
            if sat in graph.nodes:
                sat_neighbor_ids = list(graph.neighbors(sat))
                sat_gs_neighbors = [neighbor_id for neighbor_id in sat_neighbor_ids if isinstance(neighbor_id, int) and neighbor_id >= gs_base_index and neighbor_id < ut_base_index]
                # sat_gs_neighbors = [neighbor for neighbor in sat_gs_neighbors if ground_station_capacity_used[neighbor - gs_base_index] < ground_station_gsl_capacity] 
                if len(sat_gs_neighbors) > 0:
                    sat_gs_neighbors = sorted(sat_gs_neighbors, key=lambda x: ground_station_capacity_used[x - gs_base_index])
                    # distributed sat_gs_max_capacity among all ground stations but ensuring that they don't exceed ground_station_gsl_capacity
                    
                    for i, gs in enumerate(sat_gs_neighbors):
                        sat_capacity_left = sat_gs_max_capacity - sat_capacity[sat]
                        gs_capacity = min(ground_station_gsl_capacity - ground_station_capacity_used[gs - gs_base_index], sat_capacity_left)
                        graph[sat][gs]['capacity'] = gs_capacity
                        ground_station_capacity_used[gs - gs_base_index] += gs_capacity
                        sat_capacity[sat] += gs_capacity

        for sat in range(len(satellites)):
            if sat in graph.nodes:
                sat_neighbor_ids = list(graph.neighbors(sat))
                sat_isl_neighbors = [neighbor_id for neighbor_id in sat_neighbor_ids if isinstance(neighbor_id, int) and neighbor_id < gs_base_index]

                for neighbor in sat_isl_neighbors:
                    sat_capacity[sat] += graph[sat][neighbor]['capacity']

                sat_capacity[sat] = min(sat_capacity[sat], sat_gs_max_capacity)

        demands = np.genfromtxt(demands_filename, delimiter=",")
        source = "S"
        sink = "T"
        graph.add_node(source)
        graph.add_node(sink)
        
        for gs in ground_stations:
            graph.add_edge(len(satellites) + gs["gid"], sink, capacity=ground_station_gsl_capacity, weight=0)

        if routing_policy == "max_flow":
            capacity, _ = max_flow_calculator(demands, graph, source, sink, len(satellites))
        elif routing_policy == "hot_potato":
            capacity, _ = hot_potato_calculator(demands, graph, source, sink, len(satellites), len(ground_stations))
        
        with open(flow_filename, "a+") as flow_file:
            flow_file.write(str(t) + "," + str(capacity) + "\n") 

def main():
    args = sys.argv[1:]
    if len(args) == 8:
        config = args[2].split("/")[-1]
        base_output_dir = f"{args[0]}/{config}_{args[7]}/{args[6]}"

        demands_directory = f"{args[0]}/{config}_{args[7]}"


        local_shell = exputil.LocalShell()
        local_shell.make_full_dir(base_output_dir)

        generate_capacities_no_incumbents(base_output_dir,
                            demands_directory,
                            args[1],        # graph_dir
                            args[2],        # satellite_network_dir
                            int(args[3]),   # dynamic state update interval ms
                            int(args[4]),   # simulation start time s
                            int(args[5]),   # simulation end time s
                            args[6])        # routing algorithm
    else:
        print("Invalid argument selection for generate_capacities_no_incumbents.py")
        print("Usage: python generate_capacities_no_incumbents.py <base_output_dir> <graph_dir> <satellite_network_dir> <dynamic_state_update_interval_ms> <simulation_start_time_s> <simulation_end_time_s> <routing_policy> <beam_allocation>")

if __name__ == "__main__":
    main()