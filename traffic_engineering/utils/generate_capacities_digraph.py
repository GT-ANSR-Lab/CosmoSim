import json
import math
from .common import calculate_rtts
from .graph_tools import *
from utils.isls import *
from utils.ground_stations import *
from utils.tles import *
from spectrum_management.utils.beam_mapping import *
from traffic_engineering.utils.routing import *
import utils.global_variables as global_vars
import networkx as nx
from networkx.algorithms.flow import max_flow_min_cost
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
        graph.add_edge(dummy_node, start_node, capacity=int(demands[sat]), weight=0)
        graph.add_edge(sat, dummy_node, capacity=int(demands[sat]), weight=0)

    return graph

def max_flow_calculator(demands, graph, source, sink, num_satellites):
    
    graph = assign_sat_capacity(graph, demands, source, num_satellites)
    total_supply = sum(data['capacity'] for u, v, data in graph.edges(data=True) if u == 'T')

    # Calculate total demand
    total_demand = sum(data['capacity'] for u, v, data in graph.edges(data=True) if v == 'S')

    print(f"Total Supply: {total_supply}, Total Demand: {total_demand}")
    print(nx.maximum_flow_value(graph, sink, source, capacity="capacity"))
    flow_dict = max_flow_min_cost(graph, sink, source)
    # max_flow, flow_dict = nx.maximum_flow(graph, source, sink)
    
    return flow_dict

def hot_potato_calculator(demands, graph, source, sink, num_satellites, num_groundstations):
    graph = hot_potato_modifications(graph, demands, num_groundstations, num_satellites)
    graph = assign_sat_capacity(graph, demands, source, num_satellites)

    print(nx.shortest_path(graph, sink, source))

    # for u, v, data in graph.edges(data=True):
    #     if data.get('capacity') > 0:
    #         print(f"Edge ({u}, {v}) capacity: {data.get('capacity', float('inf'))}")

    # for node, data in graph.nodes(data=True):
    #     print(f"Node {node}, Attributes: {data}")
    
    total_supply = sum(data['capacity'] for u, v, data in graph.edges(data=True) if u == 'T')

    # Calculate total demand
    total_demand = sum(data['capacity'] for u, v, data in graph.edges(data=True) if v == 'S')

    print(f"Total Supply: {total_supply}, Total Demand: {total_demand}")
    

    flow_value, flow_dict = nx.maximum_flow(graph, sink, source)
    print(flow_value)
    print(flow_dict)
    
    return flow_dict




def generate_capacities_digraph(data_dir, demands_directory, graph_dir, satellite_network_dir, 
                                    dynamic_state_update_interval_ms, simulation_start_time_s, simulation_end_time_s, 
                                    routing_policy, ku_band_capacity):
    
    ground_stations = read_ground_stations_extended(satellite_network_dir + "/ground_stations.txt")
    
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

    simulation_start_time_ns = simulation_start_time_s * 1000 * 1000 * 1000
    simulation_end_time_ns = simulation_end_time_s * 1000 * 1000 * 1000
    dynamic_state_update_interval_ns = dynamic_state_update_interval_ms * 1000 * 1000

    demands_filename = f"{demands_directory}/demands_{simulation_start_time_s}.txt"
    flow_filename = f"{data_dir}/{routing_policy}_{simulation_start_time_s}.txt"
    flow_dict_filename = data_dir + "/flow_dict_" + str(simulation_start_time_s) + ".txt"

    open(flow_filename, 'w').close()


    gs_base_index = len(satellites)
    ut_base_index = len(satellites) + len(ground_stations)

    subgraph_num_nodes = len(ground_stations) + len(satellites)
    subgraph_nodes = list(range(subgraph_num_nodes))

    for t in range(simulation_start_time_ns, simulation_end_time_ns, dynamic_state_update_interval_ns):
        time_seconds = t / 1000 / 1000 / 1000
        print(f"Time: {time_seconds}s")

        # Load graph
        graph_path = graph_dir + "/graph_" + str(t) + ".txt"
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)

        # Initialize capacity dictionaries
        graph = nx.DiGraph(graph.subgraph(subgraph_nodes))
        # total_demand = sum(data['demand'] for _, data in graph.nodes(data=True))
        # print("Total demand:", total_demand)

        ground_station_capacity_used = {}
        for gs_index in range(len(ground_stations)):
            ground_station_capacity_used[gs_index + gs_base_index] = 0

        sat_capacity = {}
        for sat in range(len(satellites)):
            sat_capacity[sat] = 0

        demands = np.genfromtxt(demands_filename, delimiter=",")

        non_zero_demand_sats = np.nonzero(demands)[0]
        demand_per_shell = [0] * len(shell_satellite_indices)
        for sat in non_zero_demand_sats:
            # shell_idx = np.searchsorted(shell_satellite_indices, sat)
            print(sat)
            sat_neighbors = graph.predecessors(sat)
            sat_gs_neighbors = [n for n in sat_neighbors if isinstance(n, int) and n >= gs_base_index and n < ut_base_index]
            if len(sat_gs_neighbors) > 0:
                sat_gs_neighbors = sorted(sat_gs_neighbors, key=lambda x: ground_station_capacity_used[x])
                for gs_neighbor in sat_gs_neighbors:
                    if sat_capacity[sat] < global_vars.ground_station_sat_capacity and ground_station_capacity_used[gs_neighbor] < global_vars.ground_station_gsl_capacity:
                        ka_beams_required = min(math.floor(demands[sat] / global_vars.ka_beam_capacity), 8 - (sat_capacity[sat] // global_vars.ka_beam_capacity))
                        gs_capacity = min(global_vars.ground_station_gsl_capacity - ground_station_capacity_used[gs_neighbor], ka_beams_required * global_vars.ka_beam_capacity)
                        gs_capacity = (gs_capacity // global_vars.ka_beam_capacity) * global_vars.ka_beam_capacity
                        graph[gs_neighbor][sat]['capacity'] += int(gs_capacity)
                        ground_station_capacity_used[gs_neighbor] += gs_capacity
                        sat_capacity[sat] += gs_capacity

        shell_priority_order = list(range(1584, len(satellites))) + list(range(1584))
        for sat in shell_priority_order:
            if sat_capacity[sat] < global_vars.sat_gs_max_capacity:
                sat_neighbors = graph.predecessors(sat)
                sat_gs_neighbors = [n for n in sat_neighbors if isinstance(n, int) and n >= gs_base_index and n < ut_base_index]
                if len(sat_gs_neighbors) > 0:
                    sat_gs_neighbors = sorted(sat_gs_neighbors, key=lambda x: ground_station_capacity_used[x])
                    for gs_neighbor in sat_gs_neighbors:
                        if sat_capacity[sat] < global_vars.sat_gs_max_capacity:
                            gs_capacity = min(global_vars.sat_gs_max_capacity - sat_capacity[sat], global_vars.ground_station_gsl_capacity - ground_station_capacity_used[gs_neighbor])
                            graph[gs_neighbor][sat]['capacity'] += int(gs_capacity)
                            sat_capacity[sat] += gs_capacity
                            ground_station_capacity_used[gs_neighbor] += gs_capacity


        for sat in range(len(satellites)):
            sat_neighbors = graph.predecessors(sat)
            sat_gs_neighbors = [n for n in sat_neighbors if isinstance(n, int) and n >= gs_base_index and n < ut_base_index]
            for gs_neighbor in sat_gs_neighbors:
                if graph[gs_neighbor][sat]['capacity'] == 0:
                    graph.remove_edge(gs_neighbor, sat)

        if ku_band_capacity == 2.5:
            for sat in range(len(satellites)):
                sat_neighbors = graph.predecessors(sat)
                sat_neighbors = [n for n in sat_neighbors if isinstance(n, int) and n < gs_base_index]
                for sat_neighbor in sat_neighbors:
                    graph[sat_neighbor][sat]['capacity'] = global_vars.isl_capacity * 2


        source = "S"
        sink = "T"
        graph.add_node(source)
        graph.add_node(sink)
        
        for gs in ground_stations:
            graph.add_edge(sink, len(satellites) + gs["gid"], capacity=global_vars.ground_station_gsl_capacity, weight=0)

        if routing_policy == "max_flow":
            flow_dict = max_flow_calculator(demands, graph, source, sink, len(satellites))
            print(flow_dict)
            capacity = sum(flow_dict[sink].values())
            
            
        elif routing_policy == "hot_potato":
            flow_dict = hot_potato_calculator(demands, graph, source, sink, len(satellites), len(ground_stations))
            print(flow_dict)
            capacity = sum(flow_dict[sink].values())

        with open(flow_dict_filename, "w") as first_pass_flow_dict_file:
            json.dump(flow_dict, first_pass_flow_dict_file)
        
        with open(flow_filename, "a+") as flow_file:
            flow_file.write(str(t) + "," + str(capacity) + "\n") 

def main():
    args = sys.argv[1:]
    if len(args) == 9:
        config = args[2].split("/")[-1]
        base_output_dir = f"{args[0]}/{config}_{args[7]}/{args[6]}"

        demands_directory = f"{args[0]}/{config}_{args[7]}"


        local_shell = exputil.LocalShell()
        local_shell.make_full_dir(base_output_dir)

        generate_capacities_digraph(base_output_dir,
                            demands_directory,
                            args[1],        # graph_dir
                            args[2],        # satellite_network_dir
                            int(args[3]),   # dynamic state update interval ms
                            int(args[4]),   # simulation start time s
                            int(args[5]),   # simulation end time s
                            args[6],        # routing algorithm
                            float(args[8])) # ku band capacity
    else:
        print("Invalid argument selection for generate_capacities_digraph.py")
        print("Usage: python generate_capacities_digraph.py <base_output_dir> <graph_dir> <satellite_network_dir> <dynamic_state_update_interval_ms> <simulation_start_time_s> <simulation_end_time_s> <routing_policy> <beam_allocation>")

if __name__ == "__main__":
    main()