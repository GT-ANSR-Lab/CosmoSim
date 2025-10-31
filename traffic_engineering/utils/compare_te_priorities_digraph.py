from utils.cells import *
from .graph_tools import *
from utils.isls import *
from utils.ground_stations import *
from utils.tles import *
from spectrum_management.utils.beam_mapping import *
from traffic_engineering.utils.routing import *
import utils.global_variables as global_variables
import networkx as nx
from networkx.algorithms.flow import max_flow_min_cost
from utils.distance_tools import *
from utils.user_terminals import *
import random
import sys
import exputil
import pickle
import numpy as np
import h3
import json
import math

def assign_sat_capacity(graph, demands, start_node, num_satellites):
    for sat in range(num_satellites):
        dummy_node = "D" + str(sat)
        graph.add_edge(dummy_node, start_node, capacity=demands[sat], weight=0)
        graph.add_edge(sat, dummy_node, capacity=demands[sat], weight=0)

    return graph

def remove_dummy_nodes(graph, num_satellites):
    for sat in range(num_satellites):
        dummy_node = "D" + str(sat)
        if dummy_node in graph.nodes:
            graph.remove_node(dummy_node)

    return graph

def assign_sat_capacity_equal(graph, demands, incumbent_satellites, incumbent_demand, start_node, num_satellites):
    for sat in range(num_satellites):
        incumbent_node = "I" + str(sat)
        emergency_node = "E" + str(sat)
        if sat in incumbent_satellites:
            graph.add_edge(incumbent_node, start_node, capacity=incumbent_demand, weight=0)
            graph.add_edge(sat, incumbent_node, capacity=incumbent_demand, weight=0)
        if demands[sat] > 0:
            graph.add_edge(emergency_node, start_node, capacity=demands[sat], weight=0)
            graph.add_edge(sat, emergency_node, capacity=demands[sat], weight=0)

    return graph

def emergency_traffic_max_flow(demands, graph, source, sink, num_satellites):    
    graph = assign_sat_capacity(graph, demands, source, num_satellites)
    flow_dict = max_flow_min_cost(graph, sink, source)
    
    return flow_dict

def incumbent_traffic_max_flow(graph, source, sink, num_satellites, incumbent_satellites, incumbent_demand):
    demands = np.zeros(num_satellites)
    # print("incumbent demand", incumbent_demand)
    # print("incumbent satellites", incumbent_satellites)
    demands[incumbent_satellites] = incumbent_demand
    
    graph = assign_sat_capacity(graph, demands, source, num_satellites)
    
    flow_dict = max_flow_min_cost(graph, sink, source)
    
    return flow_dict

def equal_priority_max_flow(demands, graph, source, sink, num_satellites, incumbent_satellites, incumbent_demand):
    
    graph = assign_sat_capacity_equal(graph, demands, incumbent_satellites, incumbent_demand, source, num_satellites)
    max_flow, flow_dict = nx.maximum_flow(graph, sink, source)
    
    return max_flow, flow_dict

def calculate_emergency_flows_capacity_allocation(flow_dict, demands, source, num_satellites):
    emergency_flows_capacity_allocated = [-1] * num_satellites
    for sat in range(num_satellites):
        if demands[sat] > 0:
            dummy_node = "D" + str(sat)
            emergency_flows_capacity_allocated[sat] = flow_dict[dummy_node][source] / demands[sat]
    
    return emergency_flows_capacity_allocated

def calculate_incumbent_flows_capacity_allocation(flow_dict, source, num_satellites, incumbent_satellites, incumbent_demand):
    incumbent_flows_capacity_allocated = [-1] * num_satellites
    for sat in incumbent_satellites:
        dummy_node = "D" + str(sat)
        incumbent_flows_capacity_allocated[sat] = flow_dict[dummy_node][source] / incumbent_demand
    
    return incumbent_flows_capacity_allocated

def compare_te_priorities_digraph(data_dir, demands_directory, graph_dir, satellite_network_dir, dynamic_state_update_interval_ms, simulation_start_time_s,
                         simulation_end_time_s, emergency_routing, priority, incumbent_demand):
    
    ground_stations = read_ground_stations_extended(satellite_network_dir + "/ground_stations.txt")
    cells = read_cells(satellite_network_dir + "/cells.txt")
    level3_cells = [h3.h3_to_parent(cell['cell'], 3) for cell in cells]
    cells = [cell['cell'] for cell in cells if cell["num_terminals"] > 0]
    
    level3_cells = [f"{cell}\n" for cell in level3_cells]
    level3_cells = list(set(level3_cells))
    print(len(level3_cells))
    print(level3_cells)

    starlink_cells = read_cells_starlink(satellite_network_dir + "/starlink_cells.txt")
    print(len(starlink_cells))
    starlink_cells = [cell['cell'] for cell in starlink_cells if cell['cell'] not in level3_cells]
    print(len(starlink_cells))
    print(starlink_cells)

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

    simulation_start_time_ns = simulation_start_time_s * 1000 * 1000 * 1000
    simulation_end_time_ns = simulation_end_time_s * 1000 * 1000 * 1000
    dynamic_state_update_interval_ns = dynamic_state_update_interval_ms * 1000 * 1000

    demands_filename = demands_directory + "/demands_" + str(simulation_start_time_s) + ".txt"
    flow_filename = data_dir + "/maxflow_" + str(incumbent_demand) + "_" + str(simulation_start_time_s) + ".txt"
    fulfillment_filename = data_dir + "/fulfillment_" + str(incumbent_demand) + "_" + str(simulation_start_time_s) + ".txt"
    # rtt_filename = data_dir + "/rtt_" + str(simulation_start_time_s) + ".txt"
    first_pass_flow_dict_filename = data_dir + "/first_pass_flow_dict_" + str(simulation_start_time_s) + ".txt"
    second_pass_flow_dict_filename = data_dir + "/second_pass_flow_dict_" + str(simulation_start_time_s) + ".txt"

    open(flow_filename, 'w').close()
    open(fulfillment_filename, 'w').close()
    # open(rtt_filename, 'w').close()

    gs_base_index = len(satellites)
    ut_base_index = len(satellites) + len(ground_stations)

    subgraph_num_nodes = len(ground_stations) + len(satellites)
    subgraph_nodes = list(range(subgraph_num_nodes))
    incumbent_demand = incumbent_demand * global_variables.incumbent_demand_multiplier

    for t in range(simulation_start_time_ns, simulation_end_time_ns, dynamic_state_update_interval_ns):
        time_seconds = t / 1000 / 1000 / 1000
        print(f"Time: {time_seconds}s")

        # Load graph
        graph_path = graph_dir + "/graph_" + str(t) + ".txt"
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)

        
        incumbent_satellites = []
        for sat in range(len(satellites)):
            if sat in graph.nodes:
                sat_neighbor_ids = list(graph.successors(sat))
                sat_neighbor_ids = [sat_neighbor.strip() for sat_neighbor in sat_neighbor_ids if not isinstance(sat_neighbor, int)]
                # print(sat, sat_neighbor_ids)
                for sat_neighbor in sat_neighbor_ids:
                    if sat_neighbor in starlink_cells:
                        incumbent_satellites.append(sat)
                        break
            
        print(incumbent_satellites)
        # Initialize capacity dictionaries
        graph = nx.DiGraph(graph.subgraph(subgraph_nodes))
        
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
            # print(sat)
            sat_neighbors = graph.predecessors(sat)
            sat_gs_neighbors = [n for n in sat_neighbors if isinstance(n, int) and n >= gs_base_index and n < ut_base_index]
            if len(sat_gs_neighbors) > 0:
                sat_gs_neighbors = sorted(sat_gs_neighbors, key=lambda x: ground_station_capacity_used[x])
                for gs_neighbor in sat_gs_neighbors:
                    if sat_capacity[sat] < global_variables.ground_station_sat_capacity and ground_station_capacity_used[gs_neighbor] < global_variables.ground_station_gsl_capacity:
                        ka_beams_required = min(math.floor(demands[sat] / global_variables.ka_beam_capacity), 8 - (sat_capacity[sat] // global_variables.ka_beam_capacity))
                        gs_capacity = min(global_variables.ground_station_gsl_capacity - ground_station_capacity_used[gs_neighbor], ka_beams_required * global_variables.ka_beam_capacity)
                        gs_capacity = (gs_capacity // global_variables.ka_beam_capacity) * global_variables.ka_beam_capacity
                        graph[gs_neighbor][sat]['capacity'] += int(gs_capacity)
                        ground_station_capacity_used[gs_neighbor] += gs_capacity
                        sat_capacity[sat] += gs_capacity

        shell_priority_order = list(range(1584, len(satellites))) + list(range(1584))
        for sat in shell_priority_order:
            if sat_capacity[sat] < global_variables.sat_gs_max_capacity:
                sat_neighbors = graph.predecessors(sat)
                sat_gs_neighbors = [n for n in sat_neighbors if isinstance(n, int) and n >= gs_base_index and n < ut_base_index]
                if len(sat_gs_neighbors) > 0:
                    sat_gs_neighbors = sorted(sat_gs_neighbors, key=lambda x: ground_station_capacity_used[x])
                    for gs_neighbor in sat_gs_neighbors:
                        if sat_capacity[sat] < global_variables.sat_gs_max_capacity:
                            gs_capacity = min(global_variables.sat_gs_max_capacity - sat_capacity[sat], global_variables.ground_station_gsl_capacity - ground_station_capacity_used[gs_neighbor])
                            graph[gs_neighbor][sat]['capacity'] += int(gs_capacity)
                            sat_capacity[sat] += gs_capacity
                            ground_station_capacity_used[gs_neighbor] += gs_capacity


        for sat in range(len(satellites)):
            sat_neighbors = graph.predecessors(sat)
            sat_gs_neighbors = [n for n in sat_neighbors if isinstance(n, int) and n >= gs_base_index and n < ut_base_index]
            for gs_neighbor in sat_gs_neighbors:
                if graph[gs_neighbor][sat]['capacity'] == 0:
                    graph.remove_edge(gs_neighbor, sat)


        source = "S"
        sink = "T"
        graph.add_node(source)
        graph.add_node(sink)
        
        for gs in ground_stations:
            graph.add_edge(sink, len(satellites) + gs["gid"], capacity=global_variables.ground_station_gsl_capacity, weight=0)

        max_flow_emergency, max_flow_incumbent = 0, 0

        
        demands = np.genfromtxt(demands_filename, delimiter=",")
        print("Total Demand", np.sum(demands))

        # if priority == "equal":
        #     max_flow, flow_dict = equal_priority_max_flow(demands, graph, source, sink, len(satellites), incumbent_satellites, incumbent_demand)
        #     incumbent_flows_capacity_allocated = [0] * len(satellites)
        #     emergency_flows_capacity_allocated = [0] * len(satellites)
        #     for sat in range(len(satellites)):
        #         if sat in graph.nodes:
        #             if demands[sat] > 0:
        #                 dummy_node = "E" + str(sat)
        #                 max_flow_emergency += flow_dict[dummy_node][source]
        #                 if flow_dict[dummy_node][source] > demands[sat]:
        #                     print("Emergency flow greater than demand", sat, flow_dict[source][dummy_node], demands[sat])
        #                 if flow_dict[source][dummy_node] != flow_dict[source][dummy_node]:
        #                     print("mismatch in flow")
        #                 emergency_flows_capacity_allocated[sat] = flow_dict[source][dummy_node] / demands[sat]
                    
        #             if sat in incumbent_satellites:
        #                 incumbent_node = "I" + str(sat)
        #                 max_flow_incumbent += flow_dict[source][incumbent_node]
        #                 if flow_dict[source][incumbent_node] > incumbent_demand:
        #                     print("Incumbent flow greater than demand", sat, flow_dict[source][incumbent_node], demands[sat])
        #                 if flow_dict[source][incumbent_node] != flow_dict[source][incumbent_node]:
        #                     print("mismatch in flow")
        #                 incumbent_flows_capacity_allocated[sat] = flow_dict[source][incumbent_node] / incumbent_demand

        #     print(sum(flow_dict[source].values()))
        #     print(flow_dict[source])
            
                    
        #     print(max_flow_emergency, max_flow_incumbent, max_flow_emergency + max_flow_incumbent, max_flow)

        # else:

        if priority == "emergency":
            flow_dict = emergency_traffic_max_flow(demands, graph, source, sink, len(satellites))
            print(sum(flow_dict[sink].values()))
            print(flow_dict)
            emergency_flows_capacity_allocated = calculate_emergency_flows_capacity_allocation(flow_dict, demands, source, len(satellites))
            max_flow_emergency = sum(flow_dict[sink].values())
        else:
            flow_dict = incumbent_traffic_max_flow(graph, source, sink, len(satellites), incumbent_satellites, incumbent_demand)
            incumbent_flows_capacity_allocated = calculate_incumbent_flows_capacity_allocation(flow_dict, source, len(satellites), incumbent_satellites, incumbent_demand)
            max_flow_incumbent = sum(flow_dict[sink].values())
            print(max_flow_incumbent)
            print(flow_dict)

        with open(first_pass_flow_dict_filename, "w") as first_pass_flow_dict_file:
            json.dump(flow_dict, first_pass_flow_dict_file)

        # remove capacity used for max flow from the graph
        for nodeA in flow_dict:
            for nodeB in flow_dict[nodeA]:
                # if flow_dict[nodeA][nodeB] > 0:
                #     print(nodeA, nodeB, graph[nodeA][nodeB]['capacity'], flow_dict[nodeA][nodeB])
                
                
                graph[nodeA][nodeB]['capacity'] -= flow_dict[nodeA][nodeB]
                
                
                # if flow_dict[nodeA][nodeB] > 0:
                #     print(nodeA, nodeB, graph[nodeA][nodeB]['capacity'], flow_dict[nodeA][nodeB])

        remove_dummy_nodes(graph, len(satellites))

        if priority == "emergency":
            flow_dict = incumbent_traffic_max_flow(graph, source, sink, len(satellites), incumbent_satellites, incumbent_demand)
            incumbent_flows_capacity_allocated = calculate_incumbent_flows_capacity_allocation(flow_dict, source, len(satellites), incumbent_satellites, incumbent_demand)
            print(sum(flow_dict[sink].values()))
            max_flow_incumbent = sum(flow_dict[sink].values())
            print(flow_dict)
        else:
            flow_dict = emergency_traffic_max_flow(demands, graph, source, sink, len(satellites))
            print(sum(flow_dict[sink].values()))
            max_flow_emergency = sum(flow_dict[sink].values())
            print(flow_dict)
            emergency_flows_capacity_allocated = calculate_emergency_flows_capacity_allocation(flow_dict, demands, source, len(satellites))

        with open(second_pass_flow_dict_filename, "w") as second_pass_flow_dict_file:
            json.dump(flow_dict, second_pass_flow_dict_file)

        # for nodeA in flow_dict:
        #     for nodeB in flow_dict[nodeA]:
        #         if flow_dict[nodeA][nodeB] > 0:
        #             print(nodeA, nodeB, graph[nodeA][nodeB]['capacity'], flow_dict[nodeA][nodeB])
        
        with open(flow_filename, "a+") as flow_file:
            flow_file.write(f"{time_seconds},{max_flow_emergency},{max_flow_incumbent}\n")

        with open(fulfillment_filename, "a+") as fulfillment_file:
            for sat in range(len(satellites)):
                fulfillment_file.write(f"{time_seconds},{sat},{demands[sat]},{emergency_flows_capacity_allocated[sat]},{incumbent_flows_capacity_allocated[sat]}\n")

def main():
    args = sys.argv[1:]
    if len(args) == 10:
        config = args[2].split("/")[-1]
        base_output_dir = f"{args[0]}/{config}_{args[7]}/{args[8]}/{args[6]}"
        demands_directory = f"{args[0]}/{config}_{args[7]}"


        local_shell = exputil.LocalShell()
        local_shell.make_full_dir(base_output_dir)

        compare_te_priorities_digraph(base_output_dir,
                            demands_directory,
                            args[1],        # graph_dir
                            args[2],        # satellite_network_dir
                            int(args[3]),   # dynamic state update interval ms
                            int(args[4]),   # simulation start time s
                            int(args[5]),   # simulation end time s
                            args[6],        # emergency traffic routing policy
                            args[8],        # priority
                            float(args[9])   # incumbent demand
                )
    else:
        print("Invalid argument selection for compare_te_priorities_digraph.py")
        print("Usage: python compare_te_priorities_digraph.py <base_output_dir> <graph_dir> <satellite_network_dir> <dynamic_state_update_interval_ms> <simulation_start_time_s> <simulation_end_time_s> <priority> <incumbent_demand> <emergency_routing>")

if __name__ == "__main__":
    main()