import networkx as nx
import numpy as np

def hot_potato_modifications(graph, demands, num_ground_stations, ground_station_offset):
    
    satellites = np.nonzero(demands)[0]
    print(satellites)
    print(demands[satellites])
    edges = set()
    for sat in satellites:
        shortest_path = []
        gs_distance = float('inf')
        for gs in range(num_ground_stations):
            # print(f"sat: {sat}, gs: {gs + ground_station_offset}")
            try:
                path = nx.shortest_path(graph, source=gs + ground_station_offset, target=sat)
                
                if len(path) < gs_distance:
                    gs_distance = len(path)
                    shortest_path = path
            except nx.NetworkXNoPath:
                continue
                
        
        print(shortest_path)
        # add all edges in the shortest path to the set, with the min node coming before the max node
        for i in range(len(shortest_path) - 1):
            # print(shortest_path[i], shortest_path[i + 1], type(shortest_path[i]), type(shortest_path[i + 1]))
            # if isinstance(shortest_path[i], int) and isinstance(shortest_path[i + 1], int):
            edges.add((shortest_path[i], shortest_path[i + 1]))
            # print(edges)

    print(edges)

    # set "capacity" of all edges in the graph to 0 except those in edges
    for edge in graph.edges:
        # if both nodes aren't integers, skip
        if isinstance(edge[0], int) and isinstance(edge[1], int):
                
            if edge not in edges:
                graph[edge[0]][edge[1]]['capacity'] = 0
                
    return graph



def hot_potato_modifications_old(graph, satellites, num_ground_stations, ground_station_offset):
    
    edges = set()
    for sat in satellites:
        shortest_path = []
        gs_distance = float('inf')
        for gs in range(num_ground_stations):
            path = nx.shortest_path(graph, source=sat, target=gs + ground_station_offset)
            
            if len(path) < gs_distance:
                gs_distance = len(path)
                shortest_path = path
                
        
        # add all edges in the shortest path to the set, with the min node coming before the max node
        for i in range(len(shortest_path) - 1):
            if isinstance(shortest_path[i], int) and isinstance(shortest_path[i + 1], int):
                edges.add((min(shortest_path[i], shortest_path[i + 1]), max(shortest_path[i], shortest_path[i + 1])))

    print(edges)

    # set "capacity" of all edges in the graph to 0 except those in edges
    for edge in graph.edges:
        # if both nodes aren't integers, skip
        if isinstance(edge[0], int) and isinstance(edge[1], int):
            # ensure the edge has smaller value first
            if edge[0] < edge[1]:
                edge = (edge[0], edge[1])
            else:
                edge = (edge[1], edge[0])
                
            if edge not in edges:
                graph[edge[0]][edge[1]]['capacity'] = 0
                
    return graph