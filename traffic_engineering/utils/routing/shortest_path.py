import networkx as nx

def get_path_length(graph, path):
    length = 0
    for i in range(len(path) - 1):
        length += graph[path[i]][path[i + 1]]['weight']
    return length

def shortest_path_modifications(graph, satellites, num_ground_stations, ground_station_offset):
    
    edges = set()
    for sat in satellites:
        shortest_path = []
        gs_distance = float('inf')
        for gs in range(num_ground_stations):
            try:
                path = nx.shortest_path(graph, source=sat, target=gs + ground_station_offset, weight='weight')
                path_length = get_path_length(graph, path)
                if path_length < gs_distance:
                    gs_distance = len(path)
                    shortest_path = path

            except nx.NetworkXNoPath:
                continue
                
        
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