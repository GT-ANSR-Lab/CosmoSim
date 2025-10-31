from .graph_tools import compute_path_length_with_graph

def find_all_paths(flow_dict, current_node, sink, path=[]):
    # print("current node", current_node, sink)
    path = path + [current_node]
    if current_node == sink:
        return [path]
    all_paths = []
    next_nodes = flow_dict[current_node]
    for next_node, flow in next_nodes.items():
        if flow > 0 and next_node not in path:
            # print(path, next_node, flow)
            new_paths = find_all_paths(flow_dict, next_node, sink, path)
            for new_path in new_paths:
                all_paths.append(new_path)
    return all_paths

def calculate_rtts(ut_start, ut_end, graph, flow_dict, sink):
    all_user_paths = {}
    ut_path_lengths = {}
    ut_path_rtts = {}
    PROPAGATION_SPEED = 299792458
    for ut_id in range(ut_start, ut_end + 1):
        # print(ut_id, sink, flow_dict)
        paths = find_all_paths(flow_dict, ut_id, sink)
        if paths:
            all_user_paths[ut_id] = paths
            path_lengths = [compute_path_length_with_graph(path, graph) for path in paths]
            ut_path_lengths[ut_id] = path_lengths
            ut_path_rtts[ut_id] = [(2 * length / PROPAGATION_SPEED) * 1000 for length in path_lengths]
        else:
            print(f"No paths found for user terminal {ut_id}")
    return ut_path_rtts
