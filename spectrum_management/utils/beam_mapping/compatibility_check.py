import h3

def check_compatibility(cell, mappings, proposed_beam):
    proposed_sat = int(proposed_beam.split("_")[1])
    proposed_beam_idx = int(proposed_beam.split("_")[2])
    curr_cells = [cell + "_" + str(idx) for idx in range(8)]
    for curr_cell in curr_cells:
        if curr_cell in mappings:
            if proposed_beam_idx == int(mappings[curr_cell].split("_")[2]):
                return False

    neighbors = h3.k_ring(cell, 1)
    
    for neighbor in neighbors:
        nbr_cells = [neighbor + "_" + str(idx) for idx in range(8)]
        for nbr_cell in nbr_cells:
            if nbr_cell in mappings:
                if proposed_sat == int(mappings[nbr_cell].split("_")[1]) and proposed_beam_idx == int(mappings[nbr_cell].split("_")[2]):
                    return False
                
    return True