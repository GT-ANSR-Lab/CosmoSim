from utils.global_variables import *

def read_cells(filename_cells):
    """
    Reads user terminals from the input file.

    :param filename_cells: Filename of cells

    :return: List of user terminals
    """
    cells = []
    with open(filename_cells, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            split = [part.strip() for part in line.split(',')]
            if split[0].lower() == "h3_index":
                continue
            if len(split) != 2:
                raise ValueError("Cell file requires 2 columns")
            # if int(split[0]) != uid:
            #     print(uid, split[0])
            #     raise ValueError("User terminal id must increment each line")
            cell = {
                "cell": split[0],
                "num_terminals": int(split[1])
            }
            cells.append(cell)
        
    return cells


def read_cells_starlink(filename_cells_starlink):
    """
    Reads cells from the input file.

    :param filename_cells_starlink: Filename for starlink cells list

    :return: List of user terminals
    """
    cells_extended = []
    with open(filename_cells_starlink, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            split = [part.strip() for part in line.split(',')]
            if split[0].lower() == "h3_index":
                continue
            if len(split) != 2:
                raise ValueError("Starlink cell file has 2 columns: " + line)
            cell = {
                "cell": split[0],
                "population": int(float(split[1]))
            }
            cells_extended.append(cell)
            
    return cells_extended
