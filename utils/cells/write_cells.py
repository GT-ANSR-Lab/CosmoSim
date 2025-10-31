from utils.distance_tools import *
from .read_cells import *


def write_cells(filename_cells_basic_in, filename_cells_out):
    cells = read_cells(filename_cells_basic_in)
    # print(len(cells, filename_cells_out))
    with open(filename_cells_out, "w+") as f_out:
        for cell in cells:            
            f_out.write(
                "%s,%d\n" % (
                    cell["cell"],
                    cell["num_terminals"]
                )
            )

def write_cells_starlink(filename_cells_basic_in, filename_cells_out):
    cells = read_cells_starlink(filename_cells_basic_in)
    # print(len(cells, filename_cells_out))
    with open(filename_cells_out, "w+") as f_out:
        for cell in cells:            
            f_out.write("%s,%d\n" % (cell["cell"], cell["population"]))
