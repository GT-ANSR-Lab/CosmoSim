from graph_generation.helpers.graph_tools import *
from utils.distance_tools import *
from utils.user_terminals import *
from utils.global_variables import *
from .compatibility_check import check_compatibility
import numpy as np

def uniform_beam_mapping(all_cells, satellites, satellite_cells, cell_satellites, user_terminals, sat_capacity):
    cell_sat_mapping = {}
    beams_assigned = 0
    max_beams_possible = len(satellites) * 8 * frequency_reuse_factor
    all_sat_beams = set()
    sat_cells_assigned = {}
    for sat in satellites:
        sat_cells_assigned[sat] = []
        for i in range(8):
            for j in range(frequency_reuse_factor):
                all_sat_beams.add(str(j) + "_" + str(sat) + "_" + str(i))
    print("Max number of beams possible", max_beams_possible)
    print("Number of cells", len(all_cells))
    cell_satellite_assignments = {}
    cell_channels_assigned = {}
    

    cell_priority = {}
    for cell in all_cells:
        cell_priority[cell] = 8

    # initialize cell_satellite_assignments and cell_channels_assigned for all cells in all_cells
    for cell in all_cells:
        cell_satellite_assignments[cell] = []
        cell_channels_assigned[cell] = []

    # sort keys in satellite_cells based on number of cells
    satellite_cells = dict(sorted(satellite_cells.items(), key=lambda item: len(item[1])))
    print("Satellite Cells:", satellite_cells)

    for _ in range(8):
        # sort all cells based on priority and length of cell_satellites
        all_cells = sorted(all_cells, key=lambda cell: (-cell_priority[cell], len(cell_satellites[cell])))
        for cell in all_cells:
            if cell_priority[cell] == 0:
                continue
            print(cell, cell_priority[cell], len(cell_satellites[cell]), cell_satellites[cell])

            sats = cell_satellites[cell]
            
            sats = sorted(sats, key=lambda sat: len(sat_cells_assigned[sat]))

            assigned = False
            for ch in range(8):
                dummy_node = cell + "_" + str(ch)
                if not assigned and dummy_node not in cell_sat_mapping:
                    for sat in sats:
                        if assigned:
                            break
                        for freq in range(frequency_reuse_factor):
                            sat_beam = str(freq) + "_" + str(sat) + "_" + str(ch)
                            if sat_beam in all_sat_beams and check_compatibility(cell, cell_sat_mapping, sat_beam):
                                cell_sat_mapping[dummy_node] = sat_beam
                                beams_assigned += 1
                                cell_priority[cell] = max(0, cell_priority[cell] - 1)
                                cell_satellite_assignments[cell].append(sat)
                                cell_channels_assigned[cell].append(ch)
                                all_sat_beams.remove(sat_beam)
                                sat_cells_assigned[sat].append(cell)
                                print("mapping", dummy_node, sat_beam)
                                assigned = True
                                break
                        if dummy_node in cell_sat_mapping:
                            break


    print("Beams assigned:", beams_assigned)
    cell_satellite_assignments = dict(sorted(cell_satellite_assignments.items(), key=lambda item: len(item[1])))
    print("Cell Satellite Assignments:", cell_satellite_assignments)
    return cell_sat_mapping


# from graph_generation.helpers.graph_tools import *
# from utils.distance_tools import *
# from utils.user_terminals import *
# from utils.global_variables import *
# from .compatibility_check import check_compatibility

# def uniform_beam_mapping(all_cells, satellites, satellite_cells):
#     cell_sat_mapping = {}
#     beams_assigned = 0
#     max_beams_possible = len(satellites) * 8 * frequency_reuse_factor
#     all_sat_beams = set()
#     for sat in satellites:
#         for i in range(8):
#             for j in range(frequency_reuse_factor):
#                 all_sat_beams.add(str(j) + "_" + str(sat) + "_" + str(i))
#     print("Max number of beams possible", max_beams_possible)
#     print("Number of cells", len(all_cells))
#     cell_satellite_assignments = {}
#     cell_channels_assigned = {}

#     # initialize cell_satellite_assignments and cell_channels_assigned for all cells in all_cells
#     for cell in all_cells:
#         cell_satellite_assignments[cell] = []
#         cell_channels_assigned[cell] = []

#     # sort keys in satellite_cells based on number of cells
#     satellite_cells = dict(sorted(satellite_cells.items(), key=lambda item: len(item[1])))
#     print("Satellite Cells:", satellite_cells)

#     for sat, cells in satellite_cells.items():
#         print("Satellite iteration", sat, len(cells), cells)

#         # sort cells based on length of cell_satellite_assignments[cell] 
#         # so that cells with fewer satellites get higher preference
#         cells = sorted(cells, key=lambda cell: len(cell_satellite_assignments[cell]))

#         # assign one beam from this satellite to all the cells possible
#         for cell in cells:
#             for cell_ch in range(8):
#                 if sat in cell_satellite_assignments[cell]:
#                     break
#                 dummy_node = cell + "_" + str(cell_ch)
#                 if dummy_node not in cell_sat_mapping:
#                     for freq in range(frequency_reuse_factor):
#                         sat_beam = str(freq) + "_" + str(sat) + "_" + str(cell_ch)
#                         if sat_beam in all_sat_beams and check_compatibility(cell, cell_sat_mapping, sat_beam):
#                             cell_sat_mapping[dummy_node] = sat_beam
#                             beams_assigned += 1
#                             cell_satellite_assignments[cell].append(sat)
#                             cell_channels_assigned[cell].append(cell_ch)
#                             all_sat_beams.remove(sat_beam)
#                             print("mapping", dummy_node, sat_beam)
#                             break

#     print("Beams assigned:", beams_assigned)
#     cell_satellite_assignments = dict(sorted(cell_satellite_assignments.items(), key=lambda item: len(item[1])))
#     print("Cell Satellite Assignments:", cell_satellite_assignments)
#     return cell_sat_mapping