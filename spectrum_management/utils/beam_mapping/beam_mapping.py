from graph_generation.helpers.graph_tools import *
from utils.distance_tools import *
from utils.user_terminals import *
import utils.global_variables as global_variables
import geopandas as gpd
from .compatibility_check import check_compatibility
import h3
import numpy as np
import math

def get_cell_population(all_cells, country):
    # TODO: figure out a better way to pass this input
    population_data_path = f"../constellation_configurations/user_terminals/shp_files/{country}/{country}.gpkg"
    population_gdf = gpd.read_file(population_data_path)
    cell_population = {}
    for cell in all_cells:
        cell_population[cell] = 0
    for _, row in population_gdf.iterrows():
        res8_cell = row['h3']
        population = row['population']
        res5_parent = h3.h3_to_parent(res8_cell, 5)
        if res5_parent in all_cells:
            cell_population[res5_parent] = cell_population.get(res5_parent, 0) + population

    return cell_population

def uniform(all_cells, country):
    cell_priority = {}
    for cell in all_cells:
        cell_priority[cell] = 8
    return cell_priority

def priority(all_cells, country):
    cell_population = get_cell_population(all_cells, country)

    max_population = max(cell_population.values())
    interval = max_population / 8
    cell_priority = {}
    for cell in all_cells:
        # all cells with any population should have a beam; and hence minimum priority is fixed as 1.
        cell_priority[cell] = max(1, math.ceil(cell_population[cell] / interval))

    return cell_priority

def waterfill_cell_priority(cells, users_per_channel):
    cell_priority = {}

    for cell in cells:
        cell_priority[cell["cell"]] = min(math.ceil(cell["num_terminals"] / users_per_channel), 8)

    return cell_priority


def popwaterfill_allocation(cells, cell_priority, cell_satellites, satellites, satellite_cells, country, shell_satellite_indices):
    print(cell_priority)
    cell_population = get_cell_population(cells, country)
    cell_sat_mapping = {}
    all_sat_beams = set()
    sat_cells_assigned = {}
    beams_assigned = 0
    for sat in satellites:
        sat_cells_assigned[sat] = []
        for i in range(8):
            for j in range(global_variables.frequency_reuse_factor):
                all_sat_beams.add(str(j) + "_" + str(sat) + "_" + str(i))

    cell_satellite_assignments = {}
    cell_channels_assigned = {}
    # initialize cell_satellite_assignments and cell_channels_assigned for all cells in all_cells
    for cell in cells:
        cell_satellite_assignments[cell] = []
        cell_channels_assigned[cell] = []

    # sort keys in satellite_cells based on number of cells
    satellite_cells = dict(sorted(satellite_cells.items(), key=lambda item: len(item[1])))
    print("Satellite Cells:", satellite_cells)
    num_shells = len(shell_satellite_indices)

    # sort all cells based on priority and cell populations
    all_cells = sorted(cells, key=lambda cell: (-cell_priority[cell], -cell_population[cell]))
    for cell in all_cells:
        if cell_priority[cell] == 0:
            break
        print(cell, cell_priority[cell], len(cell_satellites[cell]), cell_satellites[cell])

        sats = cell_satellites[cell]
        sat_priority = {}
        sat_from_shells = [[] for _ in range(num_shells)]
        for sat in sats:
            for i, (start, end) in enumerate(shell_satellite_indices):
                if start <= sat < end:
                    sat_from_shells[i].append(sat)
                    break

        for idx, shell in enumerate(sat_from_shells):
            num_sats_in_shell = len(shell)
            for i, sat in enumerate(shell):
                if i <= int(0.6 * num_sats_in_shell):
                    sat_priority[sat] = idx
                else:
                    sat_priority[sat] = idx + num_shells

        for ch in range(8):
            if cell_priority[cell] == 0:
                break
            sats = sorted(sats, key=lambda sat: (sat_priority[sat], len(sat_cells_assigned[sat])))
            sats_per_shell = []

            dummy_node = cell + "_" + str(ch)
            if dummy_node not in cell_sat_mapping:
                for sat in sats:
                    # if sat_gsl_capacities[sat] < global_variables.ku_beam_capacity and sat_isl_capacities[sat] < global_variables.ku_beam_capacity:
                    #     continue
                    for freq in range(global_variables.frequency_reuse_factor):
                        sat_beam = str(freq) + "_" + str(sat) + "_" + str(ch)
                        if sat_beam in all_sat_beams and check_compatibility(cell, cell_sat_mapping, sat_beam):
                            cell_sat_mapping[dummy_node] = sat_beam
                            beams_assigned += 1
                            cell_priority[cell] = max(0, cell_priority[cell] - 1)
                            cell_satellite_assignments[cell].append(sat)
                            cell_channels_assigned[cell].append(ch)
                            # if sat_gsl_capacities[sat] > global_variables.ku_beam_capacity:
                            #     sat_gsl_capacities[sat] -= global_variables.ku_beam_capacity
                            # else:
                            #     sat_isl_capacities[sat] -= (global_variables.ku_beam_capacity - sat_gsl_capacities[sat])
                            #     sat_gsl_capacities[sat] = 0
                            all_sat_beams.remove(sat_beam)
                            sat_cells_assigned[sat].append(cell)
                            print("mapping", dummy_node, sat_beam)
                            break
                    if dummy_node in cell_sat_mapping:
                        break

        

    
    for cell in all_cells:
        if cell_priority[cell] == 0:
            continue
        print(cell, cell_priority[cell], len(cell_satellites[cell]), cell_satellites[cell])

        sats = cell_satellites[cell]
        sat_priority = {}
        sat_from_shells = [[] for _ in range(num_shells)]
        for sat in sats:
            for i, (start, end) in enumerate(shell_satellite_indices):
                if start <= sat < end:
                    sat_from_shells[i].append(sat)
                    break

        for idx, shell in enumerate(sat_from_shells):
            num_sats_in_shell = len(shell)
            for i, sat in enumerate(shell):
                if i <= int(0.6 * num_sats_in_shell):
                    sat_priority[sat] = idx
                else:
                    sat_priority[sat] = idx + num_shells
        
        
        for ch in range(8):
            if cell_priority[cell] == 0:
                break
            sats = sorted(sats, key=lambda sat: (sat_priority[sat], len(sat_cells_assigned[sat])))
            dummy_node = cell + "_" + str(ch)
            if dummy_node not in cell_sat_mapping:
                for sat in sats:
                    for freq in range(global_variables.frequency_reuse_factor):
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
                            break
                    if dummy_node in cell_sat_mapping:
                        break



    print("Beams assigned:", beams_assigned)
    cell_satellite_assignments = dict(sorted(cell_satellite_assignments.items(), key=lambda item: len(item[1])))
    print("Cell Satellite Assignments:", cell_satellite_assignments)
    return cell_sat_mapping


def waterfill_allocation(cells, cell_priority, cell_satellites, satellites, satellite_cells, shell_satellite_indices):
    print(cell_priority)
    cell_sat_mapping = {}
    all_sat_beams = set()
    sat_cells_assigned = {}
    beams_assigned = 0
    for sat in satellites:
        sat_cells_assigned[sat] = []
        for i in range(8):
            for j in range(global_variables.frequency_reuse_factor):
                all_sat_beams.add(str(j) + "_" + str(sat) + "_" + str(i))

    cell_satellite_assignments = {}
    cell_channels_assigned = {}
    # initialize cell_satellite_assignments and cell_channels_assigned for all cells in all_cells
    for cell in cells:
        cell_satellite_assignments[cell] = []
        cell_channels_assigned[cell] = []

    # sort keys in satellite_cells based on number of cells
    satellite_cells = dict(sorted(satellite_cells.items(), key=lambda item: len(item[1])))
    print("Satellite Cells:", satellite_cells)

    # sort all cells based on priority and length of cell_satellites
    all_cells = sorted(cells, key=lambda cell: (-cell_priority[cell], len(cell_satellites[cell])))
    for cell in all_cells:
        print(cell, cell_priority[cell], len(cell_satellites[cell]), cell_satellites[cell])

        sats = cell_satellites[cell]
        
        
        for ch in range(8):
            sats = sorted(sats, key=lambda sat: len(sat_cells_assigned[sat]))
            dummy_node = cell + "_" + str(ch)
            if dummy_node not in cell_sat_mapping:
                for sat in sats:
                    for freq in range(global_variables.frequency_reuse_factor):
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
                            break
                    if dummy_node in cell_sat_mapping:
                        break

            if cell_priority[cell] == 0:
                break

    
    # for cell in all_cells:
    #     print(cell, cell_priority[cell], len(cell_satellites[cell]), cell_satellites[cell])

    #     sats = cell_satellites[cell]
        
        
    #     for ch in range(8):
    #         sats = sorted(sats, key=lambda sat: len(sat_cells_assigned[sat]))
    #         dummy_node = cell + "_" + str(ch)
    #         if dummy_node not in cell_sat_mapping:
    #             for sat in sats:
    #                 if sat_capacities[sat] <= 0:
    #                     continue
    #                 for freq in range(global_variables.frequency_reuse_factor):
    #                     sat_beam = str(freq) + "_" + str(sat) + "_" + str(ch)
    #                     if sat_beam in all_sat_beams and check_compatibility(cell, cell_sat_mapping, sat_beam):
    #                         cell_sat_mapping[dummy_node] = sat_beam
    #                         beams_assigned += 1
    #                         cell_priority[cell] = max(0, cell_priority[cell] - 1)
    #                         cell_satellite_assignments[cell].append(sat)
    #                         cell_channels_assigned[cell].append(ch)
    #                         all_sat_beams.remove(sat_beam)
    #                         sat_cells_assigned[sat].append(cell)
    #                         sat_capacities[sat] -= global_variables.ku_beam_capacity
    #                         print("mapping", dummy_node, sat_beam, sat_capacities[sat])
    #                         break
    #                 if dummy_node in cell_sat_mapping:
    #                     break



    print("Beams assigned:", beams_assigned)
    cell_satellite_assignments = dict(sorted(cell_satellite_assignments.items(), key=lambda item: len(item[1])))
    print("Cell Satellite Assignments:", cell_satellite_assignments)
    return cell_sat_mapping

def priority_allocation(cells, cell_priority, cell_satellites, satellites, satellite_cells):
    print(cell_priority)
    cell_sat_mapping = {}
    all_sat_beams = set()
    sat_cells_assigned = {}
    beams_assigned = 0
    for sat in satellites:
        sat_cells_assigned[sat] = []
        for i in range(8):
            for j in range(global_variables.frequency_reuse_factor):
                all_sat_beams.add(str(j) + "_" + str(sat) + "_" + str(i))

    cell_satellite_assignments = {}
    cell_channels_assigned = {}
    # initialize cell_satellite_assignments and cell_channels_assigned for all cells in all_cells
    for cell in cells:
        cell_satellite_assignments[cell] = []
        cell_channels_assigned[cell] = []

    # sort keys in satellite_cells based on number of cells
    satellite_cells = dict(sorted(satellite_cells.items(), key=lambda item: len(item[1])))
    print("Satellite Cells:", satellite_cells)

    for _ in range(8):
        # sort all cells based on priority and length of cell_satellites
        all_cells = sorted(cells, key=lambda cell: (-cell_priority[cell], len(cell_satellites[cell])))
        for cell in all_cells:
            # if cell_priority[cell] == 0:
            #     continue
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
                        for freq in range(global_variables.frequency_reuse_factor):
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


def beam_mapping(policy, cells, satellites, satellite_cells, cell_satellites, config, shell_satellite_indices, users_per_channel):
    country = config.split("_")[9]
    print(config, country)
    
    max_beams_possible = len(satellites) * 8 * global_variables.frequency_reuse_factor
    
    print("Max number of beams possible", max_beams_possible)
    print("Number of cells", len(cells))
    
    cells_list = [cell["cell"] for cell in cells]
    if policy == "uniform":
        cell_priority = uniform(cells_list, country)
    elif policy == "priority":
        cell_priority = priority(cells_list, country)
    elif policy == "waterfill" or policy == "popwaterfill":
        cell_priority = waterfill_cell_priority(cells, users_per_channel)

    if policy == "waterfill":
        cell_sat_mapping = waterfill_allocation(cells_list, cell_priority, cell_satellites, satellites, satellite_cells, shell_satellite_indices)
    elif policy == "popwaterfill":
        cell_sat_mapping = popwaterfill_allocation(cells_list, cell_priority, cell_satellites, satellites, satellite_cells, country, shell_satellite_indices)
    else:
        cell_sat_mapping = priority_allocation(cells_list, cell_priority, cell_satellites, satellites, satellite_cells)

    return cell_sat_mapping