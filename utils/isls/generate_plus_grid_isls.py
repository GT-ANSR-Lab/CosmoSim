
def generate_plus_grid_isls(output_filename_isls, n_orbits, n_sats_per_orbit, isl_shift, idx_offset=0):
    """
    Generate plus grid ISL file.

    :param output_filename_isls     Output filename
    :param n_orbits:                Number of orbits
    :param n_sats_per_orbit:        Number of satellites per orbit
    :param isl_shift:               ISL shift between orbits (e.g., if satellite id in orbit is X,
                                    does it also connect to the satellite at X in the adjacent orbit)
    :param idx_offset:              Index offset (e.g., if you have multiple shells)
    """

    if n_orbits < 3 or n_sats_per_orbit < 3:
        raise ValueError("Number of x and y must each be at least 3")

    list_isls = []
    for i in range(n_orbits):
        for j in range(n_sats_per_orbit):
            sat = i * n_sats_per_orbit + j

            # Link to the next in the orbit
            sat_same_orbit = i * n_sats_per_orbit + ((j + 1) % n_sats_per_orbit)
            sat_adjacent_orbit = ((i + 1) % n_orbits) * n_sats_per_orbit + ((j + isl_shift) % n_sats_per_orbit)
            
            # in place just to simulate variations in connectivity with phase
            if i % 2 == 0:
                sat_adjacent_orbit = ((i + 1) % n_orbits) * n_sats_per_orbit + ((j + isl_shift) % n_sats_per_orbit)
            else:
                sat_adjacent_orbit = ((i + 1) % n_orbits) * n_sats_per_orbit + ((j - 1 + isl_shift) % n_sats_per_orbit)

            

            # Same orbit
            list_isls.append((idx_offset + min(sat, sat_same_orbit), idx_offset + max(sat, sat_same_orbit)))

            # Adjacent orbit
            list_isls.append((idx_offset + min(sat, sat_adjacent_orbit), idx_offset + max(sat, sat_adjacent_orbit)))

    with open(output_filename_isls, 'w+') as f:
        for (a, b) in list_isls:
            f.write(str(a) + " " + str(b) + "\n")

    return list_isls

def generate_plus_grid_isls_shells(output_filename_isls, n_shells, n_orbits, n_sats_per_orbit, isl_shift, idx_offset=0):
    """
    Generate plus grid ISL file.

    :param output_filename_isls     Output filename
    :param n_shells:                Number of shells
    :param n_orbits:                Number of orbits
    :param n_sats_per_orbit:        Number of satellites per orbit
    :param isl_shift:               ISL shift between orbits (e.g., if satellite id in orbit is X,
                                    does it also connect to the satellite at X in the adjacent orbit)
    :param idx_offset:              Index offset (e.g., if you have multiple shells)
    """

    # if n_orbits < 3 or n_sats_per_orbit < 3:
    #     raise ValueError("Number of x and y must each be at least 3")

    list_isls = []
    sats_so_far = 0
    for n in range(n_shells):
        for i in range(n_orbits[n]):
            for j in range(n_sats_per_orbit[n]):
                sat = sats_so_far + i * n_sats_per_orbit[n] + j

                # Link to the next in the orbit
                sat_same_orbit = sats_so_far + i * n_sats_per_orbit[n] + ((j + 1) % n_sats_per_orbit[n])
                sat_adjacent_orbit = sats_so_far + ((i + 1) % n_orbits[n]) * n_sats_per_orbit[n] + ((j + isl_shift) % n_sats_per_orbit[n])

                # Same orbit
                list_isls.append((idx_offset + min(sat, sat_same_orbit), idx_offset + max(sat, sat_same_orbit)))

                # Adjacent orbit
                list_isls.append((idx_offset + min(sat, sat_adjacent_orbit), idx_offset + max(sat, sat_adjacent_orbit)))
        
        sats_so_far += n_orbits[n] * n_sats_per_orbit[n]

    with open(output_filename_isls, 'w+') as f:
        for (a, b) in list_isls:
            f.write(str(a) + " " + str(b) + "\n")

    return list_isls
