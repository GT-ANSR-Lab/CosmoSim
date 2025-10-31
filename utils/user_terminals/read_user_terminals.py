from utils.global_variables import *

def read_user_terminals_basic(filename_user_terminals_basic, num_user_terminals):
    """
    Reads user terminals from the input file.

    :param filename_user_terminals_basic: Filename of user terminals basic (typically /path/to/user_terminals.txt)

    :return: List of user terminals
    """
    user_terminals_basic = []
    uid = 0
    with open(filename_user_terminals_basic, 'r') as f:
        if len(f.readlines()) < num_user_terminals:
            raise ValueError("Number of user terminals cannot exceed the number in the given file")
        f.seek(0)
        for line in f:
            split = line.split(',')
            # remove the newline character
            split[-1] = split[-1].strip()
            # print(split)
            if len(split) != 6:
                raise ValueError("Basic user terminal file has 6 columns")
            # if int(split[0]) != uid:
            #     print(uid, split[0])
            #     raise ValueError("User terminal id must increment each line")
            user_terminal_basic = {
                "uid": uid,
                "name": split[1],
                "latitude_degrees_str": split[2],
                "longitude_degrees_str": split[3],
                "elevation_m_float": float(split[4]),
                "cell_id": split[5],
                "sid" : None,
                "hop_count" : satellite_handoff_seconds,
            }
            user_terminals_basic.append(user_terminal_basic)
            uid += 1
            # Quit reading once we reach the required number of user terminals
            if uid == num_user_terminals:
                break
        
    return user_terminals_basic


def read_user_terminals_extended(filename_user_terminals_extended):
    """
    Reads user terminals from the input file.

    :param filename_user_terminals_extended: Filename of user terminals basic (typically /path/to/user_terminals.txt)

    :return: List of user terminals
    """
    user_terminals_extended = []
    uid = 0
    with open(filename_user_terminals_extended, 'r') as f:
        for line in f:
            split = line.split(',')
            if len(split) != 9:
                raise ValueError("Extended user terminal file has 9 columns: " + line)
            if int(split[0]) != uid:
                raise ValueError("user terminal id must increment each line")
            user_terminal_basic = {
                "uid": uid,
                "name": split[1],
                "latitude_degrees_str": split[2],
                "longitude_degrees_str": split[3],
                "elevation_m_float": float(split[4]),
                "cell_id": split[5],
                "cartesian_x": float(split[6]),
                "cartesian_y": float(split[7]),
                "cartesian_z": float(split[8]),
                # "demand" : float(split[8]),
                "sid" : None,
                "hop_count" : satellite_handoff_seconds
            }
            user_terminals_extended.append(user_terminal_basic)
            uid += 1
    return user_terminals_extended
