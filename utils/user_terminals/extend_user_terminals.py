from utils.distance_tools import *
from .read_user_terminals import *


def extend_user_terminals(filename_user_terminals_basic_in, filename_user_terminals_out, num_user_terminals):
    user_terminals = read_user_terminals_basic(filename_user_terminals_basic_in, num_user_terminals)
    # print(len(user_terminals, filename_user_terminals_out))
    with open(filename_user_terminals_out, "w+") as f_out:
        for user_terminal in user_terminals:
            cartesian = geodetic2cartesian(
                float(user_terminal["latitude_degrees_str"]),
                float(user_terminal["longitude_degrees_str"]),
                user_terminal["elevation_m_float"]
            )
            
            f_out.write(
                "%d,%s,%f,%f,%f,%s,%f,%f,%f\n" % (
                    user_terminal["uid"],
                    user_terminal["name"],
                    float(user_terminal["latitude_degrees_str"]),
                    float(user_terminal["longitude_degrees_str"]),
                    user_terminal["elevation_m_float"],
                    user_terminal["cell_id"],
                    cartesian[0],
                    cartesian[1],
                    cartesian[2],
                    # user_terminal["demand"],
                )
            )
