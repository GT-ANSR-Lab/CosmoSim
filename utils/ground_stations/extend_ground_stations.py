from utils.distance_tools import *
from .read_ground_stations import *


def extend_ground_stations(filename_ground_stations_basic_in, filename_ground_stations_out):
    ground_stations = read_ground_stations_basic(filename_ground_stations_basic_in)
    with open(filename_ground_stations_out, "w+") as f_out:
        for ground_station in ground_stations:
            cartesian = geodetic2cartesian(
                float(ground_station["latitude_degrees_str"]),
                float(ground_station["longitude_degrees_str"]),
                ground_station["elevation_m_float"]
            )
            f_out.write(
                "%d,%s,%f,%f,%f,%f,%f,%f\n" % (
                    ground_station["gid"],
                    ground_station["name"],
                    float(ground_station["latitude_degrees_str"]),
                    float(ground_station["longitude_degrees_str"]),
                    ground_station["elevation_m_float"],
                    cartesian[0],
                    cartesian[1],
                    cartesian[2]
                )
            )
