
def read_ground_stations_basic(filename_ground_stations_basic):
    """
    Reads ground stations from the input file.

    :param filename_ground_stations_basic: Filename of ground stations basic (typically /path/to/ground_stations.txt)

    :return: List of ground stations
    """
    ground_stations_basic = []
    gid = 0
    with open(filename_ground_stations_basic, 'r') as f:
        f.seek(0)
        for line in f:
            split = line.split(',')
            if len(split) != 5:
                raise ValueError("Basic ground station file has 5 columns")
            if int(split[0]) != gid:
                raise ValueError("Ground station id must increment each line")
            ground_station_basic = {
                "gid": gid,
                "name": split[1],
                "latitude_degrees_str": split[2],
                "longitude_degrees_str": split[3],
                "elevation_m_float": float(split[4]),
            }
            ground_stations_basic.append(ground_station_basic)
            gid += 1
            
    return ground_stations_basic


from utils.distance_tools import geodetic2cartesian


def read_ground_stations_extended(filename_ground_stations_extended):
    """
    Reads ground stations from the input file.

    :param filename_ground_stations_extended: Filename of ground stations basic (typically /path/to/ground_stations.txt)

    :return: List of ground stations
    """
    ground_stations_extended = []
    gid = 0
    with open(filename_ground_stations_extended, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            split = [part.strip() for part in line.split(',')]
            if len(split) not in (5, 8):
                raise ValueError("Extended ground station file must have 5 or 8 columns: " + line)
            if int(split[0]) != gid:
                raise ValueError("Ground station id must increment each line")
            elevation = float(split[4])
            if len(split) == 8:
                cartesian_x = float(split[5])
                cartesian_y = float(split[6])
                cartesian_z = float(split[7])
            else:
                lat = float(split[2])
                lon = float(split[3])
                cartesian_x, cartesian_y, cartesian_z = geodetic2cartesian(lat, lon, elevation)
            ground_station_basic = {
                "gid": gid,
                "name": split[1],
                "latitude_degrees_str": split[2],
                "longitude_degrees_str": split[3],
                "elevation_m_float": elevation,
                "cartesian_x": cartesian_x,
                "cartesian_y": cartesian_y,
                "cartesian_z": cartesian_z,
            }
            ground_stations_extended.append(ground_station_basic)
            gid += 1
    return ground_stations_extended
