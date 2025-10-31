from exputil import parse_positive_int


def read_isls(filename_isls, num_satellites):
    """
    Read ISLs file into a list of undirected edges

    :param filename_isls:  Filename of ISLs (typically /path/to/isls.txt)
    :param num_satellites: Number of satellites (to verify indices)

    :return: List of all undirected ISL edges
    """
    isls_list = []
    with open(filename_isls, 'r') as f:
        isls_set = set()
        for line in f:
            line_spl = line.split()
            a = parse_positive_int(line_spl[0])
            b = parse_positive_int(line_spl[1])

            # Verify the input
            if a >= num_satellites:
                raise ValueError("Satellite does not exist: %d" % a)
            if b >= num_satellites:
                raise ValueError("Satellite does not exist: %d" % b)
            if b <= a:
                raise ValueError("The second satellite index must be strictly larger than the first")
            if (a, b) in isls_set:
                raise ValueError("Duplicate ISL: (%d, %d)" % (a, b))
            isls_set.add((a, b))

            # Finally add it to the list
            isls_list.append((a, b))

    return isls_list
