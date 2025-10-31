
def generate_empty_isls(output_filename_isls):
    """
    Generate empty ISL file.

    :param output_filename_isls     Output filename
    """
    with open(output_filename_isls, 'w+') as f:
        f.write("")

    return []
