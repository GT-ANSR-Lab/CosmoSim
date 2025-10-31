import json

def generate_description(
        filename_description,
        max_gsl_length_m,
        max_isl_length_m
):
    with open(filename_description, "w+") as f_out:
        f_out.write("num_shells=1\n")
        f_out.write("max_gsl_length_m=%.10f\n" % max_gsl_length_m)
        f_out.write("max_isl_length_m=%.10f\n" % max_isl_length_m)

def generate_description_shells(
        filename_description,
        num_orbits,
        num_sats_per_orbit,
        max_gsl_length_m,
        max_isl_length_m
):
    with open(filename_description, "w+") as f_out:
        f_out.write("num_shells=%d\n" % len(num_orbits))
        f_out.write("num_orbits=" + json.dumps(num_orbits) + "\n")
        f_out.write("num_sats_per_orbit=" + json.dumps(num_sats_per_orbit) + "\n")
        f_out.write("max_gsl_length_m=" + json.dumps(max_gsl_length_m) + "\n")
        f_out.write("max_isl_length_m=" + json.dumps(max_isl_length_m) + "\n")
