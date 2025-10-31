import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml
import math

# Allow importing the local utilities without requiring installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils.tles.generate_tles_from_scratch import generate_tles_from_scratch_manual_shells
from utils.isls.generate_empty_isls import generate_empty_isls
from utils.isls.generate_three_isls import generate_three_isls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TLE and ISL files for a constellation definition."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="YAML file describing the constellation shell layout.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "configs",
        help="Directory where generated configs are stored (default: %(default)s).",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping.")
    return data


def ensure_required_fields(config: Dict[str, Any]) -> None:
    required_top = ["name", "nice_name", "shells"]
    for field in required_top:
        if field not in config:
            raise ValueError(f"Missing required config key: {field}")
    if not isinstance(config["shells"], list) or not config["shells"]:
        raise ValueError("`shells` must be a non-empty list.")


def extract_shell_parameter(shells: List[Dict[str, Any]], key: str) -> List[Any]:
    values = []
    for idx, shell in enumerate(shells):
        if key not in shell:
            raise ValueError(f"Shell index {idx} missing required key `{key}`.")
        values.append(shell[key])
    return values


def write_tles(config: Dict[str, Any], output_dir: Path) -> None:
    shells = config["shells"]
    num_shells = len(shells)
    num_orbits = extract_shell_parameter(shells, "num_orbits")
    sats_per_orbit = extract_shell_parameter(shells, "sats_per_orbit")
    inclinations = extract_shell_parameter(shells, "inclination_deg")
    altitudes = extract_shell_parameter(shells, "altitude_m")

    eccentricity = config.get("eccentricity", 1e-7)
    arg_of_perigee = config.get("arg_of_perigee_deg", 0.0)
    phase_diff = bool(config.get("phase_diff", True))

    # Convert altitude to mean motion (revolutions per day)
    mean_motions = []
    mu = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)
    earth_radius = 6_378_135.0  # meters
    for altitude_m in altitudes:
        semi_major = earth_radius + float(altitude_m)
        mean_motion_rad_s = math.sqrt(mu / semi_major ** 3)
        mean_motion_rev_day = mean_motion_rad_s * 86400.0 / (2.0 * math.pi)
        mean_motions.append(mean_motion_rev_day)

    tles_path = output_dir / "tles.txt"
    generate_tles_from_scratch_manual_shells(
        tles_path,
        config["nice_name"],
        num_shells,
        num_orbits,
        sats_per_orbit,
        phase_diff,
        inclinations,
        eccentricity,
        arg_of_perigee,
        mean_motions,
    )


def write_isls(config: Dict[str, Any], output_dir: Path) -> None:
    isls_path = output_dir / "isls.txt"

    shells = config["shells"]
    default_isl = config.get("isl", {})

    edges = set()
    sat_offset = 0

    for idx, shell in enumerate(shells):
        shell_isl = shell.get("isl", {})
        isl_cfg = {**default_isl, **shell_isl}
        isl_type = isl_cfg.get("type", "plus_grid")

        n_orbits = int(shell["num_orbits"])
        sats_per_orbit = int(shell["sats_per_orbit"])

        if isl_type == "none":
            sat_offset += n_orbits * sats_per_orbit
            continue

        elif isl_type == "plus_grid":
            shift = int(isl_cfg.get("shift", 0))

            for orbit in range(n_orbits):
                for sat_idx in range(sats_per_orbit):
                    sat = sat_offset + orbit * sats_per_orbit + sat_idx

                    same_orbit = sat_offset + orbit * sats_per_orbit + ((sat_idx + 1) % sats_per_orbit)
                    edges.add(tuple(sorted((sat, same_orbit))))

                    adjacent_orbit = (orbit + 1) % n_orbits
                    adj_sat_idx = (sat_idx + shift) % sats_per_orbit
                    adjacent_sat = sat_offset + adjacent_orbit * sats_per_orbit + adj_sat_idx
                    edges.add(tuple(sorted((sat, adjacent_sat))))

        elif isl_type == "three_isls":
            shift = int(isl_cfg.get("shift", 0))
            local_edges = generate_three_isls("/dev/null", n_orbits, sats_per_orbit, shift)
            for a, b in local_edges:
                edges.add(tuple(sorted((a + sat_offset, b + sat_offset))))

        else:
            raise ValueError(f"Unsupported ISL type `{isl_type}` in shell index {idx}.")

        sat_offset += n_orbits * sats_per_orbit

    if not edges:
        generate_empty_isls(isls_path)
        return

    with isls_path.open("w") as fh:
        for a, b in sorted(edges):
            fh.write(f"{a} {b}\n")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_required_fields(config)

    output_dir = args.output_root / config["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    write_tles(config, output_dir)
    write_isls(config, output_dir)

    print(f"Wrote constellation artefacts to {output_dir}")


if __name__ == "__main__":
    main()
