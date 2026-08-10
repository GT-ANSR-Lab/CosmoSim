#!/usr/bin/env python3
"""Shared RF interference and geometry calculations for CosmoSim."""

from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Dict, Tuple, Optional

import numpy as np
from sgp4.api import Satrec, jday
from astropy.coordinates import TEME, ITRS
from astropy.time import Time
import astropy.units as u

C = 299792458.0      # Speed of light (m/s)
F_START = 10.7e9     # Lower bound of Ku-band downlink (10.7 GHz)
F_END = 12.7e9       # Upper bound of Ku-band downlink (12.7 GHz)
BW = 250e6           # Channel bandwidth (250 MHz)
T_SYS = 200.0        # System noise temperature (K)
K_BOLTZMANN = 1.38064852e-23

N0_WATT = K_BOLTZMANN * T_SYS * BW
N0_DB = 10 * np.log10(N0_WATT)

D_UT = 0.6           # Ground terminal dish diameter (m)
D_ST = 0.863         # Satellite Tx dish equivalent diameter (m)
EIRP_DENSITY_MAX = -51.1

EPOCH_BASE = datetime(2022, 1, 1)


def get_channel_frequency(channel_idx: int, f_start: float = F_START, bw: float = BW) -> float:
    """Computes the center frequency (Hz) for a given hardware channel slot."""
    return f_start + (channel_idx + 0.5) * bw


def norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-15)


def angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(norm(a), norm(b)), -1.0, 1.0))))


def ecef(lat: float, lon: float, alt: float = 0.0) -> np.ndarray:
    a = 6378137.0           # Semi-major axis (meters)
    f = 1.0 / 298.257223563 # Flattening
    e2 = 2 * f - f ** 2     # First eccentricity squared
    
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    
    N = a / np.sqrt(1.0 - e2 * (sin_lat ** 2))
    
    x = (N + alt) * cos_lat * np.cos(lon_rad)
    y = (N + alt) * cos_lat * np.sin(lon_rad)
    z = (N * (1.0 - e2) + alt) * sin_lat
    return np.array([x, y, z])


def itu_s1528_tx(psi: float, D: float) -> float:
    psi = max(psi, 1e-9)
    G = 20 * np.log10(D) + 7.7
    pb = np.sqrt(1200) / D
    Y = 1.5 * pb
    Z = Y * 10 ** (0.04 * (G - 6.75))
    if psi <= Y:
        return G - 3 * (psi / pb) ** 2
    if psi <= Z:
        return G - 6.75 - 25 * np.log10(psi / Y)
    return 0.0


def itu_s1428_rx(phi: float, D: float) -> float:
    phi = max(phi, 1e-9)
    D = float(np.clip(D, 20.0, 25.0))
    Gmax = 20 * np.log10(D) + 7.7
    G1 = 29 - 25 * np.log10(95 / D)
    phim = np.sqrt((Gmax - G1) / 2.5e-3) / D
    phir = 95 / D
    if phi < phim:
        return Gmax - 2.5e-3 * (D * phi) ** 2
    if phi < phir:
        return G1
    if phi < 33.1:
        return 29 - 25 * np.log10(phi)
    if phi < 80:
        return -9.0
    return -5.0


def calculate_received_interference_watt(
    u_tgt_pos: np.ndarray,
    u_inf_serving_pos: np.ndarray,
    sat_tgt_pos: np.ndarray,
    sat_inf_pos: np.ndarray,
    channel_idx: int = 0,
) -> float:
    """Calculates received co-channel interference power in Watts for a specific channel slot."""
    freq = get_channel_frequency(channel_idx)
    lam = C / freq
    
    d_lambda_rx = D_UT / lam
    d_lambda_tx = D_ST / lam
    g_sat_max = 20 * np.log10(d_lambda_tx) + 7.7
    p_tx_density = EIRP_DENSITY_MAX - g_sat_max

    d = float(np.linalg.norm(u_tgt_pos - sat_inf_pos))
    vec_tx_main = u_inf_serving_pos - sat_inf_pos
    vec_tx_to_tgt = u_tgt_pos - sat_inf_pos
    psi_tx = angle(vec_tx_to_tgt, vec_tx_main)
    
    vec_rx_main = sat_tgt_pos - u_tgt_pos
    vec_rx_from_inf = sat_inf_pos - u_tgt_pos
    phi_rx = angle(vec_rx_main, vec_rx_from_inf)
    
    gtx = itu_s1528_tx(psi_tx, d_lambda_tx)
    grx = itu_s1428_rx(phi_rx, d_lambda_rx)
    
    eirp = p_tx_density + gtx + 10 * np.log10(BW)
    fspl = 20 * np.log10(d) + 20 * np.log10(freq) - 20 * np.log10(C / (4 * np.pi))
    I_dbw = eirp + grx - fspl
    return 10 ** (I_dbw / 10.0)


def get_satellite_positions(tle_path: Path, time_offset_seconds: float) -> Dict[int, np.ndarray]:
    sats: Dict[int, Satrec] = {}
    if not tle_path.exists():
        return {}

    with open(tle_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.replace("\xa0", " ").strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        line = lines[i]
        if "Starlink" in line or "5-Shells" in line:
            m = re.search(r"(\d+)$", line)
            if m and (i + 2 < len(lines)):
                sat_id = int(m.group(1))
                line1 = lines[i + 1]
                line2 = lines[i + 2]

                if line1.startswith("1 ") and line2.startswith("2 "):
                    try:
                        sats[sat_id] = Satrec.twoline2rv(line1, line2)
                    except Exception:
                        pass
                    i += 3
                    continue
        i += 1

    positions: Dict[int, np.ndarray] = {}
    t = EPOCH_BASE + timedelta(seconds=float(time_offset_seconds))
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond / 1e6)

    for sid, rec in sats.items():
        e, r, v = rec.sgp4(jd, fr)
        if e == 0:
            obstime = Time(t, scale="utc")
            teme = TEME(x=r[0] * u.km, y=r[1] * u.km, z=r[2] * u.km, obstime=obstime)
            itrs = teme.transform_to(ITRS(obstime=obstime))
            positions[sid] = np.array([itrs.x.to(u.m).value, itrs.y.to(u.m).value, itrs.z.to(u.m).value])

    return positions