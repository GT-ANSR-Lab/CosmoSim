#!/usr/bin/env python3

"""
This script performs an interference-to-noise ratio (I/N) analysis across unique satellite pairs in a Starlink constellation snapshot for a specific pair of ground cells. It propagates orbital ephemeris using SGP4 and Astropy, evaluates antenna gain patterns and link budgets, and visualizes the results through scatter plots, histograms, percentile envelopes, and violin plots grouped by satellite separation distance bins comprising of all possible separation values according to the 5shells constellation.
"""

import re
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import h3
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sgp4.api import Satrec, jday
from astropy.coordinates import TEME, ITRS
from astropy.time import Time
import astropy.units as u


TLE_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "constellation_configurations"
    / "configs"
    / "starlink_5shells"
    / "tles.txt"
)

cellA = "854c8a2bfffffff"
cellB = "854c8a23fffffff"

TARGET_SAT_1 = 67
TARGET_SAT_2 = 23

EPOCH_BASE = datetime(2022, 1, 1)
TIME_SECONDS = 1.0

BIN_WIDTH = 50  

R_EARTH = 6371e3

C = 299792458.0
FREQ = 10.7e9

T_SYS = 200.0
k = 1.38064852e-23
B = 250e6

N0 = 10 * np.log10(k * T_SYS * B)

D_UT = 0.6
D_ST = 0.863

lam = C / FREQ

D_LAMBDA_RX = D_UT / lam
D_LAMBDA_TX = D_ST / lam

G_SAT_MAX = 20 * np.log10(D_LAMBDA_TX) + 7.7
EIRP_DENSITY_MAX = -51.1
P_TX_DENSITY = EIRP_DENSITY_MAX - G_SAT_MAX

MIN_ELEV = 25

def load_tles(path):
    sats = {}
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    while i < len(lines) - 2:
        m = re.search(r"Starlink 5-Shells (\d+)", lines[i])
        if m:
            sat = int(m.group(1))
            sats[sat] = Satrec.twoline2rv(lines[i+1], lines[i+2])
            i += 3
        else:
            i += 1
    return sats

def propagate(rec):
    t = EPOCH_BASE + timedelta(seconds=TIME_SECONDS)
    jd, fr = jday(
        t.year, t.month, t.day,
        t.hour, t.minute,
        t.second + t.microsecond / 1e6
    )

    e, r, v = rec.sgp4(jd, fr)
    if e != 0:
        return None

    obstime = Time(t, scale="utc")
    teme = TEME(
        x=r[0] * u.km,
        y=r[1] * u.km,
        z=r[2] * u.km,
        obstime=obstime
    )
    itrs = teme.transform_to(ITRS(obstime=obstime))

    return np.array([
        itrs.x.to(u.m).value,
        itrs.y.to(u.m).value,
        itrs.z.to(u.m).value
    ])

def norm(v):
    return v / (np.linalg.norm(v) + 1e-15)

def angle(a, b):
    a = norm(a)
    b = norm(b)
    return np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1)))

def elevation(user, sat):
    los = sat - user
    z = angle(user, los)
    return 90 - z

def itu_s1528_tx(psi, D):
    psi = max(psi, 1e-9)
    G = 20 * np.log10(D) + 7.7
    pb = np.sqrt(1200) / D
    Y = 1.5 * pb
    Z = Y * 10**(0.04 * (G - 6.75))

    if psi <= Y:
        return G - 3 * (psi / pb)**2
    if psi <= Z:
        return G - 6.75 - 25 * np.log10(psi / Y)
    return 0

def itu_s1428_rx(phi, D):
    phi = max(phi, 1e-9)
    D = np.clip(D, 20, 25)
    Gmax = 20 * np.log10(D) + 7.7
    G1 = 29 - 25 * np.log10(95 / D)
    phim = np.sqrt((Gmax - G1) / 2.5e-3) / D
    phir = 95 / D

    if phi < phim:
        return Gmax - 2.5e-3 * (D * phi)**2
    if phi < phir:
        return G1
    if phi < 33.1:
        return 29 - 25 * np.log10(phi)
    if phi < 80:
        return -9
    return -5

print("Using explicit cells:")
print(f"  Cell A: {cellA}")
print(f"  Cell B: {cellB}")

latA, lonA = h3.cell_to_latlng(cellA)
latB, lonB = h3.cell_to_latlng(cellB)

def ecef(lat, lon):
    lat = np.radians(lat)
    lon = np.radians(lon)
    return np.array([
        R_EARTH * np.cos(lat) * np.cos(lon),
        R_EARTH * np.cos(lat) * np.sin(lon),
        R_EARTH * np.sin(lat)
    ])

userA = ecef(latA, lonA)
userB = ecef(latB, lonB)

print("\nLoading and propagating satellites...")
satdb = load_tles(TLE_FILE)
positions = {}

for sid in tqdm(satdb, desc="Propagation"):
    p = propagate(satdb[sid])
    if p is not None:
        positions[sid] = p

print("\nEvaluating all unique satellite pairs...")
ids = list(positions.keys())
results = []
highlighted_sample = None

for i in tqdm(range(len(ids) - 1), desc="Pairs Processing"):
    sa = ids[i]
    posA = positions[sa]
    
    elevA = elevation(userA, posA)
    if elevA < MIN_ELEV:
        continue

    for j in range(i + 1, len(ids)):
        sb = ids[j]
        posB = positions[sb]

        elevB = elevation(userB, posB)
        if elevB < MIN_ELEV:
            continue

        if elevation(userA, posB) < 0:
            continue

        dist = np.linalg.norm(posA - posB)
        dist_km = dist / 1000.0

        d = np.linalg.norm(userA - posB)
        phi_tx = angle(userA - posB, userB - posB)
        phi_rx = angle(posA - userA, posB - userA)

        gtx = itu_s1528_tx(phi_tx, D_LAMBDA_TX)
        grx = itu_s1428_rx(phi_rx, D_LAMBDA_RX)

        eirp = P_TX_DENSITY + gtx + 10 * np.log10(B)
        fspl = 20 * np.log10(d) + 20 * np.log10(FREQ) - 147.55
        I = eirp + grx - fspl
        in_db = I - N0

        dist_bin = (dist_km // BIN_WIDTH) * BIN_WIDTH + (BIN_WIDTH / 2)

        results.append([
            dist_km,
            dist_bin,
            in_db
        ])

        if (sa == TARGET_SAT_1 and sb == TARGET_SAT_2) or (sa == TARGET_SAT_2 and sb == TARGET_SAT_1):
            highlighted_sample = (dist_km, in_db)

df = pd.DataFrame(results, columns=["distance_km", "distance_bin", "IN_dB"])

if df.empty:
    print("No satellite pairs met the geometric and elevation criteria.")
    exit()

print(f"\nAnalysis complete. Found {len(df)} valid interfering pairs.")
if highlighted_sample:
    print(f"Target Sat Pair ({TARGET_SAT_1}, {TARGET_SAT_2}) found at Separation: {highlighted_sample[0]:.2f} km, I/N: {highlighted_sample[1]:.2f} dB")
else:
    print(f"Target Sat Pair ({TARGET_SAT_1}, {TARGET_SAT_2}) did not meet visibility constraints for this epoch and cell combination.")

plt.figure(figsize=(11, 5))
plt.scatter(df.distance_km, df.IN_dB, s=6, alpha=0.3, color="tab:blue", label="All Constellation Pairs")

if highlighted_sample:
    plt.scatter(
        highlighted_sample[0], 
        highlighted_sample[1], 
        color="red", 
        marker="o", 
        s=150, 
        edgecolor="black", 
        zorder=5, 
        label=f"Pair Sat {TARGET_SAT_1} & Sat {TARGET_SAT_2}"
    )
    plt.annotate(
        f"Sat {TARGET_SAT_1}-{TARGET_SAT_2}\n({highlighted_sample[0]:.1f} km, {highlighted_sample[1]:.1f} dB)",
        xy=highlighted_sample,
        xytext=(highlighted_sample[0] + 80, highlighted_sample[1] + 5),
        arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
        fontweight='bold'
    )

plt.axhline(-12, color="red", linestyle="-.", linewidth=1.5, label="ITU Threshold (-12 dB)")
plt.xlabel("Satellite Separation (km)")
plt.ylabel("I/N (dB)")
plt.title(f"Empirical I/N vs Satellite Separation\nCells: {cellA} & {cellB}")
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4.5))
bins_50km = np.arange(0, df.distance_km.max() + BIN_WIDTH, BIN_WIDTH)
plt.hist(
    df.distance_km,
    bins=bins_50km,
    edgecolor="black",
    color="dimgray"
)
plt.xlabel("Satellite Separation (km) [50 km Bins]")
plt.ylabel("Number of Satellite Pairs")
plt.title("Starlink 5 shells Constellation Pair Distance Distribution (50 km Bins)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

stats = (
    df.groupby("distance_bin")["IN_dB"]
      .agg(
          p5=lambda x: np.percentile(x, 5),
          median="median",
          p95=lambda x: np.percentile(x, 95)
      )
      .reset_index()
)

plt.figure(figsize=(12, 6))
plt.fill_between(
    stats["distance_bin"],
    stats["p5"],
    stats["p95"],
    alpha=0.3,
    color="tab:blue",
    label="5th–95th Percentile"
)
plt.plot(
    stats["distance_bin"],
    stats["median"],
    linewidth=2,
    color="black",
    label="Median"
)
plt.axhline(
    -12,
    color="red",
    linestyle="-.",
    linewidth=2,
    label="ITU Threshold"
)
plt.xlabel("Satellite Separation (km) [50 km Bin Center]")
plt.ylabel("I/N (dB)")
plt.title("Synthetic I/N Envelope Using Real Starlink Satellite Geometry (50 km Bins)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

bins = np.arange(0, df.distance_km.max() + BIN_WIDTH, BIN_WIDTH)
labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]

df["bin50"] = pd.cut(
    df.distance_km,
    bins=bins,
    labels=labels,
    include_lowest=True
)

df_filtered_bins = df.dropna(subset=["bin50"])

plt.figure(figsize=(18, 8))
sns.violinplot(
    data=df_filtered_bins,
    x="bin50",
    y="IN_dB",
    inner="quart",
    density_norm="width",
    cut=0
)
plt.axhline(
    -12,
    color="red",
    linestyle="-.",
    linewidth=2,
    label="ITU Threshold"
)
plt.xticks(rotation=45)
plt.xlabel("Satellite Separation Bin (km)")
plt.ylabel("I/N (dB)")
plt.title("Distribution of I/N for Real Starlink Satellite Pair Separations (50 km Bins)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
