import numpy as np
import h3
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm  

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 16,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

H3_RES = 5
R_EARTH = 6371e3
ALT = 540e3
R_SAT = R_EARTH + ALT  

C = 299792458.0
lowestf = 10.7e9
MIN_ELEV = 25.0


MAX_ATTEMPTS_NEAR_ZERO = 800 
MIN_ATTEMPTS_LIMIT = 50     

T_SYS = 200.0
k = 1.38064852e-23
B = 250e6
N0 = 10 * np.log10(k * T_SYS * B)

D_UT = 0.6
D_ST = 0.863
lam = C / lowestf
D_LAMBDA_RX = D_UT / lam
D_LAMBDA_TX = D_ST / lam

G_SAT_MAX = 20 * np.log10(D_LAMBDA_TX) + 7.7
EIRP_DENSITY_MAX = -51.1
P_TX_DENSITY = EIRP_DENSITY_MAX - G_SAT_MAX

def ecef(lat, lon, r):
    lat = np.radians(lat)
    lon = np.radians(lon)
    return np.array([
        r * np.cos(lat) * np.cos(lon),
        r * np.cos(lat) * np.sin(lon),
        r * np.sin(lat)
    ])

def norm(v):
    n = np.linalg.norm(v, axis=0, keepdims=True)
    return v / (n + 1e-15)

def angle(a, b):
    c = np.clip(np.dot(norm(a), norm(b)), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def get_elevation_angle(user_pos, sat_pos_vec):
    slant_range_vec = sat_pos_vec - user_pos
    zenith_angle = angle(user_pos, slant_range_vec)
    return 90.0 - zenith_angle


def random_satellite_in_view(lat0, lon0, min_elev_deg):
    psi_max = np.arccos((R_EARTH / R_SAT) * np.cos(np.radians(min_elev_deg))) - np.radians(min_elev_deg)
    u, v = np.random.rand(), np.random.rand()
    psi = np.arccos(1 - u * (1 - np.cos(psi_max)))
    az = 2 * np.pi * v

    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)

    lat = np.arcsin(np.sin(lat0) * np.cos(psi) + np.cos(lat0) * np.sin(psi) * np.cos(az))
    lon = lon0 + np.arctan2(np.sin(az) * np.sin(psi) * np.cos(lat0), np.cos(psi) - np.sin(lat0) * np.sin(lat))
    
    g = ecef(np.degrees(lat), np.degrees(lon), R_EARTH)
    return norm(g) * R_SAT


def itu_s1528_tx(psi_deg, D_lambda):
    psi = np.maximum(np.atleast_1d(psi_deg).astype(float), 1e-12)
    g = np.zeros_like(psi)
    Gm = 20.0 * np.log10(D_lambda) + 7.7
    psi_b = np.sqrt(1200.0) / D_lambda
    Ls, LF = -6.75, 0.0
    Y = 1.5 * psi_b
    Z = Y * 10**((0.04 * (Gm + Ls - LF)))

    m1 = psi <= Y
    g[m1] = Gm - 3.0 * (psi[m1] / psi_b)**2
    m2 = (psi > Y) & (psi <= Z)
    g[m2] = Gm + Ls - 25.0 * np.log10(psi[m2] / Y)
    m3 = (psi > Z) & (psi <= 180.0)
    g[m3] = LF
    return g[0] if np.isscalar(psi_deg) else g

def itu_s1428_rx(phi_deg, D_lambda):
    phi = np.maximum(np.atleast_1d(phi_deg).astype(float), 1e-12)
    g = np.zeros_like(phi)
    D_lambda_clipped = np.clip(float(D_lambda), 20.0, 25.0)

    Gmax = 20.0 * np.log10(D_lambda_clipped) + 7.7
    G1 = 29.0 - 25.0 * np.log10(95.0 / D_lambda_clipped)
    phi_m = (1.0 / D_lambda_clipped) * np.sqrt((Gmax - G1) / 2.5e-3)
    phi_r = 95.0 / D_lambda_clipped

    m1 = phi < phi_m
    g[m1] = Gmax - 2.5e-3 * (D_lambda_clipped * phi[m1])**2
    m2 = (phi >= phi_m) & (phi < phi_r)
    g[m2] = G1
    m3 = (phi >= phi_r) & (phi < 33.1)
    g[m3] = 29.0 - 25.0 * np.log10(phi[m3])
    m4 = (phi >= 33.1) & (phi < 80.0)
    g[m4] = -9.0
    m5 = phi >= 80.0
    g[m5] = -5.0
    return g[0] if np.isscalar(phi_deg) else g


cell_a = h3.latlng_to_cell(0, 0, H3_RES)
cell_b = list(h3.grid_ring(cell_a, 1))[0] 

lat_a, lon_a = h3.cell_to_latlng(cell_a)
lat_b, lon_b = h3.cell_to_latlng(cell_b)

user_a = norm(ecef(lat_a, lon_a, R_EARTH)) * R_EARTH
user_b = norm(ecef(lat_b, lon_b, R_EARTH)) * R_EARTH

I_N = []
sep = []

for target_dist_km in tqdm(range(1, 3001), desc="Sweeping Kilometer Steps"):
    target_dist_m = target_dist_km * 1000.0
    
    cos_theta = (2 * R_SAT**2 - target_dist_m**2) / (2 * R_SAT**2)
    if cos_theta < -1.0 or cos_theta > 1.0:
        continue  
    theta = np.arccos(cos_theta)
    
    slope = (MAX_ATTEMPTS_NEAR_ZERO - MIN_ATTEMPTS_LIMIT) / 3000.0
    allowed_attempts = int(MAX_ATTEMPTS_NEAR_ZERO - (slope * target_dist_km))
    allowed_attempts = max(MIN_ATTEMPTS_LIMIT, allowed_attempts)
    
    for _ in range(allowed_attempts):
            
        sa = random_satellite_in_view(lat_a, lon_a, MIN_ELEV)
        
        v_z = norm(sa)
        ref_axis = np.array([0.0, 0.0, 1.0]) if abs(v_z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        v_x = norm(ref_axis - np.dot(ref_axis, v_z) * v_z)
        v_y = np.cross(v_z, v_x)
        
        az = 2 * np.pi * np.random.rand()
        sb = R_SAT * (v_z * np.cos(theta) + (v_x * np.cos(az) + v_y * np.sin(az)) * np.sin(theta))
        
        if get_elevation_angle(user_a, sa) < MIN_ELEV:
            continue
        if get_elevation_angle(user_b, sb) < MIN_ELEV:
            continue
        if get_elevation_angle(user_a, sb) < 0.0: 
            continue

        d = np.linalg.norm(user_a - sb)
        
        phi_tx = angle(user_a - sb, user_b - sb)
        g_tx = itu_s1528_tx(phi_tx, D_LAMBDA_TX)
        eirp = P_TX_DENSITY + g_tx + 10 * np.log10(B)

        phi_rx = angle(sa - user_a, sb - user_a)
        g_rx = itu_s1428_rx(phi_rx, D_LAMBDA_RX)

        fspl = 20 * np.log10(d) + 20 * np.log10(lowestf) - 147.55

        I = eirp + g_rx - fspl
        I_N.append(I - N0)
        sep.append(target_dist_km)

I_N = np.array(I_N)
sep = np.array(sep)

plt.figure(figsize=(10, 4.5))
plt.hist(sep, bins=range(1, 3002), color='darkslateblue', alpha=0.85, edgecolor='none')
plt.xlim(0, 3000)
plt.xlabel("Satellite Distance Domain (1 km Increments)")
plt.ylabel("Accepted Sample Volume Count")
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

unique_seps = np.unique(sep)
p5_curve = []
median_curve = []
p95_curve = []

for s_val in unique_seps:
    vals_at_s = I_N[sep == s_val]
    p5_curve.append(np.percentile(vals_at_s, 5))
    median_curve.append(np.median(vals_at_s))
    p95_curve.append(np.percentile(vals_at_s, 95))

plt.figure(figsize=(12, 6.5))
plt.fill_between(unique_seps, p5_curve, p95_curve, alpha=0.25, color='tab:blue', label='5th–95th Percentile Envelope')
plt.plot(unique_seps, median_curve, color='black', linewidth=2, label='Geometric Median')
plt.axhline(-12, color='red', linestyle='-.', linewidth=2, label='ITU -12 dB Reference')

plt.xlim(0, 3000)
plt.xlabel("Inter-Satellite Separation Distance (km)")
plt.ylabel("Interference-to-Noise Ratio (I/N) [dB]")
plt.title("Parametric Interference Sweep: 5% - 95% Bound Envelope Analysis")
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

bins_50km = np.arange(0, 3050, 50)
labels_50km = [f"{bins_50km[i]}-{bins_50km[i+1]}" for i in range(len(bins_50km)-1)]

sep_binned = pd.cut(sep, bins=bins_50km, labels=labels_50km, include_lowest=True)

plot_df = pd.DataFrame({
    "Separation Bin (50 km Steps)": sep_binned,
    "I/N (dB)": I_N
}).dropna()

plt.figure(figsize=(16, 8))
sns.violinplot(
    data=plot_df,
    x="Separation Bin (50 km Steps)",
    y="I/N (dB)",
    inner="quart",
    density_norm="width",
    palette="viridis",
    cut=0
)
plt.axhline(-12, color="red", linestyle="-.", linewidth=2, label="ITU -12 dB Reference Threshold")
plt.xlabel("Satellite Spacing Interval Bin (km)")
plt.ylabel("Interference-to-Noise Ratio (I/N) [dB]")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend(loc='upper right')
plt.tight_layout()

print(f"Simulation complete. Total valid data metrics captured: {len(sep):,}")
plt.show()