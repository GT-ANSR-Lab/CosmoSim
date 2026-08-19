#!/usr/bin/env python3

"""
This script models the allowable elevation angle range of nearby cells based on a sole directly overhead interferer on a fixed central Atlanta cell. It plots safe elevation angle ranges of nearby cells by performing a sweep between min elevation angle extremes on the plane of the interfering beam.
"""

import tempfile
import webbrowser
import numpy as np
import pandas as pd
import h3
import plotly.graph_objects as go

C = 299792458.0           # Speed of light (m/s)
FREQ = 10.7e9             # Ku-Band Downlink (Hz)
B = 250e6                 # Bandwidth (Hz)
T_SYS = 200.0             # System noise temperature (K)
k_B = 1.38064852e-23      # Boltzmann's constant

N0_WATT = k_B * T_SYS * B
N0_DB = 10 * np.log10(N0_WATT)

D_UT = 0.6                # User Terminal (m)
D_ST = 0.863              # Satellite Dish (m)
lam = C / FREQ
D_LAMBDA_RX = D_UT / lam
D_LAMBDA_TX = D_ST / lam

G_SAT_MAX = 20 * np.log10(D_LAMBDA_TX) + 7.7
EIRP_DENSITY_MAX = -51.1  # dBW/Hz
P_TX_DENSITY = EIRP_DENSITY_MAX - G_SAT_MAX

def ecef_vectorized(lats, lons, alt=0.0):
    rad_lat = np.radians(lats)
    rad_lon = np.radians(lons)
    a, f = 6378137.0, 1.0 / 298.257223563
    e2 = 2 * f - f**2
    chi = np.sqrt(1.0 - e2 * np.sin(rad_lat)**2)
    x = (a / chi + alt) * np.cos(rad_lat) * np.cos(rad_lon)
    y = (a / chi + alt) * np.cos(rad_lat) * np.sin(rad_lon)
    z = (a / chi * (1.0 - e2) + alt) * np.sin(rad_lat)
    return np.column_stack((x, y, z))

def norm_vectorized(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-15)

def angle_vectorized(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(norm_vectorized(a) * norm_vectorized(b), axis=-1), -1, 1)))

def itu_s1528_tx_vectorized(psi, D):
    psi = np.maximum(np.asarray(psi), 1e-9)
    G = 20 * np.log10(D) + 7.7
    pb = np.sqrt(1200) / D
    Y = 1.5 * pb
    Z = Y * 10**(0.04 * (G - 6.75))
    g = np.zeros_like(psi)
    g[psi <= Y] = G - 3 * (psi[psi <= Y] / pb)**2
    idx2 = (psi > Y) & (psi <= Z)
    g[idx2] = G - 6.75 - 25 * np.log10(psi[idx2] / Y)
    return g

def itu_s1428_rx_vectorized(phi, D):
    phi = np.maximum(np.asarray(phi), 1e-9)
    D = np.clip(D, 20, 25)
    Gmax = 20 * np.log10(D) + 7.7
    G1 = 29 - 25 * np.log10(95 / D)
    phim = np.sqrt((Gmax - G1) / 2.5e-3) / D
    phir = 95 / D
    g = np.zeros_like(phi)
    g[phi < phim] = Gmax - 2.5e-3 * (D * phi[phi < phim])**2
    g[(phi >= phim) & (phi < phir)] = G1
    g[(phi >= phir) & (phi < 33.1)] = 29 - 25 * np.log10(phi[(phi >= phir) & (phi < 33.1)])
    g[(phi >= 33.1) & (phi < 80)] = -9.0
    g[phi >= 80] = -5.0
    return g

def main():
    initial_lat, initial_lon = 33.7490, -84.3880
    center_cell = h3.latlng_to_cell(initial_lat, initial_lon, 5)
    center_lat, center_lon = h3.cell_to_latlng(center_cell)
    
    max_k = 35
    h3_cells = list(h3.grid_disk(center_cell, max_k))
    
    sat_ecef = ecef_vectorized(np.array([center_lat]), np.array([center_lon]), 550e3)[0]
    boresight_ecef = ecef_vectorized(np.array([center_lat]), np.array([center_lon]))[0]
    vec_tx_main = boresight_ecef - sat_ecef

    cell_coords = [h3.cell_to_latlng(c) for c in h3_cells]
    cell_lats = np.array([coords[0] for coords in cell_coords])
    cell_lons = np.array([coords[1] for coords in cell_coords])
    cell_grid_ecef = ecef_vectorized(cell_lats, cell_lons)
    
    cell_vec_tx_to_grid = cell_grid_ecef - sat_ecef[None, :]
    cell_psi_tx = angle_vectorized(cell_vec_tx_to_grid, vec_tx_main[None, :])
    cell_gtx = itu_s1528_tx_vectorized(cell_psi_tx, D_LAMBDA_TX)
    
    distances = np.linalg.norm(cell_grid_ecef - sat_ecef[None, :], axis=1)
    fspl = 20 * np.log10(distances) + 20 * np.log10(FREQ) - 20 * np.log10(C / (4 * np.pi))
    incident_power_db = (P_TX_DENSITY + cell_gtx + 10 * np.log10(B)) - fspl - N0_DB

    local_up = norm_vectorized(cell_grid_ecef)
    vec_to_interferer = norm_vectorized(sat_ecef[None, :] - cell_grid_ecef)
    

    theta_int = np.degrees(np.arcsin(np.clip(np.sum(vec_to_interferer * local_up, axis=1), -1, 1)))

    sweep_full = np.linspace(25.0, 155.0, 1300)
    threshold = -12.0

    fig = go.Figure()
    plot_x = []
    plot_y = []
    direct_interferer_elevs = {}

    for i, cell in enumerate(h3_cells):
        k_dist = h3.grid_distance(center_cell, cell)
        t_int = theta_int[i]
        
        if k_dist not in direct_interferer_elevs:
            direct_interferer_elevs[k_dist] = []
        direct_interferer_elevs[k_dist].append(t_int)

        phi_full = np.abs(sweep_full - t_int)

        g_rx_sweep = itu_s1428_rx_vectorized(phi_full, D_LAMBDA_RX)
        inr_sweep = incident_power_db[i] + g_rx_sweep

        valid_mask = (inr_sweep <= threshold)
        
        if np.any(valid_mask):
            x_val = k_dist + np.random.uniform(-0.12, 0.12)
            
            valid_indices = np.where(valid_mask)[0]
            splits = np.where(np.diff(valid_indices) > 1)[0]
            
            segment_starts = np.insert(valid_indices[splits + 1], 0, valid_indices[0])
            segment_ends   = np.append(valid_indices[splits], valid_indices[-1])
            
            for s, e in zip(segment_starts, segment_ends):
                plot_x.extend([x_val, x_val, None])
                plot_y.extend([sweep_full[s], sweep_full[e], None])

    fig.add_trace(go.Scatter(
        x=plot_x,
        y=plot_y,
        mode="lines",
        line=dict(color="rgba(0, 102, 204, 0.5)", width=1.8),
        name="INR ≤ -12 dB (Allowed)"
    ))

    k_unique = sorted(direct_interferer_elevs.keys())
    direct_y = [np.mean(direct_interferer_elevs[k]) for k in k_unique]

    fig.add_trace(go.Scatter(
        x=k_unique,
        y=direct_y,
        mode="lines+markers",
        line=dict(color="red", width=2.5, dash="dash"),
        marker=dict(size=5, color="red"),
        name="Direct Interferer Direction"
    ))

    fig.update_layout(
        title="Co-Channel Allowed Beam Angle Ranges vs. Distance from Interferer Cell",
        xaxis=dict(
            title="Cell Distance from Interferer Center Cell (in Cell counts)",
            dtick=2,
            gridcolor="rgba(200, 200, 200, 0.3)"
        ),
        yaxis=dict(
            title="Continuous Sweep Angle (25° = Interferer Horizon, 90° = Zenith, 155° = Opposite Horizon)",
            range=[20, 160],
            dtick=10,
            gridcolor="rgba(200, 200, 200, 0.3)"
        ),
        template="plotly_white",
        showlegend=True
    )

    with tempfile.NamedTemporaryFile(suffix="_cochannel_sweep_corrected.html", delete=False) as f:
        fig.write_html(f.name)
        webbrowser.open(f"file://{f.name}")

    print("Graph Generation Complete.")

if __name__ == "__main__":
    main()
