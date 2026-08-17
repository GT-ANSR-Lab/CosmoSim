#!/usr/bin/env python3

"""
This script models a satellite's downlink footprint and evaluates interference-to-noise ratio (INR) across H3 geographical cells relative to a specified satellite position and orientation. Useful to evaluate cell buffer required before a reuse beam can be assigned to the same satellite on the same channel. 
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

def get_satellite_ecef(lat_deg, lon_deg, az_deg, el_deg, altitude_m=550e3):
    """
    Computes the Satellite ECEF coordinates given the ground center's lat/lon, 
    and the desired satellite Azimuth, Elevation, and Altitude.
    """
  
    ground_ecef = ecef_vectorized(np.array([lat_deg]), np.array([lon_deg]))[0]
    
    az_rad = np.radians(az_deg)
    el_rad = np.radians(el_deg)
    

    e_u = np.cos(el_rad) * np.sin(az_rad)
    n_u = np.cos(el_rad) * np.cos(az_rad)
    u_u = np.sin(el_rad)
    vec_enu_unit = np.array([e_u, n_u, u_u])

    a, f = 6378137.0, 1.0 / 298.257223563
    Re = a 
    d = Re * (np.sqrt((np.sin(el_rad))**2 + 2 * (altitude_m / Re) + (altitude_m / Re)**2) - np.sin(el_rad))
    
    vec_enu = vec_enu_unit * d

    lat_rad, lon_rad = np.radians(lat_deg), np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)

    R = np.array([
        [-sin_lon,              -sin_lat * cos_lon,           cos_lat * cos_lon],
        [ cos_lon,              -sin_lat * sin_lon,           cos_lat * sin_lon],
        [ 0,                     cos_lat,                     sin_lat]
    ])

    vec_ecef = R @ vec_enu
    sat_ecef = ground_ecef + vec_ecef
    return sat_ecef, ground_ecef

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

# 
def main():
    # 1. Align Boresight Center Perfectly to the H3 Resolution 5 Center
    initial_lat, initial_lon = 33.7490, -84.3880
    center_cell = h3.latlng_to_cell(initial_lat, initial_lon, 5)
    center_lat, center_lon = h3.cell_to_latlng(center_cell)
    
    # Define Satellite Position Geometry relative to Boresight Target
    AZIMUTH_DEG = 277.82    
    ELEVATION_DEG = 28.45    
    SAT_ALTITUDE_M = 550e3 

    h3_cells = list(h3.grid_disk(center_cell, 35))
    
    # 2. Setup Vector Geometries using Azimuth and Elevation
    sat_ecef, boresight_ecef = get_satellite_ecef(
        center_lat, center_lon, 
        az_deg=AZIMUTH_DEG, 
        el_deg=ELEVATION_DEG, 
        altitude_m=SAT_ALTITUDE_M
    )
    
    # Transmission vector from Satellite to the Target Center
    vec_tx_main = boresight_ecef - sat_ecef

    cell_coords = [h3.cell_to_latlng(c) for c in h3_cells]
    cell_lats = np.array([coords[0] for coords in cell_coords])
    cell_lons = np.array([coords[1] for coords in cell_coords])
    
    cell_grid_ecef = ecef_vectorized(cell_lats, cell_lons)
    cell_vec_tx_to_grid = cell_grid_ecef - sat_ecef[None, :]
    cell_psi_tx = angle_vectorized(cell_vec_tx_to_grid, vec_tx_main[None, :])
    cell_gtx = itu_s1528_tx_vectorized(cell_psi_tx, D_LAMBDA_TX)
    
    cell_inr = (P_TX_DENSITY + cell_gtx + 10 * np.log10(B)) + \
               itu_s1428_rx_vectorized(np.zeros_like(cell_psi_tx), D_LAMBDA_RX) - \
               (20 * np.log10(np.linalg.norm(cell_grid_ecef - sat_ecef[None, :], axis=1)) + 20 * np.log10(FREQ) - 20 * np.log10(C / (4 * np.pi))) - \
               N0_DB

    threshold = -12.0
    pass_cells, pass_inrs = [], []
    fail_cells, fail_inrs = [], []
    
    for cell, inr_val in zip(h3_cells, cell_inr):
        if inr_val <= threshold:
            pass_cells.append(cell)
            pass_inrs.append(inr_val)
        else:
            fail_cells.append(cell)
            fail_inrs.append(inr_val)

    def make_geojson(cells):
        features = []
        for cell in cells:
            vertices = h3.cell_to_boundary(cell)
            coords = [[v[1], v[0]] for v in vertices]
            coords.append(coords[0])
            features.append({
                "type": "Feature",
                "id": cell,
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {"cell_id": cell}
            })
        return {"type": "FeatureCollection", "features": features}

    pass_geojson = make_geojson(pass_cells)
    fail_geojson = make_geojson(fail_cells)

    df_pass = pd.DataFrame({"cell_id": pass_cells, "val": 1})
    df_fail = pd.DataFrame({"cell_id": fail_cells, "val": 1})

    fig = go.Figure()

    if fail_cells:
        fig.add_trace(go.Choroplethmap(
            geojson=fail_geojson,
            locations=df_fail["cell_id"],
            z=df_fail["val"],
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
            showscale=False,
            marker=dict(line=dict(width=2.0, color="rgb(0, 0, 0)")),
            hoverinfo="text",
            text=[f"Failing Cell: {c}<br>INR: {inr:.2f} dB" for c, inr in zip(fail_cells, fail_inrs)],
            name="Failing Cells (Fail)"
        ))

    if pass_cells:
        fig.add_trace(go.Choroplethmap(
            geojson=pass_geojson,
            locations=df_pass["cell_id"],
            z=df_pass["val"],
            colorscale=[[0, "rgba(0, 100, 255, 0.25)"], [1, "rgba(0, 100, 255, 0.25)"]],
            showscale=False,
            marker=dict(line=dict(width=2.0, color="rgb(0, 100, 255)")),
            hoverinfo="text",
            text=[f"Passing Cell: {c}<br>INR: {inr:.2f} dB" for c, inr in zip(pass_cells, pass_inrs)],
            name="Passing Cells (Pass)"
        ))

    fig.add_trace(go.Scattermap(
        lat=[center_lat],
        lon=[center_lon],
        mode="markers+text",
        marker=dict(size=[14], color=["cyan"]),
        text=["Boresight Center"],
        textposition="top center",
        name="Boresight"
    ))

    fig.update_layout(
        map=dict(style="open-street-map", center=dict(lat=center_lat, lon=center_lon), zoom=6.5),
        margin={"r":0,"t":50,"l":0,"b":0},
        title=f"Satellite Footprint (Az: {AZIMUTH_DEG}°, El: {ELEVATION_DEG}°)"
    )

    with tempfile.NamedTemporaryFile(suffix="_segmented_reuse.html", delete=False) as f:
        fig.write_html(f.name)
        webbrowser.open(f"file://{f.name}")

if __name__ == "__main__":
    main()
