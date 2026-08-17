#!/usr/bin/env python3

"""
This script plots spatial transmission gain values relative to boresight direction overlaid on a map.
"""

import os
import webbrowser
import tempfile
import folium
import h3
import numpy as np
import branca.colormap as cm

CENTER_LAT = 33.7490
CENTER_LON = -84.3880
H3_RES = 5

ELEVATION_DEG = 28.45
AZIMUTH_DEG = 277.82

ALT = 550e3      
R_EARTH = 6371e3  

# Antenna 
C = 299792458.0
FREQ = 10.7e9     
D_ST = 0.863       # Antenna diameter in meters

lam = C / FREQ
D_LAMBDA_TX = D_ST / lam

GRID_SIZE = 100   
SPAN_DEG = 1.2     

def ecef(lat, lon, r):
    lat = np.radians(lat)
    lon = np.radians(lon)
    return np.column_stack((
        r * np.cos(lat) * np.cos(lon),
        r * np.cos(lat) * np.sin(lon),
        r * np.sin(lat)
    ))

def get_satellite_pos(target_ecef, target_lat, target_lon, elev_deg, az_deg, alt):
    lat = np.radians(target_lat)
    lon = np.radians(target_lon)
    elev = np.radians(elev_deg)
    az = np.radians(az_deg)

    east = np.array([-np.sin(lon), np.cos(lon), 0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])

    dir_v = (
        np.cos(elev) * np.sin(az) * east
        + np.cos(elev) * np.cos(az) * north
        + np.sin(elev) * up
    )
    dir_v /= np.linalg.norm(dir_v)

    r_sat = R_EARTH + alt
    b = 2 * np.dot(target_ecef, dir_v)
    c = np.dot(target_ecef, target_ecef) - r_sat**2
    disc = np.maximum(b * b - 4 * c, 0)
    d = (-b + np.sqrt(disc)) / 2

    return target_ecef + d * dir_v

def angle_vectorized(v1, v2):
    a = v1 / np.linalg.norm(v1)
    b = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1, 1)))

def itu_s1528_tx(psi_deg, D_lambda):
    psi_deg = np.maximum(np.atleast_1d(psi_deg), 1e-12)
    Gm = 20 * np.log10(D_lambda) + 7.7
    psi_b = np.sqrt(1200) / D_lambda
    Ls = -6.75
    Y = 1.5 * psi_b
    Z = Y * 10 ** (0.04 * (Gm + Ls))

    g = np.zeros_like(psi_deg)

    idx1 = psi_deg <= Y
    g[idx1] = Gm - 3 * (psi_deg[idx1] / psi_b) ** 2

    idx2 = (psi_deg > Y) & (psi_deg <= Z)
    g[idx2] = Gm + Ls - 25 * np.log10(psi_deg[idx2] / Y)

    idx3 = psi_deg > Z
    g[idx3] = 0

    return g

target_cell = h3.latlng_to_cell(CENTER_LAT, CENTER_LON, H3_RES)
cell_center_lat, cell_center_lon = h3.cell_to_latlng(target_cell)
boundary = h3.cell_to_boundary(target_cell)

boresight_ecef = ecef(cell_center_lat, cell_center_lon, R_EARTH)[0]
sat_ecef = get_satellite_pos(
    boresight_ecef, cell_center_lat, cell_center_lon, ELEVATION_DEG, AZIMUTH_DEG, ALT
)
sat_to_boresight = boresight_ecef - sat_ecef

lat_vec = np.linspace(cell_center_lat - SPAN_DEG, cell_center_lat + SPAN_DEG, GRID_SIZE)
lon_vec = np.linspace(cell_center_lon - SPAN_DEG, cell_center_lon + SPAN_DEG, GRID_SIZE)
LAT, LON = np.meshgrid(lat_vec, lon_vec)
lats, lons = LAT.flatten(), LON.flatten()

users_ecef = ecef(lats, lons, R_EARTH)
sat_to_users = users_ecef - sat_ecef
psi = angle_vectorized(sat_to_boresight, sat_to_users)
gains = itu_s1528_tx(psi, D_LAMBDA_TX)

mask = gains > 0
lats, lons, gains = lats[mask], lons[mask], gains[mask]

m = folium.Map(
    location=[cell_center_lat, cell_center_lon], 
    zoom_start=9, 
    tiles="CartoDB positron"
)

colormap = cm.LinearColormap(
    colors=["#2c7bb6", "#00a6ca", "#00ccbc", "#90eb9d", "#f9d057", "#d7191c"], 
    vmin=np.min(gains), 
    vmax=np.max(gains)
).to_step(n=6)
colormap.caption = "Antenna Gain Pattern (dBi)"
colormap.add_to(m)

fg = folium.FeatureGroup(name="Gain Distribution Grid")
for lat, lon, gain in zip(lats, lons, gains):
    color_hex = colormap(gain)
    folium.CircleMarker(
        location=[lat, lon],
        radius=3,
        color=color_hex,
        fill=True,
        fill_color=color_hex,
        fill_opacity=0.6,
        weight=0,
        tooltip=f"{gain:.2f} dBi"
    ).add_to(fg)
fg.add_to(m)

folium.Polygon(
    locations=boundary, 
    color="black", 
    weight=3, 
    fill=False, 
    popup=f"Target Cell: Res-{H3_RES}"
).add_to(m)

folium.CircleMarker(
    location=[cell_center_lat, cell_center_lon],
    radius=5,
    color="black",
    fill=True,
    fill_color="white",
    tooltip="Boresight Intersection Point"
).add_to(m)

with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as f:
    f.write(m.get_root().render())
    temp_path = f.name

webbrowser.open("file://" + temp_path)
