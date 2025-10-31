import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def read_locations(file_path):
    locations = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                idx = int(parts[0])
                name = parts[1]
                lat = float(parts[2])
                lon = float(parts[3])
                locations.append((idx, name, lat, lon))
    return locations

places = read_locations('ground_stations_starlink.basic.txt')

# Create a new map
plt.figure(figsize=(15,10))
m = Basemap(projection='merc', llcrnrlat=-60, urcrnrlat=80,
            llcrnrlon=-180, urcrnrlon=180, resolution='c')

m.drawcoastlines()
m.drawcountries()
m.drawmapboundary(fill_color='aqua')
m.fillcontinents(color='lightgreen', lake_color='aqua')

# Plot each place
for idx, name, lat, lon in places:
    x, y = m(lon, lat)
    # m.plot(x, y, 'bo', markersize=5)
    plt.text(x, y, str(idx))

# Show the plot
plt.title("World Map with Specified Locations")
plt.savefig('ground_stations_starlink_idxs.png')
plt.savefig('ground_stations_starlink_idxs.svg')