import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from src.whale_sim.config import GridConfig

def get_grid_indices(lat, lon):
    """Maps geographic coordinates to the nearest grid array indices."""
    lat_idx = np.argmin(np.abs(GridConfig.LAT_RANGE - lat))
    lon_idx = np.argmin(np.abs(GridConfig.LON_RANGE - lon))
    return lat_idx, lon_idx

def plot_simulation(env, whales):
    """Renders the environment and whale migration paths."""
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set map extent
    ax.set_extent([GridConfig.LON_MIN, GridConfig.LON_MAX, 
                   GridConfig.LAT_MIN, GridConfig.LAT_MAX])

    # Add geographic features
    ax.add_feature(cfeature.LAND, facecolor='#2d2d2d')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)

    # Plot Environmental Layer (e.g., Krill Density)
    im = ax.pcolormesh(env.lon, env.lat, env.krill, 
                       transform=ccrs.PlateCarree(), 
                       cmap='YlGnBu', alpha=0.6)
    plt.colorbar(im, label='Krill Density Suitability', fraction=0.03, pad=0.04)

    # Plot Whale Tracks
    for whale in whales:
        history = np.array(whale.history)
        if len(history) > 1:
            # History is [lat, lon], so plot as [lon, lat]
            ax.plot(history[:, 1], history[:, 0], 
                    color='salmon', linewidth=1.5, alpha=0.8,
                    transform=ccrs.PlateCarree())
            
            # Mark current position
            ax.scatter(whale.pos[1], whale.pos[0], 
                       color='red', s=20, edgecolors='white',
                       transform=ccrs.PlateCarree())

    plt.title("Whale Migration Simulation: SST & Krill-Driven Movement")
    plt.show()

def calculate_stats(whales):
    """Basic analytics on whale behavior across the fleet."""
    states = [w.state for w in whales]
    return {
        "foraging_count": states.count('foraging'),
        "transit_count": states.count('transit'),
        "avg_lat": np.mean([w.pos[0] for w in whales])
    }