from src.whale_sim.environment import Environment
from src.whale_sim.agents import Whale
from src.whale_sim.utils import plot_simulation, calculate_stats


def create_whales():
    """
    Whale starting positions from Bailey et al. (2009).
    Northbound whales start near Baja/central California coast;
    Southbound whales start near Alaska/Pacific Northwest feeding grounds.
    """
    # Northbound: start near Baja/central CA, migrate toward Alaska
    north_coords = [
        (29.70, -123.14),
        (31.98, -124.33),
        (34.84, -125.31),
        (24.66, -121.32),
        (25.23, -118.52),
        (26.61, -114.33),
    ]

    # Southbound: start near Alaska/Pacific NW, migrate toward Baja
    south_coords = [
        (56.56, -145.45),
        (46.57, -126.00),
        (47.68, -125.52),
        (47.30, -125.75),
        (47.17, -124.85),
        (47.83, -128.74),
    ]

    whales = []
    whale_id = 0

    for lat, lon in north_coords:
        whales.append(Whale(whale_id, [lat, lon], north_to_south=False))
        whale_id += 1

    for lat, lon in south_coords:
        whales.append(Whale(whale_id, [lat, lon], north_to_south=True))
        whale_id += 1

    return whales


def run_experiment(num_steps=500):
    print("Building environment (coastline mask may take ~1 min)...")
    env = Environment().build()
    print("Environment ready.")

    whales = create_whales()
    print(f"Simulating {len(whales)} whales for {num_steps} steps...")

    for t in range(num_steps):
        for whale in whales:
            whale.update(env)

        if t % 100 == 0:
            stats = calculate_stats(whales)
            print(f"  Step {t:>4}: {stats['foraging_count']:>2} foraging | "
                  f"Avg lat {stats['avg_lat']:.2f}°")

    print("\nSimulation complete. Plotting...")
    plot_simulation(env, whales)


if __name__ == "__main__":
    run_experiment()