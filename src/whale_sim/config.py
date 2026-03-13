import numpy as np

class GridConfig:
    LAT_SIZE, LON_SIZE = 300, 400
    LAT_MIN, LAT_MAX = 0, 60
    LON_MIN, LON_MAX = -160, -80
    LAT_RANGE = np.linspace(LAT_MIN, LAT_MAX, LAT_SIZE)
    LON_RANGE = np.linspace(LON_MIN, LON_MAX, LON_SIZE)


class EnvConfig:
    # SST: base = 25 - 0.2*(lat - 20) + noise, per model spec
    SST_BASE_TEMP = 25
    SST_LAT_COEFF = 0.2
    SST_LAT_OFFSET = 20
    SST_NOISE_STD = 0.8
    SST_SMOOTH_SIGMA = 5.0
    SST_COAST_LON = -124        # approximate West Coast longitude for upwelling
    SST_COAST_EFFECT = -2.0     # upwelling cools SST by up to 2°C near coast
    SST_COAST_DECAY = 0.5
    SST_SEASONAL_AMP = 2.0      # seasonal ±2°C swing

    # Krill density field
    KRILL_PATCHES = 80
    KRILL_COAST_DECAY = 0.8     # steeper = more concentrated near shore
    KRILL_PATCH_SIZE_MIN = 2.0
    KRILL_PATCH_SIZE_MAX = 8.0
    KRILL_BASE_INTENSITY = 0.7
    KRILL_COASTAL_THRESHOLD = 0.9   # min coast_proximity for patch placement
    KRILL_COASTAL_FALLBACK = 0.7    # relaxed threshold if placement fails
    KRILL_BACKGROUND = 0.01

    # Blend weights for final krill field (must sum to 1.0)
    KRILL_W_PATCHES = 0.25
    KRILL_W_COLD = 0.15
    KRILL_W_UPWELLING = 0.40
    KRILL_W_LATITUDE = 0.20

    # Latitude bands for patch distribution [°N]
    KRILL_LAT_BANDS = [0, 15, 30, 50, 60]
    KRILL_BAND_WEIGHTS = [0.1, 0.4, 0.95, 1.0]  # fraction of base patches per band


class MovementConfig:
    # Bailey et al. (2009), Table 2 — Gamma(shape, scale) step length distributions
    TRANSIT_GAMMA_SHAPE = 2.96
    TRANSIT_GAMMA_SCALE = 7495.9   # metres
    FORAGE_GAMMA_SHAPE = 1.17
    FORAGE_GAMMA_SCALE = 5376.6    # metres

    METRES_PER_DEGREE = 111_000.0

    # VonMises concentration for turning angles
    TRANSIT_VM_KAPPA = 1.5     # directed transit
    FORAGE_VM_KAPPA = 0.3      # relatively undirected foraging

    STATE_MEMORY_BIAS = 0.3    # probability bonus for staying in foraging state


class Formulas:
    @staticmethod
    def sst_model(lat, lon, noise):
        base = EnvConfig.SST_BASE_TEMP - EnvConfig.SST_LAT_COEFF * (lat - EnvConfig.SST_LAT_OFFSET)
        upwelling = EnvConfig.SST_COAST_EFFECT * np.exp(
            -EnvConfig.SST_COAST_DECAY * np.abs(lon - EnvConfig.SST_COAST_LON)
        )
        seasonal = EnvConfig.SST_SEASONAL_AMP * np.sin(np.pi * lat / GridConfig.LAT_MAX)
        return base + noise + upwelling + seasonal

    @staticmethod
    def foraging_logic(sst, krill):
        # Logistic state-switching from Bailey et al. (2009), eq. 3
        p_sst   = 1 / (1 + np.exp(-(-1 - 0.2 * (sst - 16))))
        p_krill = 1 / (1 + np.exp(-5 * (krill - 0.3)))
        return 0.4 * p_sst + 0.6 * p_krill