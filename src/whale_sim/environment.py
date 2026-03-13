import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter, distance_transform_edt
from shapely.ops import unary_union
from rasterio import features
from affine import Affine
from src.whale_sim.config import GridConfig, EnvConfig, Formulas


class Environment:
    def __init__(self):
        self.lon, self.lat = np.meshgrid(GridConfig.LON_RANGE, GridConfig.LAT_RANGE)
        self.land_mask = None
        self.ocean_mask = None
        self.sst = None
        self.krill = None

    def build(self):
        self._generate_land()
        self._generate_sst()
        self._generate_krill()
        return self

    def _generate_land(self):
        land_geom = cfeature.NaturalEarthFeature('physical', 'land', '50m').geometries()
        combined_land = unary_union(list(land_geom))

        res_lon = (GridConfig.LON_MAX - GridConfig.LON_MIN) / GridConfig.LON_SIZE
        res_lat = (GridConfig.LAT_MAX - GridConfig.LAT_MIN) / GridConfig.LAT_SIZE
        transform = (Affine.translation(GridConfig.LON_MIN, GridConfig.LAT_MIN)
                     * Affine.scale(res_lon, res_lat))

        self.land_mask = features.rasterize(
            [(combined_land, 1)],
            out_shape=(GridConfig.LAT_SIZE, GridConfig.LON_SIZE),
            transform=transform,
            fill=0,
            dtype='uint8'
        ).astype(bool)
        self.ocean_mask = ~self.land_mask

    def _generate_sst(self):
        noise = gaussian_filter(
            np.random.normal(0, EnvConfig.SST_NOISE_STD, self.lat.shape),
            sigma=EnvConfig.SST_SMOOTH_SIGMA
        )
        self.sst = Formulas.sst_model(self.lat, self.lon, noise)

    def _generate_krill(self):
        coast_proximity = self._coast_proximity()
        lat_effect = self._lat_effect()
        patches = self._place_patches(coast_proximity, lat_effect)

        sst_norm = (self.sst - self.sst.min()) / (self.sst.max() - self.sst.min())
        cold_water = 0.3 * (1 - sst_norm)

        krill = (EnvConfig.KRILL_W_PATCHES   * patches
               + EnvConfig.KRILL_W_COLD      * cold_water * self.ocean_mask
               + EnvConfig.KRILL_W_UPWELLING * (0.9 * coast_proximity)
               + EnvConfig.KRILL_W_LATITUDE  * lat_effect * self.ocean_mask)

        krill = krill * (coast_proximity ** 0.5) * (0.15 + 0.85 * lat_effect)
        krill *= self.ocean_mask
        self.krill = krill / krill.max()

    def _coast_proximity(self):
        coastline_mask = self._build_coastline_mask()
        dist = distance_transform_edt(~coastline_mask) * self.ocean_mask
        max_dist = dist[self.ocean_mask].max()
        return np.exp(-EnvConfig.KRILL_COAST_DECAY * dist / max_dist) * self.ocean_mask

    def _build_coastline_mask(self):
        lat_range = GridConfig.LAT_RANGE
        lon_range = GridConfig.LON_RANGE
        mask = np.zeros((GridConfig.LAT_SIZE, GridConfig.LON_SIZE), dtype=bool)

        fig = plt.figure(figsize=(1, 1))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([GridConfig.LON_MIN, GridConfig.LON_MAX,
                       GridConfig.LAT_MIN, GridConfig.LAT_MAX])

        for geom in cfeature.NaturalEarthFeature('physical', 'coastline', '10m').geometries():
            if not geom.is_valid:
                continue
            for lon_pt, lat_pt in np.array(geom.coords):
                if not (GridConfig.LON_MIN <= lon_pt <= GridConfig.LON_MAX and
                        GridConfig.LAT_MIN <= lat_pt <= GridConfig.LAT_MAX):
                    continue
                i = np.abs(lat_range - lat_pt).argmin()
                j = np.abs(lon_range - lon_pt).argmin()
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < GridConfig.LAT_SIZE and
                                0 <= nj < GridConfig.LON_SIZE and
                                self.ocean_mask[ni, nj]):
                            mask[ni, nj] = True
        plt.close(fig)
        return mask

    def _lat_effect(self):
        norm = self.lat / GridConfig.LAT_MAX
        effect = np.zeros_like(norm)
        effect[norm < 0.5]                     = 0.1 + 0.9 * (norm[norm < 0.5] / 0.5) ** 1.5
        effect[(norm >= 0.5) & (norm < 0.83)]  = 1.0
        effect[norm >= 0.83]                   = 1.0 + 0.2 * ((norm[norm >= 0.83] - 0.83) / 0.17)
        return effect

    def _place_patches(self, coast_proximity, lat_effect):
        lat_range = GridConfig.LAT_RANGE
        density = np.full((GridConfig.LAT_SIZE, GridConfig.LON_SIZE), EnvConfig.KRILL_BACKGROUND)
        y_grid, x_grid = np.mgrid[0:GridConfig.LAT_SIZE, 0:GridConfig.LON_SIZE]

        bands = EnvConfig.KRILL_LAT_BANDS
        for i, weight in enumerate(EnvConfig.KRILL_BAND_WEIGHTS):
            band_start = np.abs(lat_range - bands[i]).argmin()
            band_end   = np.abs(lat_range - bands[i + 1]).argmin()
            n_patches  = int(EnvConfig.KRILL_PATCHES * weight)

            for _ in range(n_patches):
                lat_idx, lon_idx = self._sample_coastal_point(band_start, band_end, coast_proximity)
                if lat_idx is None:
                    continue

                size = (np.random.uniform(EnvConfig.KRILL_PATCH_SIZE_MIN,
                                          EnvConfig.KRILL_PATCH_SIZE_MAX)
                        * (1 + 0.5 * lat_effect[lat_idx, lon_idx]))
                intensity = (EnvConfig.KRILL_BASE_INTENSITY
                             + 0.3 * coast_proximity[lat_idx, lon_idx]
                             + 0.3 * lat_effect[lat_idx, lon_idx])

                dist = np.sqrt((y_grid - lat_idx) ** 2 + (x_grid - lon_idx) ** 2)
                patch = intensity * np.exp(-dist ** 2 / (2 * size ** 2)) * self.ocean_mask
                density = np.maximum(density, patch)

        return density

    def _sample_coastal_point(self, band_start, band_end, coast_proximity):
        for threshold in [EnvConfig.KRILL_COASTAL_THRESHOLD, EnvConfig.KRILL_COASTAL_FALLBACK]:
            for _ in range(100):
                i = np.random.randint(band_start, band_end)
                j = np.random.randint(0, GridConfig.LON_SIZE)
                if not self.land_mask[i, j] and coast_proximity[i, j] > threshold:
                    return i, j
        return None, None