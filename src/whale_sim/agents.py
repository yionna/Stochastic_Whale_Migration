import numpy as np
import scipy.stats as stats
from src.whale_sim.config import GridConfig, MovementConfig, Formulas


class Whale:
    def __init__(self, whale_id, start_pos, north_to_south=True):
        self.id = whale_id
        self.pos = np.array(start_pos, dtype=float)
        self.north_to_south = north_to_south
        self.history = [self.pos.copy()]
        self.state = 'transit'
        self.state_history = ['transit']
        self.step_count = 0
        self.forage_count = 0
        self.total_krill = 0.0

    def update(self, env):
        self.step_count += 1

        lat_idx, lon_idx = self._grid_idx(env, self.pos)
        sst_val   = env.sst[lat_idx, lon_idx]
        krill_val = env.krill[lat_idx, lon_idx]

        if self.state == 'foraging':
            self.total_krill += krill_val
            self.forage_count += 1

        p_forage = Formulas.foraging_logic(sst_val, krill_val)
        if self.state == 'foraging':
            p_forage = np.clip(p_forage + MovementConfig.STATE_MEMORY_BIAS, 0.1, 0.9)
        else:
            p_forage = np.clip(p_forage, 0.1, 0.9)

        self.state = 'foraging' if np.random.random() < p_forage else 'transit'
        self.state_history.append(self.state)

        step_length, turning_angle = self._sample_movement()
        self._move(step_length, turning_angle, env)

    def _sample_movement(self):
        scale = MovementConfig.METRES_PER_DEGREE
        if self.state == 'transit':
            step = stats.gamma.rvs(MovementConfig.TRANSIT_GAMMA_SHAPE,
                                   scale=MovementConfig.TRANSIT_GAMMA_SCALE / scale)
            bias = 3 * np.pi / 2 if self.north_to_south else np.pi / 2
            angle = stats.vonmises.rvs(MovementConfig.TRANSIT_VM_KAPPA, loc=bias)
        else:
            step = stats.gamma.rvs(MovementConfig.FORAGE_GAMMA_SHAPE,
                                   scale=MovementConfig.FORAGE_GAMMA_SCALE / scale)
            angle = stats.vonmises.rvs(MovementConfig.FORAGE_VM_KAPPA, loc=0)
        return step, angle

    def _move(self, step_length, turning_angle, env):
        orig = self.pos.copy()
        proposed = orig + np.array([step_length * np.sin(turning_angle),
                                    step_length * np.cos(turning_angle)])
        proposed[0] = np.clip(proposed[0], GridConfig.LAT_MIN, GridConfig.LAT_MAX)
        proposed[1] = np.clip(proposed[1], GridConfig.LON_MIN, GridConfig.LON_MAX)

        if not env.land_mask[self._grid_idx(env, proposed)]:
            self.pos = proposed
        else:
            for dist_factor in [0.8, 0.6, 0.4, 0.2]:
                r = step_length * dist_factor
                for test_angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                    test = orig + np.array([r * np.sin(test_angle), r * np.cos(test_angle)])
                    test[0] = np.clip(test[0], GridConfig.LAT_MIN, GridConfig.LAT_MAX)
                    test[1] = np.clip(test[1], GridConfig.LON_MIN, GridConfig.LON_MAX)
                    if not env.land_mask[self._grid_idx(env, test)]:
                        self.pos = test
                        self.history.append(self.pos.copy())
                        return

        self.history.append(self.pos.copy())

    def _grid_idx(self, env, pos):
        i = int(np.clip(np.argmin(np.abs(env.lat[:, 0] - pos[0])), 0, GridConfig.LAT_SIZE - 1))
        j = int(np.clip(np.argmin(np.abs(env.lon[0, :] - pos[1])), 0, GridConfig.LON_SIZE - 1))
        return i, j