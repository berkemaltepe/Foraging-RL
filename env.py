
import numpy as np

class ForagingEnv:
    EMPTY = 0
    FOOD  = 1

    VARIANTS = {
        "V1": dict(grid_size=10, n_food=10,  max_steps=500, action_noise_eps=0.0),
        "V2": dict(grid_size=20, n_food=20,  max_steps=500, action_noise_eps=0.0),
        "V3": dict(grid_size=20, n_food=20,  max_steps=500, action_noise_eps=0.1),
    }

    def __init__(self, grid_size=20, n_food=20, max_steps=500,
                 vision_radius=2, action_noise_eps=0.0,
                 food_respawn=True, seed=None):
        self.grid_size        = grid_size
        self.n_food           = n_food
        self.max_steps        = max_steps
        self.vision_radius    = vision_radius
        self.action_noise_eps = action_noise_eps
        self.food_respawn     = food_respawn
        self.rng              = np.random.default_rng(seed)
        self.obs_size         = (2 * vision_radius + 1) ** 2
        self.n_actions        = 4
        self._action_map      = {0:(-1,0),1:(1,0),2:(0,-1),3:(0,1)}
        self.reset()

    @classmethod
    def make(cls, variant, seed=None, vision_radius=2):
        cfg = cls.VARIANTS[variant].copy()
        return cls(**cfg, vision_radius=vision_radius, seed=seed)

    def reset(self):
        self.grid             = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.steps            = 0
        self.total_food_eaten = 0
        self.agent_pos        = self._random_empty_cell()
        self.food_positions   = set()
        while len(self.food_positions) < self.n_food:
            pos = tuple(self._random_empty_cell())
            if pos != tuple(self.agent_pos):
                self.food_positions.add(pos)
                self.grid[pos] = self.FOOD
        return self._get_obs()

    def step(self, action):
        if self.action_noise_eps > 0:
            if self.rng.random() < self.action_noise_eps:
                action = self.rng.integers(0, self.n_actions)
        dr, dc = self._action_map[action]
        new_r  = self.agent_pos[0] + dr
        new_c  = self.agent_pos[1] + dc
        reward = -0.01
        if not (0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size):
            reward -= 0.5
        else:
            self.agent_pos = [new_r, new_c]
            pos = tuple(self.agent_pos)
            if pos in self.food_positions:
                reward += 1.0
                self.total_food_eaten += 1
                self.food_positions.remove(pos)
                self.grid[pos] = self.EMPTY
                if self.food_respawn:
                    self._spawn_food()
        self.steps += 1
        done = self.steps >= self.max_steps
        return self._get_obs(), reward, done, {"food_eaten": self.total_food_eaten}

    def _get_obs(self):
        r, c   = self.agent_pos
        radius = self.vision_radius
        window = np.full((2*radius+1, 2*radius+1), 2, dtype=np.float32)
        for dr in range(-radius, radius+1):
            for dc in range(-radius, radius+1):
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    val = self.FOOD if (nr,nc) in self.food_positions else self.EMPTY
                    window[dr+radius, dc+radius] = val
        return window.flatten() / 2.0

    def _random_empty_cell(self):
        while True:
            r = self.rng.integers(0, self.grid_size)
            c = self.rng.integers(0, self.grid_size)
            if self.grid[r, c] == self.EMPTY:
                return [r, c]

    def _spawn_food(self):
        for _ in range(1000):
            r, c = self.rng.integers(0, self.grid_size, size=2)
            pos  = (int(r), int(c))
            if pos not in self.food_positions and pos != tuple(self.agent_pos):
                self.food_positions.add(pos)
                self.grid[pos] = self.FOOD
                return
