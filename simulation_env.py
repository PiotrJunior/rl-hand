import gymnasium as gym
from gymnasium import spaces
import numpy as np

class OrcaHandEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        
        self.action_space = spaces.Dict({
            "joint_positions": spaces.Box(
                low=-1.0, high=1.0, shape=(17,), dtype=np.float32
            ),
            "hand_position": spaces.Box(
                low=-1.0, high=1.0, shape=(3,), dtype=np.float32
            )
        })
        
        self.observation_space = spaces.Dict({
            "position": spaces.Dict({
                "hand_position": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "joint_positions": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
                ),
            }),
            "cameras": spaces.Dict({
                "top": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            })
        })
        
        # np. self.simulator = GenesisWorld(urdf_path)

    def _get_obs(self):
        pass

    def reset(self, seed=None, options=None):
        # Gymnasium wymaga wywołania super().reset w celu ustawienia generatora losowego
        super().reset(seed=seed)
        
        # TUTAJ: Zresetuj pozycję robota i obiektów w symulatorze
        
        obs = self._get_obs()
        info = {} # Dodatkowe metadane (opcjonalne)
        return obs, info

    def step(self, action):
        obs = self._get_obs()
        
        reward = 0.0 
        
        terminated = False  # True, jeśli zadanie zostało wykonane (np. sukces)
        truncated = False   # True, jeśli skończył się czas (np. osiągnięto limit kroków)
        
        info = {"is_success": terminated}
        
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            pass