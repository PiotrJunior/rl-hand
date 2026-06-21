import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2

# Importujemy naszego robota, którego już napisałeś!
from robot import OrcaRobot, OrcaRobotConfig

class OrcaHandEnv(gym.Env):
    """
    Gymnasium wrapper for the Orca Hand simulation in MuJoCo.
    Compliant with GymHIL standards.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # 👇 DODANO: image_obs=True oraz **kwargs na ewentualne inne ukryte parametry
    def __init__(
        self, 
        render_mode=None, 
        use_gripper=False, 
        gripper_penalty=0.0, 
        image_obs=True, 
        **kwargs
    ):
        super().__init__()
        self.render_mode = render_mode
        self.image_obs = image_obs  # Możesz zostawić tę zmienną do celów dokumentacyjnych
        
        # 1. Inicjalizacja robota z Twojego kodu
        config = OrcaRobotConfig(show_viewer=(render_mode == "human"), render=False)
        self.robot = OrcaRobot(config)
        self.robot.connect()

        # 2. Definicja przestrzeni (Gym Spaces)
        self.num_motors = 20 # 17 palców + 3 osie XYZ
        
        # Akcje (dla uproszczenia załóżmy od -1 do 1)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_motors,), dtype=np.float32
        )

        # Obserwacje (Pozycje + Obrazy z kamer)
        obs_dict = {}
        # Stan silników
        obs_dict["agent_pos"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_motors,), dtype=np.float32
        )
        
        # Kamery - odczytujemy dynamicznie z Twojego robota
        for cam_name, shape in self.robot._cameras_ft.items():
            h, w, c = shape
            obs_dict[cam_name] = spaces.Box(
                low=0, high=255, shape=(h, w, c), dtype=np.uint8
            )
            
        self.observation_space = spaces.Dict(obs_dict)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot.calibrate() # Resetuje pozycje w MuJoCo
        return self._get_obs(), {}

    def step(self, action):
        # 1. Konwersja wektora do słownika dla Twojego robota (bezpieczne łatanie braków!)
        action_dict = {}
        for i, name in enumerate(self.robot.action_features.keys()):
            if i < len(action):
                action_dict[name] = float(action[i])
            else:
                # Fallback: Jeśli HIL-SERL wyśle wektor [0,0,0], wypełnij resztę zerami
                action_dict[name] = 0.0
                
        # 2. Wykonanie kroku symulacji
        self.robot.send_action(action_dict)
        obs = self._get_obs()

        # W RL zwykle sami definiujemy nagrodę. Tutaj używamy 0, 
        # bo HIL-SERL użyje Reward Classifiera!
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """Pobiera i formatuje obserwacje dla środowiska Gym."""
        raw_obs = self.robot.get_observation()
        gym_obs = {}
        
        # Ekstrakcja agent_pos do jednej tablicy
        agent_pos = np.zeros(self.num_motors, dtype=np.float32)
        for i, name in enumerate(self.robot.action_features.keys()):
            agent_pos[i] = raw_obs.get(name, 0.0)
        gym_obs["agent_pos"] = agent_pos
        
        # Ekstrakcja obrazów (natywny format HWC)
        for cam_name in self.robot.config.cameras.keys():
            gym_obs[cam_name] = raw_obs[cam_name]
            
        return gym_obs

    def render(self):
        if self.render_mode == "human" and self.robot.viewer is not None:
            self.robot.viewer.sync()

    def close(self):
        self.robot.disconnect()