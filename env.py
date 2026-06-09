# env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from scene_builder import build_orca_scene 

class OrcaHandEnv(gym.Env):
    """
    A custom Gymnasium environment that wraps the Genesis simulation.
    Fully compatible with Hugging Face LeRobot imitation/reinforcement learning frameworks.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        
        # LeRobot REQUIRES flat continuous action spaces.
        # Dimension mapping: 17 (joint_positions) + 3 (hand_position) = 20 total dimensions.
        # Actions are normalized between [-1.0, 1.0] for optimal neural network convergence.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(20,), dtype=np.float32
        )
        
        # Observation space structured to match LeRobot naming conventions:
        # - 'agent_pos': Flat proprioceptive/state vector (must match action_space dim).
        # - 'pixels': Nested dictionary containing visual inputs (HWC format for Gym).
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32),
            "pixels": spaces.Dict({
                "top": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            })
        })
        
        # Instantiate the isolated Genesis simulation world
        self.scene, self.entities = build_orca_scene(show_viewer=False)
        
        # Set up quick-access aliases for the simulation entities
        self.cube = self.entities["cube"]
        self.sphere = self.entities["sphere"]
        self.box = self.entities["box"]
        self.cam = self.entities["camera"]

    def _get_to_numpy(self, tensor_or_array):
        """
        Helper method to safely cast physics engine positions to a flat NumPy array,
        abstracting whether Genesis is running on CPU or GPU (PyTorch tensor).
        """
        if isinstance(tensor_or_array, torch.Tensor):
            return tensor_or_array.detach().cpu().numpy().flatten()[:3]
        return np.array(tensor_or_array).flatten()[:3]

    def _get_obs(self):
        """
        Gathers and formats current simulation state into LeRobot-compliant tensors.
        """
        # Fetch the 3D position of the controlled blue cube
        cube_pos = self._get_to_numpy(self.cube.get_pos())
        
        # Construct the state vector. We pad the first 17 indices (joints placeholder)
        # and store the 3D proxy hand position in the final 3 slots.
        agent_pos = np.zeros(20, dtype=np.float32)
        agent_pos[17:20] = cube_pos 
        
        # Render the RGB frame from the overhead camera
        rgb, _, _, _ = self.cam.render(rgb=True)
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()
            
        return {
            "agent_pos": agent_pos,
            "pixels": {"top": rgb.astype(np.uint8)}
        }

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial configuration.
        """
        super().reset(seed=seed)
        
        # Reset spatial positions of the objects
        self.cube.set_pos([0.0, 0.0, 0.1])
        self.sphere.set_pos([0.0, 0.5, 0.1])
        
        # Step the physics engine once to apply changes before generating the observation
        self.scene.step()
        
        return self._get_obs(), {}

    def step(self, action):
        """
        Executes one environment step based on policy actions.
        """
        # Extract the hand position commands (the final 3 dimensions of the action vector)
        hand_position_action = action[17:20]
        
        # Apply incremental (delta) control to move the blue cube smoothly
        current_cube_pos = self._get_to_numpy(self.cube.get_pos())
        new_cube_pos = current_cube_pos + hand_position_action * 0.05 # 0.05 acts as speed scaling
        self.cube.set_pos(new_cube_pos)
        
        # Advance physics simulation
        self.scene.step()
        
        # Collect the post-action observation data
        obs = self._get_obs()
        
        # Reward and Termination Logic (Calculate Euclidean distance between sphere and goal box)
        sphere_pos = self._get_to_numpy(self.sphere.get_pos())
        box_pos = self._get_to_numpy(self.box.get_pos())
        distance_to_box = np.linalg.norm(sphere_pos - box_pos)
        
        reward = 0.0
        terminated = False
        truncated = False
        
        # Success condition: Terminate with a sparse reward if the red sphere enters the box radius
        if distance_to_box < 0.15:
            reward = 100.0
            terminated = True
            
        return obs, reward, terminated, truncated, {"is_success": terminated}

    def render(self):
        """
        Renders the current frame for video recording/dataset logging utilities.
        """
        if self.render_mode == "rgb_array":
            rgb, _, _, _ = self.cam.render(rgb=True)
            if isinstance(rgb, torch.Tensor):
                return rgb.detach().cpu().numpy().astype(np.uint8)
            return rgb.astype(np.uint8)
        

def make_env(n_envs: int = 1, use_async_envs: bool = False, cfg=None):
    """
    Główny punkt wejścia wywoływany przez skrypty LeRobot.
    Tworzy jedno lub wiele równoległych środowisk.
    """
    def env_fn():
        return OrcaHandEnv()

    if n_envs == 1:
        return gym.vector.SyncVectorEnv([env_fn])
    else:
        # Przydatne przy masowym zbieraniu danych w symulatorach typu Genesis/Isaac
        return gym.vector.AsyncVectorEnv([env_fn for _ in range(n_envs)])