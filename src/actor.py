import robot
import teleop
import lerobot.rl.gym_manipulator

# Your existing monkey patches
original_setup_spaces = lerobot.rl.gym_manipulator.RobotEnv._setup_spaces
def safe_setup_spaces(self):
    original_setup_spaces(self)
    import gymnasium as gym
    import numpy as np
    action_dim = len(self.robot.bus.motors)
    print(f"[HIL PATCH] Overriding action space to dim={action_dim}")
    self.action_space = gym.spaces.Box(
        low=-np.ones(action_dim, dtype=np.float32),
        high=np.ones(action_dim, dtype=np.float32),
        shape=(action_dim,),
        dtype=np.float32,
    )
lerobot.rl.gym_manipulator.RobotEnv._setup_spaces = safe_setup_spaces

import torch
original_step_env = lerobot.rl.gym_manipulator.step_env_and_process_transition
def safe_step_env(env, transition, action, env_processor, action_processor):
    # Action comes in as [1, 20] (batched), squeeze to [20]
    if action.dim() == 2 and action.shape[0] == 1:
        action = action.squeeze(0)
    
    target_dim = env.action_space.shape[0]
    action_dim = action.shape[0]
    
    if action_dim != target_dim:
        print(f"[HIL PATCH] action dim mismatch: {action_dim} -> {target_dim}")
        new_action = torch.zeros(target_dim, dtype=torch.float32)
        copy_dim = min(action_dim, target_dim)
        new_action[:copy_dim] = action[:copy_dim]
        action = new_action
    
    return original_step_env(env, transition, action, env_processor, action_processor)
lerobot.rl.gym_manipulator.step_env_and_process_transition = safe_step_env

original_env_reset = lerobot.rl.gym_manipulator.RobotEnv.reset
def safe_env_reset(self, *args, **kwargs):
    if hasattr(self.robot, "calibrate"):
        print("[HIL PATCH] Hard-resetting MuJoCo environment...")
        self.robot.calibrate()
    return original_env_reset(self, *args, **kwargs)
lerobot.rl.gym_manipulator.RobotEnv.reset = safe_env_reset

from lerobot.rl.actor import actor_cli

if __name__ == "__main__":
    actor_cli()