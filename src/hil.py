import logging
import torch
import numpy as np
import gymnasium as gym

# Import our custom hardware/driver and teleoperation configurations
import robot
import teleop

# Import the core LeRobot gym manipulator runtime execution script
import lerobot.rl.gym_manipulator

# ==============================================================================
# LEROBOT RUNTIME MONKEY PATCHES (Adapting HIL-SERL for 20-DoF Hands)
# ==============================================================================

# PATCH 1: Override the hardcoded 3-DoF action space configuration limits
original_setup_spaces = lerobot.rl.gym_manipulator.RobotEnv._setup_spaces

def safe_setup_spaces(self):
    """Dynamically scales the Gym environment action bounds to match our motor count."""
    # Execute the original setup sequence to construct observation image structures
    original_setup_spaces(self)
    
    # Scale action space bounds dynamically to accommodate our 20 actuators
    action_dim = len(self.robot.bus.motors)
    self.action_space = gym.spaces.Box(
        low=-np.ones(action_dim, dtype=np.float32),
        high=np.ones(action_dim, dtype=np.float32),
        shape=(action_dim,),
        dtype=np.float32,
    )

# Override the default class method
lerobot.rl.gym_manipulator.RobotEnv._setup_spaces = safe_setup_spaces


# PATCH 2: Expand hardcoded neutral actions ([0.0, 0.0, 0.0]) to 20 dimensions
original_step_env = lerobot.rl.gym_manipulator.step_env_and_process_transition

def safe_step_env(env, transition, action, env_processor, action_processor):
    """Pads low-dimensional placeholder actions to match our 20-DoF control input topology."""
    target_dim = env.action_space.shape[0]
    
    # If the framework emits a fallback 3-DoF placeholder, pad it up with zeros
    if len(action) < target_dim:
        padded_action = torch.zeros(target_dim, dtype=torch.float32)
        padded_action[:len(action)] = action
        action = padded_action
        
    # Route the corrected 20-dimensional action back into the official control step
    return original_step_env(env, transition, action, env_processor, action_processor)

# Override the default environment step function
lerobot.rl.gym_manipulator.step_env_and_process_transition = safe_step_env


# PATCH 3: Hard-reset MuJoCo physics simulation context on episode boundaries
original_env_reset = lerobot.rl.gym_manipulator.RobotEnv.reset

def safe_env_reset(self, *args, **kwargs):
    """Intercepts environment resets to fully reload object and joint states in MuJoCo."""
    # Call our custom hardware driver's calibration routine to reset qpos, qvel, and ctrl
    if hasattr(self.robot, "calibrate"):
        print("[HIL PATCH] Executing physical environment context hard-reset in MuJoCo...")
        self.robot.calibrate()
        
    # Proceed with LeRobot's standard procedural reset trajectory rules
    return original_env_reset(self, *args, **kwargs)

# Override the default environment reset function
lerobot.rl.gym_manipulator.RobotEnv.reset = safe_env_reset

# ==============================================================================
# MAIN RUNTIME ENTRYPOINT
# ==============================================================================

# Silence redundant low-level logging spam from MuJoCo internal processes
logging.getLogger("mujoco").propagate = False

if __name__ == "__main__":
    print("[INFO] 20-DoF runtime monkey patches successfully applied. Starting HIL-SERL...")
    # Execute the primary training/recording loop engine
    lerobot.rl.gym_manipulator.main()