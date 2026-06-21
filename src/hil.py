import logging
import torch
import numpy as np
import gymnasium as gym

# Importujemy customowe klasy
import robot
import teleop

# Importujemy wadliwą bibliotekę LeRobot
import lerobot.rl.gym_manipulator

# ==============================================================================
# LEROBOT MONKEY PATCHES (Łatki uodparniające HIL-SERL na dłonie z 20 DoF)
# ==============================================================================

# ŁATKA 1: Naprawa na twardo wpisanej przestrzeni akcji
original_setup_spaces = lerobot.rl.gym_manipulator.RobotEnv._setup_spaces

def safe_setup_spaces(self):
    # Wywołaj oryginalną funkcję, aby zainicjować obrazy itp.
    original_setup_spaces(self)
    
    # Podmień przestrzeń akcji na prawdziwy wymiar Twoich silników
    action_dim = len(self.robot.bus.motors)
    self.action_space = gym.spaces.Box(
        low=-np.ones(action_dim, dtype=np.float32),
        high=np.ones(action_dim, dtype=np.float32),
        shape=(action_dim,),
        dtype=np.float32,
    )

lerobot.rl.gym_manipulator.RobotEnv._setup_spaces = safe_setup_spaces


# ŁATKA 2: Naprawa na twardo wpisanej akcji neutralnej ([0,0,0])
original_step_env = lerobot.rl.gym_manipulator.step_env_and_process_transition

def safe_step_env(env, transition, action, env_processor, action_processor):
    # Sprawdź, jakiego wymiaru akcji oczekuje załatane środowisko
    target_dim = env.action_space.shape[0]
    
    # Jeśli LeRobot wysłał swoje małe [0,0,0], wypełnij resztę zerami dla dłoni
    if len(action) < target_dim:
        padded_action = torch.zeros(target_dim, dtype=torch.float32)
        padded_action[:len(action)] = action
        action = padded_action
        
    # Przekaż naprawioną, 20-wymiarową akcję z powrotem do oficjalnej pętli
    return original_step_env(env, transition, action, env_processor, action_processor)

lerobot.rl.gym_manipulator.step_env_and_process_transition = safe_step_env

# ==============================================================================

# Wyciszamy spamujące logi z podglądu MuJoCo
logging.getLogger("mujoco").propagate = False

if __name__ == "__main__":
    print("[INFO] Załatano ograniczenia wymiarów w HIL-SERL. Uruchamiam system z 20 DoF...")
    # Odpalamy główną logikę z załatanymi klasami
    lerobot.rl.gym_manipulator.main()