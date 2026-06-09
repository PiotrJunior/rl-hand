import gymnasium as gym
import numpy as np
from pynput import keyboard
from env import OrcaHandEnv

# 1. Dynamic registration so LeRobot can find the environment
gym.register(
    id="OrcaHandEnv-v0",
    entry_point="env:OrcaHandEnv",
)

# 2. Re-use our keyboard listener logic from before
current_teleop_action = np.zeros(3, dtype=np.float32)

def on_press(key):
    global current_teleop_action
    try:
        if key.char == 'w': current_teleop_action[0] = 1.0
        elif key.char == 's': current_teleop_action[0] = -1.0
        elif key.char == 'a': current_teleop_action[1] = -1.0
        elif key.char == 'd': current_teleop_action[1] = 1.0
        elif key.char == 'r': current_teleop_action[2] = 1.0
        elif key.char == 'f': current_teleop_action[2] = -1.0
    except AttributeError: pass

def on_release(key):
    global current_teleop_action
    try:
        if key.char in ['w', 's']: current_teleop_action[0] = 0.0
        if key.char in ['a', 'd']: current_teleop_action[1] = 0.0
        if key.char in ['r', 'f']: current_teleop_action[2] = 0.0
    except AttributeError: pass

# Start the keyboard listener in the background immediately
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# 3. Define a custom Policy class that LeRobot will use during recording.
# Instead of a neural network, this policy simply outputs our keyboard commands.
class KeyboardTeleopPolicy:
    def __init__(self):
        # Neural network policies in LeRobot track internal state, we don't need it
        self.expected_image_keys = ["top"] 

    def select_action(self, observation):
        """
        LeRobot calls this function at every step (30 times per second).
        We return the current 20-dimensional vector filled with keyboard inputs.
        """
        full_action = np.zeros(20, dtype=np.float32)
        # Inject our 3D keyboard controls into the hand_position slots
        full_action[17:20] = current_teleop_action
        return full_action

    def reset(self):
        pass

if __name__ == "__main__":
    from lerobot.scripts.lerobot_record import record
    # Importujemy klasy konfiguracyjne, których wymaga walidator LeRobot
    # from lerobot.configs.robot import PreTouchRobotConfig
    # from lerobot.configs.cli import RecordConfig
    
    print("Starting LeRobot Record Utility...")
    print("Controls: WASD for XY plane, R/F for Z axis. Press Q in Genesis window to exit.")

    # 1. Tworzymy uproszczoną konfigurację "robota" (w naszym przypadku to tylko środowisko Gym)
    # LeRobot wymaga, aby RecordConfig miał zdefiniowane pola 'robot' i 'dataset'
    robot_cfg = dict(
        type="env",
        env_id="OrcaHandEnv-v0"
    )

    # 2. Składamy pełny obiekt RecordConfig, który Draccus bez problemu zwaliduje
    cfg = dict(
        robot=None,
        dataset=dict(
            type="local",
        ),
        # repo_id="piotr-maksymiuk/orca_hand_ds",  # Twoja nazwa repozytorium na HF
        num_episodes=10,
        fps=30,
        warmup_steps=10,
        root="data"
    )

    # 3. Uruchamiamy funkcję przekazując gotowy obiekt konfiguracji jako argument kluczowy 'cfg'
    record(cfg=cfg)
    
    listener.stop()