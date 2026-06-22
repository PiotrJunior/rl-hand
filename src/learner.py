# run_learner.py
import robot   # registers mujoco_orca with RobotConfig
import teleop  # registers keyboard_wasd with TeleoperatorConfig

# Now launch the learner — mujoco_orca is registered so config parsing won't fail
from lerobot.rl.learner import train_cli

if __name__ == "__main__":
    train_cli()