import sys
import time
import numpy as np
from pathlib import Path
import questionary
import shutil

# LeRobot imports
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Hardware class imports
from robot import OrcaRobot, OrcaRobotConfig
from teleop import KeyboardTeleoperator, KeyboardTeleoperatorConfig


# ==============================================================================
# 1. HARDWARE REGISTRIES WITH DATASET STRUCTURE DEFINITIONS (FEATURES)
# ==============================================================================

ROBOT_REGISTRY = {
    "orca_genesis": {
        "title": "🤖 Orca Hand (MuJoCo)",
        "class": OrcaRobot,
        "config_class": OrcaRobotConfig,
        "default_args": {"show_viewer": False},
        # Define observation features for this specific robot
        "features": {
            "observation.state": {"dtype": "float32", "shape": (20,), "names": None},
            "observation.images.top": {"dtype": "video", "shape": (3, 480, 480), "names": ["c", "h", "w"]},
        },
        # Mapping: key from robot obs_dict -> target key in LeRobot dataset
        "image_mapping": {
            "obs_key": "pixels/top",
            "dataset_key": "observation.images.top"
        }
    },
}

TELEOP_REGISTRY = {
    "keyboard": {
        "title": "⌨️  Keyboard (WASD + RF)",
        "class": KeyboardTeleoperator,
        "config_class": KeyboardTeleoperatorConfig,
        "default_args": {"id": "keyboard"},
        # Keyboard generates a full 20 DoF action vector
        "features": {
            "action": {"dtype": "float32", "shape": (20,), "names": None}
        }
    },
}


# ==============================================================================
# 2. CORE RECORDING PIPELINE (Main Loop)
# ==============================================================================

def main(config: dict):
    """
    Execution function. Accepts configuration and dynamically builds 
    the dataset structure based on selected components.
    """
    print("\nInitializing components...")
    
    # 1. Dynamically instantiate the robot and teleoperation interface
    selected_robot = ROBOT_REGISTRY[config["robot_type"]]
    robot_cfg = selected_robot["config_class"](**selected_robot["default_args"])
    robot = selected_robot["class"](robot_cfg)
    
    selected_teleop = TELEOP_REGISTRY[config["teleop_type"]]
    teleop_cfg = selected_teleop["config_class"](**selected_teleop["default_args"])
    teleop = selected_teleop["class"](teleop_cfg)

    # 2. DYNAMICALLY COMPOSE FEATURES STRUCTURE
    dynamic_features = {}
    dynamic_features.update(selected_robot["features"])
    dynamic_features.update(selected_teleop["features"])

    dataset = None
    if config["save_trajectory"]:
        print(f"[INFO] Initializing LeRobot dataset for: {config['repo_id']}")
        print(f"[INFO] Detected dynamic features structure: {list(dynamic_features.keys())}")
        
        # Safe initialization (directory check already passed in prompt step)
        dataset = LeRobotDataset.create(
            repo_id=config["repo_id"],
            fps=config["fps"],
            features=dynamic_features,
        )

    print("Setup complete. Starting recording loop...")

    # 3. Recording loop inside context managers
    with robot, teleop:
        for ep_idx in range(config["episodes_to_record"]):
            print(f"\n--- Starting Episode {ep_idx + 1}/{config['episodes_to_record']} ---")
            
            robot.calibrate()
            done = False
            step = 0
            
            while not done:
                start_time = time.perf_counter()
                
                action_dict = teleop.get_action()
                robot.send_action(action_dict)
                obs_dict = robot.get_observation()
                
                # Frame logging (if enabled)
                if config["save_trajectory"] and dataset is not None:
                    img_obs_key = selected_robot["image_mapping"]["obs_key"]
                    img_dataset_key = selected_robot["image_mapping"]["dataset_key"]
                    
                    img_chw = obs_dict[img_obs_key]
                    
                    # Add the frame to the dataset using dynamic keys
                    dataset.add_frame({
                        "observation.state": obs_dict["agent_pos"],
                        img_dataset_key: img_chw,
                        "action": action_dict["action"],
                        "task": "Get the red sphere into the green box!"
                    })
                
                # Loop safeguard (simulation cutoff)
                if step > 300: 
                    done = True
                
                elapsed = time.perf_counter() - start_time
                time_to_wait = (1.0 / config["fps"]) - elapsed
                if time_to_wait > 0:
                    time.sleep(time_to_wait)
                    
                step += 1
                
            # End of episode - save the episode to disk
            if config["save_trajectory"] and dataset is not None:
                print(f"[INFO] Saving episode {ep_idx + 1}...")
                dataset.save_episode()

    # Consolidate and optionally upload the dataset after the session ends
    if config["save_trajectory"] and dataset is not None:
        print("\nAll episodes recorded! Consolidating dataset...")
        dataset.finalize()
        print("[SUCCESS] Data has been consolidated and saved locally!")
        
        # Trigger Hugging Face Hub upload if requested
        if config["push_to_hub"]:
            print(f"🚀 Uploading dataset repository to Hugging Face Hub ({config['repo_id']})...")
            dataset.push_to_hub()
            print("[SUCCESS] Dataset successfully uploaded to the Hub!")


# ==============================================================================
# 3. USER INTERFACE (Questionary Parsing)
# ==============================================================================

def parse_cli_arguments() -> dict:
    """Builds the menu based on registries and returns the configuration."""
    print("✨ ORCA HAND DYNAMIC RECORDING WIZARD ✨\n")

    answers = questionary.form(
        robot_type=questionary.select(
            "🤖 Select robot configuration:",
            choices=[questionary.Choice(title=cfg["title"], value=key) for key, cfg in ROBOT_REGISTRY.items()]
        ),
        teleop_type=questionary.select(
            "🎮 Select teleoperation system:",
            choices=[questionary.Choice(title=cfg["title"], value=key) for key, cfg in TELEOP_REGISTRY.items()]
        ),
        fps=questionary.select(
            "⏱️  Frequency (FPS):",
            choices=[
                questionary.Choice(title="30 Hz", value=30),
                questionary.Choice(title="50 Hz", value=50)
            ],
            default=30
        ),
        episodes_to_record=questionary.text(
            "🔢 Number of episodes:",
            default="10",
            validate=lambda text: text.isdigit() and int(text) > 0 or "Please enter a number greater than 0!"
        ),
        save_trajectory=questionary.confirm(
            "💾 Save trajectory?",
            default=True
        )
    ).ask()

    # If user interrupts via Ctrl+C
    if not answers:
        return {}

    # Convert episode count from str to int
    answers["episodes_to_record"] = int(answers["episodes_to_record"])

    # Conditional path prompts (only if save_trajectory=True)
    if answers["save_trajectory"]:
        repo_details = questionary.form(
            repo_id=questionary.text(
                "📝 LeRobot repository ID:",
                default="PiotrJunior/orca_hand_genesis_ds"
            )
        ).ask()
        if not repo_details:
            return {}
        answers.update(repo_details)

        # INTERACTIVE OVERWRITE LOGIC
        cache_path = Path.home() / ".cache" / "huggingface" / "lerobot" / answers["repo_id"]
        if cache_path.exists():
            overwrite = questionary.confirm(
                f"⚠️  Dataset path '{answers['repo_id']}' already exists locally. Overwrite it?",
                default=False
            ).ask()
            
            if not overwrite:
                print("\n[ABORTED] Session cancelled to protect existing local data.")
                return {}
            else:
                shutil.rmtree(cache_path)
                print(f"[INFO] Cleaned existing local cache path: {cache_path}")

        # HUGGING FACE HUB PROMPT
        push_hub = questionary.confirm(
            "🚀 Upload dataset to Hugging Face Hub when finished?",
            default=False
        ).ask()
        answers["push_to_hub"] = push_hub
    else:
        answers["repo_id"] = None
        answers["push_to_hub"] = False

    return answers


# ==============================================================================
# 4. ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    try:
        # Step 1: Build configuration from UI
        config_dict = parse_cli_arguments()
        if not config_dict:
            print("\n[ABORTED] Session not configured.")
            sys.exit(0)
            
        # Step 2: Pass clean dictionary to main()
        main(config_dict)
        
    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted. Safe exit.")
        sys.exit(0)