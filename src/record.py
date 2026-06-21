import sys
import time
import numpy as np
from pathlib import Path
import questionary
import shutil

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Hardware class imports
from robot import OrcaRobot, OrcaRobotConfig
from teleop import KeyboardTeleoperator, KeyboardTeleoperatorConfig


# ==============================================================================
# 1. HARDWARE REGISTRIES
# ==============================================================================

ROBOT_REGISTRY = {
    "orca_genesis": {
        "title": "🤖 Orca Hand (MuJoCo)",
        "class": OrcaRobot,
        "config_class": OrcaRobotConfig,
        "default_args": {"show_viewer": False},
    },
}

TELEOP_REGISTRY = {
    "keyboard": {
        "title": "⌨️  Keyboard (WASD + RF)",
        "class": KeyboardTeleoperator,
        "config_class": KeyboardTeleoperatorConfig,
        "default_args": {"id": "keyboard"},
    },
}


# ==============================================================================
# 2. CORE RECORDING PIPELINE
# ==============================================================================

def main(config: dict):
    print("\nInitializing components...")
    
    selected_robot = ROBOT_REGISTRY[config["robot_type"]]
    robot_cfg = selected_robot["config_class"](**selected_robot["default_args"])
    robot = selected_robot["class"](robot_cfg)
    
    selected_teleop = TELEOP_REGISTRY[config["teleop_type"]]
    teleop_cfg = selected_teleop["config_class"](**selected_teleop["default_args"])
    teleop = selected_teleop["class"](teleop_cfg)

    # --------------------------------------------------------------------------
    # BUDOWANIE SCHEMATU ZGODNEGO ZE STANDARDEM LEROBOT V3
    # --------------------------------------------------------------------------
    dynamic_features = {}
    
    # 1. Grupujemy wszystkie silniki w jeden wektor i pobieramy ich nazwy
    motor_keys = list(robot.action_features.keys())
    num_motors = len(motor_keys)
    
    dynamic_features["action"] = {
        "dtype": "float32",
        "shape": (num_motors,),
        "names": motor_keys  # Tutaj LeRobot zapamięta nazwy przegubów!
    }
    dynamic_features["observation.state"] = {
        "dtype": "float32",
        "shape": (num_motors,),
        "names": motor_keys
    }

    # 2. Mapujemy kamery dodając prefix i używając natywnego formatu HWC
    for key, ft_type in robot.observation_features.items():
        if isinstance(ft_type, tuple) and len(ft_type) == 3:
            h, w, c = ft_type
            dynamic_features[f"observation.images.{key}"] = {
                "dtype": "video",
                "shape": (h, w, c), # Natywny format z OpenCV/MuJoCo
                "names": ["height", "width", "channels"]
            }

    dataset = None
    if config["save_trajectory"]:
        print(f"[INFO] Initializing LeRobot dataset for: {config['repo_id']}")
        dataset = LeRobotDataset.create(
            repo_id=config["repo_id"],
            fps=config["fps"],
            features=dynamic_features,
        )

    print("Setup complete. Starting recording loop...")

    # --------------------------------------------------------------------------
    # PĘTLA NAGRYWANIA
    # --------------------------------------------------------------------------
    with robot, teleop:
        for ep_idx in range(config["episodes_to_record"]):
            print(f"\n--- Starting Episode {ep_idx + 1}/{config['episodes_to_record']} ---")
            
            robot.calibrate()
            done = False
            step = 0
            
            while not done:
                start_time = time.perf_counter()
                
                # 1. Pobranie komend z teleoperatora
                action_dict = teleop.get_action()
                
                # Jeśli klawiatura wysyła stary płaski wektor, rzutujemy go na nazwany słownik
                if "action" in action_dict and isinstance(action_dict["action"], (np.ndarray, list)):
                    flat_actions = action_dict["action"]
                    action_to_send = {name: float(flat_actions[i]) if i < len(flat_actions) else 0.0 for i, name in enumerate(motor_keys)}
                else:
                    action_to_send = action_dict

                # 2. Wysyłamy do robota i pobieramy zweryfikowane akcje oraz stan
                applied_action = robot.send_action(action_to_send)
                obs_dict = robot.get_observation()
                
                # 3. Zapisujemy ramkę
                if config["save_trajectory"] and dataset is not None:
                    frame_payload = {}
                    
                    # a) Pakowanie pozycji i akcji w czyste wektory numpy
                    obs_state_array = np.zeros(num_motors, dtype=np.float32)
                    action_array = np.zeros(num_motors, dtype=np.float32)
                    
                    for i, motor_name in enumerate(motor_keys):
                        obs_state_array[i] = obs_dict.get(motor_name, 0.0)
                        action_array[i] = applied_action.get(motor_name, 0.0)
                        
                    frame_payload["observation.state"] = obs_state_array
                    frame_payload["action"] = action_array
                    frame_payload["task"] = "Get the red sphere into the green box!"
                    
                    # b) Pakowanie obrazów HWC (Bez transpozycji!)
                    for key, ft_type in robot.observation_features.items():
                        if isinstance(ft_type, tuple) and len(ft_type) == 3:
                            dataset_cam_key = f"observation.images.{key}"
                            # Przekazujemy obraz prosto z MuJoCo
                            frame_payload[dataset_cam_key] = obs_dict[key]
                    
                    dataset.add_frame(frame_payload)
                
                if step > 300: 
                    done = True
                
                elapsed = time.perf_counter() - start_time
                time_to_wait = (1.0 / config["fps"]) - elapsed
                if time_to_wait > 0:
                    time.sleep(time_to_wait)
                    
                step += 1
                
            if config["save_trajectory"] and dataset is not None:
                print(f"[INFO] Saving episode {ep_idx + 1}...")
                dataset.save_episode()

    if config["save_trajectory"] and dataset is not None:
        print("\nAll episodes recorded! Consolidating dataset...")
        dataset.finalize()
        print("[SUCCESS] Data has been consolidated and saved locally!")
        
        if config["push_to_hub"]:
            print(f"🚀 Uploading dataset repository to Hugging Face Hub ({config['repo_id']})...")
            dataset.push_to_hub()
            print("[SUCCESS] Dataset successfully uploaded to the Hub!")


# ==============================================================================
# 3. USER INTERFACE (Questionary Parsing)
# ==============================================================================

def parse_cli_arguments() -> dict:
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

    if not answers:
        return {}

    answers["episodes_to_record"] = int(answers["episodes_to_record"])

    if answers["save_trajectory"]:
        repo_details = questionary.form(
            repo_id=questionary.text(
                "📝 LeRobot repository ID:",
                default="PiotrJunior/orcahand_ds"
            )
        ).ask()
        if not repo_details:
            return {}
        answers.update(repo_details)

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
        config_dict = parse_cli_arguments()
        if not config_dict:
            print("\n[ABORTED] Session not configured.")
            sys.exit(0)
            
        main(config_dict)
        
    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted. Safe exit.")
        sys.exit(0)