import time
import numpy as np
from pathlib import Path
# from lerobot.datasets.lerobot_dataset import LeRobotDataset

from robot import OrcaRobot, OrcaRobotConfig
from teleop import KeyboardTeleoperator, KeyboardTeleoperatorConfig

def main():
    # 1. Ustawienia nagrywania
    fps = 30
    episodes_to_record = 10
    repo_id = "twoj_nick_hf/orca_hand_genesis_ds"
    
    print("Initializing components...")
    
    # 2. Inicjalizacja instancji sprzętowych
    robot_config = OrcaRobotConfig(show_viewer=False)
    robot = OrcaRobot(robot_config)
    teleop_config = KeyboardTeleoperatorConfig(id="keyboard")
    teleop = KeyboardTeleoperator(teleop_config)

    # 3. Utworzenie obiektu LeRobotDataset
    # Ręcznie mapujemy strukturę słowników zwracanych przez robot.get_observation()
    # dataset = LeRobotDataset.create(
    #     repo_id=repo_id,
    #     fps=fps,
    #     features={
    #         "observation.state": {
    #             "dtype": "float32", 
    #             "shape": (20,), 
    #             "names": None
    #         },
    #         "observation.images.top": {
    #             "dtype": "uint8", 
    #             "shape": (3, 480, 640), # LeRobot operuje wewnętrznie na formacie CHW 
    #             "names": ["c", "h", "w"]
    #         },
    #         "action": {
    #             "dtype": "float32", 
    #             "shape": (20,), 
    #             "names": None
    #         }
    #     }
    # )

    print("Setup complete. Starting recording loop...")
    print("Controls: WASD (XY) and R/F (Z). Get the red sphere into the green box!")

    # 4. Używamy Context Managers, które same wywołają connect() i disconnect()
    with robot, teleop:
        for ep_idx in range(episodes_to_record):
            print(f"\n--- Starting Episode {ep_idx + 1}/{episodes_to_record} ---")
            
            # Kalibracja środowiska przed każdym odcinkiem (w symulacji to reset pozycji)
            robot.calibrate()
            
            done = False
            step = 0
            
            # Pętla pojedynczego epizodu
            while not done:
                start_time = time.perf_counter()
                
                action_dict = teleop.get_action()
                robot.send_action(action_dict)
                obs_dict = robot.get_observation()
                
                # Genesis renderuje obrazy jako (H, W, C). LeRobot Dataset wymaga (C, H, W).
                # img_hwc = obs_dict["pixels/top"]
                # img_chw = np.transpose(img_hwc, (2, 0, 1))
                
                # dataset.add_frame({
                #     "observation.state": obs_dict["agent_pos"],
                #     "observation.images.top": img_chw,
                #     "action": action_dict["action"]
                # })
                
                elapsed = time.perf_counter() - start_time
                time_to_wait = (1.0 / fps) - elapsed
                if time_to_wait > 0:
                    time.sleep(time_to_wait)
                    
                step += 1
                
            # dataset.save_episode()

    print("\nAll episodes recorded! Consolidating dataset...")
    # dataset.consolidate()

if __name__ == "__main__":
    main()