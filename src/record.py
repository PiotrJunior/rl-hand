import time
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Importujemy nasze stworzone wcześniej klasy
# (Załóżmy, że zapisałeś je w plikach robot.py i teleop.py)
from robot import GenesisOrcaRobot, GenesisOrcaRobotConfig
from teleop import KeyboardTeleoperator, KeyboardTeleoperatorConfig

def main():
    # 1. Ustawienia nagrywania
    fps = 30
    episodes_to_record = 10
    repo_id = "twoj_nick_hf/orca_hand_genesis_ds"
    
    print("Initializing components...")
    
    # 2. Inicjalizacja instancji sprzętowych
    robot_config = GenesisOrcaRobotConfig(show_viewer=True)
    robot = GenesisOrcaRobot(robot_config)
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
                
                # KROK A: Pobranie intencji od człowieka
                action_dict = teleop.get_action()
                
                # KROK B: Wysłanie akcji do "sprzętu" (silnika fizyki)
                robot.send_action(action_dict)
                
                # KROK C: Pobranie stanu świata po wykonaniu ruchu
                obs_dict = robot.get_observation()
                
                # Genesis renderuje obrazy jako (H, W, C). LeRobot Dataset wymaga (C, H, W).
                # Transformujemy obraz przed zapisem:
                img_hwc = obs_dict["pixels/top"]
                img_chw = np.transpose(img_hwc, (2, 0, 1))
                
                # KROK D: Zapisanie pełnej klatki do datasetu
                # dataset.add_frame({
                #     "observation.state": obs_dict["agent_pos"],
                #     "observation.images.top": img_chw,
                #     "action": action_dict["action"]
                # })
                
                # KROK E: Sprawdzenie warunku sukcesu
                # Ponieważ porzuciliśmy Gymnasium, sami musimy stwierdzić, kiedy epizod się kończy.
                # Wyciągamy na chwilę pozycje bezpośrednio z silnika na potrzeby tego skryptu:
                sphere_pos = robot._get_to_numpy(robot.entities["sphere"].get_pos())
                box_pos = robot._get_to_numpy(robot.entities["box"].get_pos())
                distance = np.linalg.norm(sphere_pos - box_pos)
                
                if distance < 0.15:
                    print(f"Goal reached in {step} steps! Saving episode...")
                    done = True
                    
                # KROK F: Utrzymanie stałego FPS (aby dane były użyteczne dla AI)
                elapsed = time.perf_counter() - start_time
                time_to_wait = (1.0 / fps) - elapsed
                if time_to_wait > 0:
                    time.sleep(time_to_wait)
                    
                step += 1
                
            # Na koniec epizodu powiadamiamy dataset, że ten ciąg klatek stanowi jedną całość
            # dataset.save_episode()

    # 5. Zakończenie pracy
    print("\nAll episodes recorded! Consolidating dataset...")
    # Ta funkcja łączy wszystkie klatki i wideo do zoptymalizowanych formatów hdf5 / mp4 / safetensors
    # dataset.consolidate()
    print(f"Success! Dataset is ready at local path: {dataset.root}")

if __name__ == "__main__":
    main()