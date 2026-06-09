import dataclasses
import os
import cv2
import numpy as np
import mujoco
import mujoco.viewer
import logging

from lerobot.robots import Robot, RobotConfig

# Wyłączenie powielania logów przez MuJoCo
logging.getLogger("mujoco").propagate = False

@RobotConfig.register_subclass("mujoco_orca")
@dataclasses.dataclass
class OrcaRobotConfig(RobotConfig):
    show_viewer: bool = True
    xml_path: str = "scene/scene.xml"  # Zmień na właściwą ścieżkę do Twojego głównego pliku MJCF

class OrcaRobot(Robot):
    name = "mujoco_orca_robot"
    config_class = OrcaRobotConfig
    hand_dof_count = 17

    def __init__(self, config: OrcaRobotConfig):
        super().__init__(config)

        self.config = config
        self._is_connected = False
        self._is_calibrated = False
        
        # Zmienne dla silnika MuJoCo
        self.model = None
        self.data = None
        self.renderer = None
        self.viewer = None
        self.camera_id = 0

    @property
    def observation_features(self) -> dict:
        return {
            "agent_pos": (20,),          
            "pixels/top": (3, 480, 480)  
        }

    @property
    def action_features(self) -> dict:
        return {
            "action": (20,)
        }

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            return

        if not os.path.exists(self.config.xml_path):
            raise FileNotFoundError(f"Nie znaleziono pliku sceny MuJoCo: {self.config.xml_path}")

        # 1. Inicjalizacja modelu i danych MuJoCo
        self.model = mujoco.MjModel.from_xml_path(self.config.xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 2. Inicjalizacja renderera off-screen dla zbierania danych (LeRobot)
        self.renderer = mujoco.Renderer(self.model, height=480, width=480)
        
        # 3. Wyszukiwanie odpowiedniej kamery
        self.camera_id = 0

        # 4. Uruchomienie okna podglądu (Passive Viewer)
        if self.config.show_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.viewer.cam.fixedcamid = self.camera_id

        # 5. Zapisanie stanu początkowego na potrzeby kalibracji (resetu)
        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()
        
        self._is_connected = True
        
        if calibrate:
            self.calibrate()

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def calibrate(self) -> None:
        """
        Resetuje środowisko fizyczne do stanu początkowego.
        """
        if not self._is_connected:
            raise RuntimeError("Nie można kalibrować przed nawiązaniem połączenia!")
            
        # Przywrócenie zapisanych pozycji i prędkości
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        
        # Przeliczenie kinematyki prostej, aby MuJoCo zaktualizowało fizykę
        mujoco.mj_forward(self.model, self.data)
        
        # Odświeżenie okna
        if self.config.show_viewer and self.viewer is not None:
            self.viewer.sync()
            
        self._is_calibrated = True

    def configure(self) -> None:
        """Puste, niewymagane w prostej symulacji."""
        pass

    def get_observation(self) -> dict[str, np.ndarray]:
        """
        Renderuje klatkę z kamery, wyświetla ją w OpenCV i wyciąga pozycje stawów z MuJoCo.
        """
        if not self._is_connected:
            raise RuntimeError("Robot nie jest połączony.")
        
        # Pobranie do 20 pozycji stawów (qpos)
        agent_pos = np.zeros(20, dtype=np.float32)
        available_qpos = len(self.data.qpos)
        # print(f"Available qpos: {available_qpos}")
        # agent_pos[:available_qpos] = self.data.qpos[:available_qpos]
        
        # Renderowanie obrazu z MuJoCo (w formacie RGB)
        self.renderer.update_scene(self.data, camera=self.camera_id)
        raw_image = self.renderer.render()
        
        # --- RYSOWANIE I WYŚWIETLANIE OPENCV ---
        # Wyświetlenie okna podglądu kamery LeRobot
        opencv_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("LeRobot Camera - pixels/top", opencv_image)
        cv2.waitKey(1)

        # Transpozycja z HWC do CHW dla kompatybilności z LeRobot
        lerobot_pixels = np.transpose(raw_image, (2, 0, 1)).astype(np.uint8)
        
        return {
            "agent_pos": agent_pos,
            "pixels/top": lerobot_pixels
        }

    def send_action(self, action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Aplikuje wektor akcji do silników MuJoCo i wykonuje krok symulacji.
        """
        if not self._is_connected:
            raise RuntimeError("Robot nie jest połączony.")

        action_vector = action["action"]
        
        # Aplikacja sygnału kontrolnego (do 20 silników w data.ctrl)
        available_actuators = min(20, self.model.nu)
        if available_actuators > 0:
            self.data.ctrl[:available_actuators] = action_vector[:available_actuators]
            
        # Wykonanie kroku fizyki w MuJoCo
        mujoco.mj_step(self.model, self.data)
        
        # Synchronizacja okna wizualnego z nowym stanem fizycznym
        if self.config.show_viewer and self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()
            
        return action

    def disconnect(self) -> None:
        """
        Bezpieczne zamknięcie renderera i okna symulacji.
        """
        self._is_connected = False
        self._is_calibrated = False
        
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
            
        cv2.destroyAllWindows()