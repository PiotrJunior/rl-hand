import dataclasses
import cv2
import numpy as np
import torch
import genesis as gs
import logging

from lerobot.robots import Robot, RobotConfig
from lerobot.types import RobotAction, RobotObservation

from scene import build_orca_scene    

logging.getLogger("genesis").propagate = False

@RobotConfig.register_subclass("genesis_orca")
@dataclasses.dataclass
class GenesisOrcaRobotConfig(RobotConfig):
    show_viewer: bool = True

class GenesisOrcaRobot(Robot):
    name = "genesis_orca_robot"
    config_class = GenesisOrcaRobotConfig

    def __init__(self, config: GenesisOrcaRobotConfig):
        super().__init__(config)

        self.config = config
        self._is_connected = False
        self._is_calibrated = False

    @property
    def observation_features(self) -> dict:
        return {
            "agent_pos": (20,),          # Wektor 20-wymiarowy
            "pixels/top": (480, 480, 3)  # Obraz w formacie HWC (Wysokość, Szerokość, Kanały)
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

        self.scene, self.entities = build_orca_scene(self.config.show_viewer)
        self.initial_hand_pos = self.entities["hand"].get_pos()
        self.initial_object_pos = self.entities["object"].get_pos()
        self._is_connected = True
        
        if calibrate:
            self.calibrate()

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def calibrate(self) -> None:
        """
        W symulacji kalibracja sprowadza się do upewnienia się, 
        że obiekty są na swoich pozycjach startowych.
        """
        if not self._is_connected:
            raise RuntimeError("Cannot calibrate before connecting!")
            
        self.entities["hand"].set_pos(self.initial_hand_pos)
        self.entities["object"].set_pos(self.initial_object_pos)
        self.scene.step()
        self._is_calibrated = True

    def configure(self) -> None:
        """Konfiguracja parametrów runtime (w naszym przypadku opcjonalna)"""
        pass

    def _get_to_numpy(self, tensor_or_array) -> np.ndarray:
        if isinstance(tensor_or_array, torch.Tensor):
            return tensor_or_array.detach().cpu().numpy().flatten()[:3]
        return np.array(tensor_or_array).flatten()[:3]

    def get_observation(self) -> RobotObservation:
        """
        Pobiera klatkę z kamery i pozycję, a następnie mapuje je ściśle 
        do kluczy zadeklarowanych w `observation_features`.
        """
        if not self._is_connected:
            raise RuntimeError("Robot is not connected.")
        
        frame, _, _, _ = self.entities["camera"].render(rgb=True)
        cv2.waitKey(20)
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()

        hand_pos = self._get_to_numpy(self.entities["hand"].get_pos())
        
        agent_pos = np.zeros(20, dtype=np.float32)
        agent_pos[17:20] = hand_pos 

        return {
            "agent_pos": agent_pos,
            "pixels/top": frame.astype(np.uint8)
        }

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self._is_connected:
            raise RuntimeError("Robot is not connected.")

        # Wyciągamy z wektora ostatnie 3 elementy dla pozycji X, Y, Z
        action_vector = action["action"]
        hand_position_action = action_vector[17:20]
        
        # Obliczamy nową pozycję
        current_hand_pos = self._get_to_numpy(self.entities["hand"].get_pos())
        new_hand_pos = current_hand_pos + hand_position_action * 0.05
        
        # Aplikujemy pozycję i wykonujemy krok fizyki
        # self.entities["hand"].set_pos(new_hand_pos)
        self.scene.step()
        
        # W prawdziwym robocie zwracamy tu faktycznie osiągniętą pozycję (po clampingu/limitach).
        # W symulacji możemy po prostu zwrócić wejściową akcję.
        return action

    def disconnect(self) -> None:
        self._is_connected = False
        self._is_calibrated = False
        # Genesis nie posiada agresywnej metody niszczenia instancji w locie,
        self.scene.viewer.stop()
        cv2.destroyAllWindows()