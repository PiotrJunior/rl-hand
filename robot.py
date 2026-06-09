import abc
import dataclasses
import numpy as np
import torch
import genesis as gs
from pathlib import Path

from lerobot.robots import Robot, RobotConfig
from lerobot.types import RobotAction, RobotObservation
    

@RobotConfig.register_subclass("genesis_orca")
@dataclasses.dataclass
class GenesisOrcaRobotConfig(RobotConfig):
    show_viewer: bool = True
    fps: int = 30

class GenesisOrcaRobot(Robot):
    """
    Wirtualny robot zaimplementowany w silniku Genesis, ściśle trzymający się 
    abstrakcji sprzętowej LeRobot.
    """
    name = "genesis_orca_robot"
    def __init__(self, config: GenesisOrcaRobotConfig):
        # Wywołanie konstruktora bazowego jest wymagane
        super().__init__(config)
        self.show_viewer = config.show_viewer
        self._is_connected = False
        self._is_calibrated = False
        
        # Referencje do obiektów symulacji
        self.scene = None
        self.cube = None
        self.sphere = None
        self.box = None
        self.cam = None

    @property
    def observation_features(self) -> dict:
        """
        Zwracamy płaski słownik zgodnie ze standardem LeRobot dla kamer i stanów.
        Wymiary muszą pasować do tego, co faktycznie wypluwa `get_observation`.
        """
        return {
            "agent_pos": (20,),          # Wektor 20-wymiarowy
            "pixels/top": (480, 640, 3)  # Obraz w formacie HWC (Wysokość, Szerokość, Kanały)
        }

    @property
    def action_features(self) -> dict:
        """
        Jakich akcji oczekuje robot. W naszym przypadku to płaski wektor 20-wymiarowy.
        """
        return {
            "action": (20,)
        }

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        """
        Nawiązanie 'połączenia', czyli de facto inicjalizacja silnika Genesis 
        i zbudowanie fizycznej sceny.
        """
        if self._is_connected:
            return

        try:
            gs.init(backend=gs.gpu)
        except RuntimeError:
            pass # Genesis jest już zainicjowane w tym procesie

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.81)),
            show_viewer=self.show_viewer
        )

        # Budowa obiektów na scenie
        self.scene.add_entity(gs.morphs.Plane())
        
        # Czerwona sfera
        self.sphere = self.scene.add_entity(
            morph=gs.morphs.Sphere(pos=(0.0, 0.5, 0.1), radius=0.08),
            surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0))
        )
        
        # Zielone pudełko (cel)
        self.box = self.scene.add_entity(
            morph=gs.morphs.Box(pos=(0.5, 0.5, 0.05), size=(0.3, 0.3, 0.1), fixed=True),
            surface=gs.surfaces.Default(color=(0.0, 1.0, 0.0)),
        )
        
        # Niebieski sześcian (efektor końcowy sterowany przez 'robota')
        self.cube = self.scene.add_entity(
            morph=gs.morphs.Box(pos=(0.0, 0.0, 0.1), size=(0.1, 0.1, 0.1)),
            surface=gs.surfaces.Default(color=(0.0, 0.0, 1.0))
        )
        
        self.cam = self.scene.add_camera(
            res=(640, 480), pos=(1.5, 1.5, 1.2), lookat=(0.3, 0.3, 0.1), fov=45
        )

        self.scene.build()
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
            
        self.cube.set_pos([0.0, 0.0, 0.1])
        self.sphere.set_pos([0.0, 0.5, 0.1])
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

        cube_pos = self._get_to_numpy(self.cube.get_pos())
        
        agent_pos = np.zeros(20, dtype=np.float32)
        agent_pos[17:20] = cube_pos 
        
        rgb, _, _, _ = self.cam.render(rgb=True)
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()

        return {
            "agent_pos": agent_pos,
            "pixels/top": rgb.astype(np.uint8)
        }

    def send_action(self, action: RobotAction) -> RobotAction:
        """
        Rozpakowuje słownik akcji, aplikuje je do silnika fizycznego i posuwa 
        czas symulacji o jedną klatkę w przód.
        """
        if not self._is_connected:
            raise RuntimeError("Robot is not connected.")

        # Wyciągamy z wektora ostatnie 3 elementy dla pozycji X, Y, Z
        action_vector = action["action"]
        hand_position_action = action_vector[17:20]
        
        # Obliczamy nową pozycję
        current_cube_pos = self._get_to_numpy(self.cube.get_pos())
        new_cube_pos = current_cube_pos + hand_position_action * 0.05
        
        # Aplikujemy pozycję i wykonujemy krok fizyki
        self.cube.set_pos(new_cube_pos)
        self.scene.step()
        
        # W prawdziwym robocie zwracamy tu faktycznie osiągniętą pozycję (po clampingu/limitach).
        # W symulacji możemy po prostu zwrócić wejściową akcję.
        return action

    def disconnect(self) -> None:
        """
        Sprzątanie zasobów.
        """
        self._is_connected = False
        self._is_calibrated = False
        # Genesis nie posiada agresywnej metody niszczenia instancji w locie,
        # ale zrzucamy referencje, by odciążyć pamięć.
        self.scene = None
        self.cube = None
        self.sphere = None
        self.box = None
        self.cam = None