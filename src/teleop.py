import dataclasses
import numpy as np
from typing import Any
from pynput import keyboard

from lerobot.teleoperators import Teleoperator, TeleoperatorConfig
# 👇 1. IMPORT KLASY ZDARZEŃ
from lerobot.teleoperators.utils import TeleopEvents 

@TeleoperatorConfig.register_subclass("keyboard_wasd")
@dataclasses.dataclass
class KeyboardTeleoperatorConfig(TeleoperatorConfig):
    pass

class KeyboardTeleoperator(Teleoperator):
    config_class = KeyboardTeleoperatorConfig
    name = "keyboard_teleop"

    def __init__(self, config: KeyboardTeleoperatorConfig):
        super().__init__(config)
        self._is_connected = False
        self.listener = None
        
        # Płaski wektor o rozmiarze 20 (indeksy 17, 18, 19 to X, Y, Z)
        self.current_action = np.zeros(20, dtype=np.float32)
        
        # 👇 2. ZMIENNE STANU DLA HIL-SERL
        self._is_intervening = False
        self._rerecord_episode = False
        self._reset_episode = False

    @property
    def action_features(self) -> dict:
        """Informuje HIL-SERL o dokładnym rozmiarze bazy danych dla akcji (20)."""
        names = [f"hand_joint_{i:02d}.pos" for i in range(1, 18)] + ["x.pos", "y.pos", "z.pos"]
        return {
            "dtype": "float32",
            "shape": (20,),
            "names": names
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected: return
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
        self._is_connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None: pass
    def configure(self) -> None: pass

    def _on_press(self, key):
        try:
            # Sterowanie pozycją (WASD + RF)
            if key.char == 'w': self.current_action[17] = 1.0
            elif key.char == 's': self.current_action[17] = -1.0
            elif key.char == 'a': self.current_action[18] = -1.0
            elif key.char == 'd': self.current_action[18] = 1.0
            elif key.char == 'r': self.current_action[19] = 1.0
            elif key.char == 'f': self.current_action[19] = -1.0
            
            # 👇 3. PRZYCISKI SPECJALNE DLA HIL-SERL
            elif key.char == 'i': # 'i' jak Intervention
                self._is_intervening = True
            elif key.char == 'x': # 'x' jak eXit / Reset
                self._reset_episode = True
            elif key.char == 'c': # 'c' jak Cancel / Rerecord
                self._rerecord_episode = True
                
        except AttributeError: pass

    def _on_release(self, key):
        try:
            if key.char in ['w', 's']: self.current_action[17] = 0.0
            if key.char in ['a', 'd']: self.current_action[18] = 0.0
            if key.char in ['r', 'f']: self.current_action[19] = 0.0
            
            # Puszczenie przycisku 'i' kończy interwencję
            if key.char == 'i':
                self._is_intervening = False
                
        except AttributeError: pass

    def get_action(self) -> np.ndarray:
        return self.current_action.copy()

    def get_teleop_events(self) -> dict[TeleopEvents, bool]:
        """
        Zwraca słownik ze zdarzeniami sterującymi.
        Flagi resetu są czyszczone natychmiast po odczytaniu (tzw. one-shot).
        """
        events = {
            TeleopEvents.IS_INTERVENTION: self._is_intervening,
            # Usunięto TeleopEvents.RESET_EPISODE - nie istnieje w tej wersji API
            TeleopEvents.RERECORD_EPISODE: self._rerecord_episode,
        }
        
        # Wyczyść flagi zdarzeń po jednorazowym odczycie
        self._rerecord_episode = False
        
        return events

    def send_feedback(self, feedback: dict[str, Any]) -> None: pass

    def disconnect(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
        self._is_connected = False