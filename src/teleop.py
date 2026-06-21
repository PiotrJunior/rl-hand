import abc
import dataclasses
import numpy as np
from typing import Any
from pynput import keyboard

from lerobot.teleoperators import Teleoperator, TeleoperatorConfig

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
        
        # Inicjalizujemy pusty słownik, który idealnie pokrywa się z wybranymi kluczami akcji w Robot
        self.current_action = {
            "x.pos": 0.0,
            "y.pos": 0.0,
            "z.pos": 0.0
        }

    @property
    def action_features(self) -> dict[str, type]:
        """Kształt akcji deklarowany przez teleoperator."""
        # Ponieważ wysyłamy pojedyncze zmienne osiowe, definiujemy je jako typ float.
        return {
            "x.pos": float,
            "y.pos": float,
            "z.pos": float
        }

    @property
    def feedback_features(self) -> dict:
        """Klawiatura nie posiada Force Feedback, więc nic nie przyjmuje."""
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        """Uruchamia nasłuchiwanie klawiatury w tle."""
        if self._is_connected:
            return

        self.listener = keyboard.Listener(
            on_press=self._on_press, 
            on_release=self._on_release
        )
        self.listener.start()
        self._is_connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _on_press(self, key):
        """Mapuje wciśnięcia klawiszy bezpośrednio na konkretne nazwane akcje."""
        try:
            # Oś X (Przód/Tył)
            if key.char == 'w': self.current_action["y.pos"] = 1.0
            elif key.char == 's': self.current_action["y.pos"] = -1.0
            # Oś Y (Lewo/Prawo)
            elif key.char == 'a': self.current_action["x.pos"] = -1.0
            elif key.char == 'd': self.current_action["x.pos"] = 1.0
            # Oś Z (Góra/Dół)
            elif key.char == 'r': self.current_action["z.pos"] = 1.0
            elif key.char == 'f': self.current_action["z.pos"] = -1.0
        except AttributeError:
            pass # Ignorowanie klawiszy specjalnych np. Shift, Cmd

    def _on_release(self, key):
        """Zeruje konkretną nazwę osi po puszczeniu klawisza."""
        try:
            if key.char in ['w', 's']: self.current_action["x.pos"] = 0.0
            if key.char in ['a', 'd']: self.current_action["y.pos"] = 0.0
            if key.char in ['r', 'f']: self.current_action["z.pos"] = 0.0
        except AttributeError:
            pass

    def get_action(self) -> dict[str, float]:
        """
        Zwraca kopię słownika z gotowymi, nazwanymi komendami akcji.
        Zwracamy kopię, by pętla zewnętrzna nie nadpisała referencji w pamięci.
        """
        if not self._is_connected:
            raise RuntimeError("Teleoperator is not connected.")
            
        return self.current_action.copy()

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
        self._is_connected = False