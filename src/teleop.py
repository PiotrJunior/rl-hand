import abc
import dataclasses
import numpy as np
from typing import Any
from pathlib import Path
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
        
        # Inicjalizujemy pusty, 20-wymiarowy wektor akcji 
        # (indeksy 17, 18, 19 będą modyfikowane przez klawiaturę)
        self.current_action = np.zeros(20, dtype=np.float32)

    @property
    def action_features(self) -> dict:
        """Kształt akcji generowanych przez ten teleoperator (musi pasować do robota)."""
        return {"action": (20,)}

    @property
    def feedback_features(self) -> dict:
        """Klawiatura nie posiada sprzężenia zwrotnego (Force Feedback), więc zwracamy pusty słownik."""
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
        """Klawiatura nie wymaga kalibracji, zawsze zwracamy True."""
        return True

    def calibrate(self) -> None:
        """Brak akcji dla klawiatury."""
        pass

    def configure(self) -> None:
        """Brak specjalnej konfiguracji sprzętowej dla klawiatury."""
        pass

    def _on_press(self, key):
        """Callback modyfikujący wektor akcji w momencie wciśnięcia klawisza."""
        try:
            # Oś X (Przód/Tył) -> indeks 17
            if key.char == 'w': self.current_action[17] = 1.0
            elif key.char == 's': self.current_action[17] = -1.0
            # Oś Y (Lewo/Prawo) -> indeks 18
            elif key.char == 'a': self.current_action[18] = -1.0
            elif key.char == 'd': self.current_action[18] = 1.0
            # Oś Z (Góra/Dół) -> indeks 19
            elif key.char == 'r': self.current_action[19] = 1.0
            elif key.char == 'f': self.current_action[19] = -1.0
        except AttributeError:
            pass # Ignorujemy klawisze specjalne (Shift, Ctrl itp.)

    def _on_release(self, key):
        """Callback zerujący daną oś po puszczeniu klawisza."""
        try:
            if key.char in ['w', 's']: self.current_action[17] = 0.0
            if key.char in ['a', 'd']: self.current_action[18] = 0.0
            if key.char in ['r', 'f']: self.current_action[19] = 0.0
        except AttributeError:
            pass

    def get_action(self) -> dict[str, np.ndarray]:
        """
        Zwraca aktualny stan wygenerowany przez teleoperator.
        Zwracamy kopię wektora, aby zapobiec mutacjom przez referencję w dalszym potoku LeRobot.
        """
        if not self._is_connected:
            raise RuntimeError("Teleoperator is not connected.")
            
        return {"action": self.current_action.copy()}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Puste wywołanie - klawiatura nie wibruje i nie stawia oporu na podstawie akcji."""
        pass

    def disconnect(self) -> None:
        """Zatrzymuje nasłuchiwanie klawiatury i sprząta zasoby."""
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
        self._is_connected = False