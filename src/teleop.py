import dataclasses
import numpy as np
from typing import Any
from pynput import keyboard

from lerobot.teleoperators import Teleoperator, TeleoperatorConfig
from lerobot.teleoperators.utils import TeleopEvents 

@TeleoperatorConfig.register_subclass("keyboard_wasd")
@dataclasses.dataclass
class KeyboardTeleoperatorConfig(TeleoperatorConfig):
    pass

class KeyboardTeleoperator(Teleoperator):
    """
    A custom keyboard teleoperator for the Orca Hand.
    Maps WASD/RF keys to base movements and handles HIL-SERL intervention events.
    """
    config_class = KeyboardTeleoperatorConfig
    name = "keyboard_teleop"

    def __init__(self, config: KeyboardTeleoperatorConfig):
        super().__init__(config)
        self._is_connected = False
        self.listener = None
        
        # Flat vector of size 20 (Indices 17, 18, 19 map to X, Y, Z base translations)
        self.current_action = np.zeros(20, dtype=np.float32)
        
        # HIL-SERL State Variables
        self._is_intervening = False
        self._rerecord_episode = False
        
        # Debounce flag to prevent rapid toggling when the 'I' key is held down
        self._i_pressed = False 

    @property
    def action_features(self) -> dict:
        """Informs HIL-SERL about the exact shape of the action space (20 DoF)."""
        hand_names = [
            "right_wrist.pos",
            "right_p-abd.pos",
            "right_p-mcp.pos",
            "right_p-pip.pos",
            "right_r-abd.pos",
            "right_r-mcp.pos",
            "right_r-pip.pos",
            "right_m-abd.pos",
            "right_m-mcp.pos",
            "right_m-pip.pos",
            "right_i-abd.pos",
            "right_i-mcp.pos",
            "right_i-pip.pos",
            "right_t-cmc.pos",
            "right_t-abd.pos",
            "right_t-mcp.pos",
            "right_t-pip.pos",
        ]
        aux_names = ["x.pos", "y.pos", "z.pos"]
        
        names = hand_names + aux_names
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
        if self._is_connected: 
            return
        
        self._print_instructions()
        
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
        self._is_connected = True

    def _print_instructions(self):
        """Prints the teleoperation manual to the terminal upon connection."""
        print("\n" + "="*55)
        print("🕹️  ORCA HAND - KEYBOARD TELEOPERATION")
        print("="*55)
        print("BASE/WRIST MOVEMENT:")
        print("  [W] / [S] : Forward / Backward (Y-Axis)")
        print("  [A] / [D] : Left / Right (X-Axis)")
        print("  [R] / [F] : Up / Down (Z-Axis)")
        print("-" * 55)
        print("HIL-SERL CONTROLS:")
        print("  [I] : Intervention Toggle (Switch between AI and Human)")
        print("  [C] : Cancel current episode and Rerecord")
        print("="*55 + "\n")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None: 
        pass
        
    def configure(self) -> None: 
        pass

    def _on_press(self, key):
        try:
            # 1. Intervention Toggle
            if key.char == 'i':
                if not self._i_pressed:
                    self._is_intervening = not self._is_intervening
                    self._i_pressed = True
                    status = "🔴 ACTIVE (Human in control)" if self._is_intervening else "🟢 INACTIVE (AI in control)"
                    print(f"[TELEOP] Intervention: {status}")

            # 2. Base Velocity Control
            elif key.char == 'w': self.current_action[18] = 1.0
            elif key.char == 's': self.current_action[18] = -1.0
            elif key.char == 'a': self.current_action[17] = -1.0
            elif key.char == 'd': self.current_action[17] = 1.0
            elif key.char == 'r': self.current_action[19] = 1.0
            elif key.char == 'f': self.current_action[19] = -1.0
            
            # 3. Episode Cancellation Flag
            elif key.char == 'c': 
                self._rerecord_episode = True
                print("[TELEOP] ⚠️ Requested episode cancellation (Rerecord)!")
                
        except AttributeError: 
            pass

    def _on_release(self, key):
        try:
            # Reset the debounce flag when the 'I' key is physically released
            if key.char == 'i':
                self._i_pressed = False
                
            # Stop movement when velocity keys are released
            elif key.char in ['w', 's']: self.current_action[18] = 0.0
            elif key.char in ['a', 'd']: self.current_action[17] = 0.0
            elif key.char in ['r', 'f']: self.current_action[19] = 0.0
                
        except AttributeError: 
            pass

    def get_action(self) -> np.ndarray:
        return self.current_action.copy()

    def get_teleop_events(self) -> dict[TeleopEvents, bool]:
        """
        Returns a dictionary of control events expected by the HIL-SERL pipeline.
        Reset flags are cleared immediately after being read (one-shot).
        """
        events = {
            TeleopEvents.IS_INTERVENTION: self._is_intervening,
            TeleopEvents.RERECORD_EPISODE: self._rerecord_episode,
        }
        
        # Clear the one-shot event flag after reading
        self._rerecord_episode = False
        return events

    def send_feedback(self, feedback: dict[str, Any]) -> None: 
        pass

    def disconnect(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
        self._is_connected = False