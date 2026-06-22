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
    Maps A/D to X-Axis, W/S to Y-Axis, and RF keys to Z-Axis base movements.
    """
    config_class = KeyboardTeleoperatorConfig
    name = "keyboard_teleop"

    def __init__(self, config: KeyboardTeleoperatorConfig):
        super().__init__(config)
        self._is_connected = False
        self.listener = None
        
        # Flat vector of size 20 (Indices 17, 18, 19 map to x.pos, y.pos, z.pos base translations)
        self.current_action = np.zeros(20, dtype=np.float32)
        
        # Native HIL-SERL State Variables
        self._is_intervening = False
        self._terminate_episode = False
        self._success = False
        self._rerecord_episode = False
        
        self._i_pressed = False 

    @property
    def action_features(self) -> dict:
        """Informs HIL-SERL about the exact shape of the action space (20 DoF)."""
        hand_names = [
            "right_wrist.pos", "right_p-abd.pos", "right_p-mcp.pos", 
            "right_p-pip.pos", "right_r-abd.pos", "right_r-mcp.pos", 
            "right_r-pip.pos", "right_m-abd.pos", "right_m-mcp.pos", 
            "right_m-pip.pos", "right_i-abd.pos", "right_i-mcp.pos", 
            "right_i-pip.pos", "right_t-cmc.pos", "right_t-abd.pos", 
            "right_t-mcp.pos", "right_t-pip.pos",
        ]
        aux_names = ["x.pos", "y.pos", "z.pos"]
        return {"dtype": "float32", "shape": (20,), "names": hand_names + aux_names}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected: return
        self._print_instructions()
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
        self._is_connected = True

    def _print_instructions(self):
        """Prints the teleoperation manual to the terminal upon connection."""
        print("\n" + "="*55)
        print("🕹️  ORCA HAND - KEYBOARD TELEOPERATION")
        print("="*55)
        print("BASE/WRIST MOVEMENT (SWAPPED):")
        print("  [A] / [D] : Forward / Backward (X-Axis)")
        print("  [W] / [S] : Left / Right (Y-Axis)")
        print("  [R] / [F] : Up / Down (Z-Axis)")
        print("-" * 55)
        print("NATIVE HIL-SERL CONTROLS:")
        print("  [I]     : Toggle Intervention (AI <-> Human)")
        print("  [C]     : FAIL (Rerecord) - Trash data & try again")
        print("  [Q]     : QUIT - End episode as failure (No save)")
        print("  [SPACE] : SUCCESS - Save good data & reset")
        print("="*55 + "\n")

    @property
    def is_calibrated(self) -> bool: return True
    def calibrate(self) -> None: pass
    def configure(self) -> None: pass

    def _on_press(self, key):
        try:
            # 1. Intervention Toggle
            if key.char == 'i':
                if not self._i_pressed:
                    self._is_intervening = not self._is_intervening
                    self._i_pressed = True
                    status = "🔴 ACTIVE (Human)" if self._is_intervening else "🟢 INACTIVE (AI)"
                    print(f"[TELEOP] Intervention: {status}")

            # 2. Base Velocity Control (Swapped Axes)
            elif key.char == 'w': self.current_action[18] = 1.0   # Y-Axis (Left)
            elif key.char == 's': self.current_action[18] = -1.0  # Y-Axis (Right)
            elif key.char == 'a': self.current_action[17] = -1.0  # X-Axis (Backward)
            elif key.char == 'd': self.current_action[17] = 1.0   # X-Axis (Forward)
            elif key.char == 'r': self.current_action[19] = 1.0   # Z-Axis (Up)
            elif key.char == 'f': self.current_action[19] = -1.0  # Z-Axis (Down)
            
            # 3. Native Rerecord (Failure - Try Again)
            elif key.char == 'c': 
                self._rerecord_episode = True
                self._terminate_episode = True
                print("[TELEOP] ⚠️ FAIL: Trashing data and restarting episode...")

            # 4. Native Quit (Failure - Move On)
            elif key.char == 'q':
                self._terminate_episode = True
                self._success = False
                print("[TELEOP] ⏭️ QUIT: Ending episode without success...")
                
        except AttributeError: 
            # 5. Native Success (Save and Move On)
            if key == keyboard.Key.space:
                self._success = True
                self._terminate_episode = True
                print("[TELEOP] 🏆 SUCCESS: Saving episode data...")
                print("[TELEOP] 🏆 SUCCESS: Saving episode...")

    def _on_release(self, key):
        try:
            if key.char == 'i': 
                self._i_pressed = False
            elif key.char in ['w', 's']: 
                self.current_action[18] = 0.0
            elif key.char in ['a', 'd']: 
                self.current_action[17] = 0.0
            elif key.char in ['r', 'f']: 
                self.current_action[19] = 0.0
        except AttributeError: pass

    def get_action(self) -> np.ndarray:
        return self.current_action.copy()

    def get_teleop_events(self) -> dict[TeleopEvents, bool]:
        """
        Natively broadcasts lifecycle events to LeRobot's trajectory processor.
        """
        events = {
            TeleopEvents.IS_INTERVENTION: self._is_intervening,
            TeleopEvents.TERMINATE_EPISODE: self._terminate_episode,
            TeleopEvents.SUCCESS: self._success,
            TeleopEvents.RERECORD_EPISODE: self._rerecord_episode,
        }
        
        # Reset one-shot event flags immediately after sending
        self._terminate_episode = False
        self._success = False
        self._rerecord_episode = False
        
        return events

    def send_feedback(self, feedback: dict[str, Any]) -> None: pass

    def disconnect(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
        self._is_connected = False