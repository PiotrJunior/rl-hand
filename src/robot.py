import sys
import os
import dataclasses
import logging
from functools import cached_property
import cv2
import numpy as np
import mujoco
import mujoco.viewer

from lerobot.robots import Robot, RobotConfig
from lerobot.cameras.camera import CameraConfig

# Disable redundant MuJoCo logging duplicates
logging.getLogger("mujoco").propagate = False

# ==============================================================================
# 1. ROBOT CONFIGURATION
# ==============================================================================

@RobotConfig.register_subclass("mujoco_orca")
@dataclasses.dataclass
class OrcaRobotConfig(RobotConfig):
    show_viewer: bool = False
    render: bool = True  
    xml_path: str = "scene/scene.xml"
    
    cameras: dict[str, CameraConfig] = dataclasses.field(default_factory=lambda: {
        "base": CameraConfig(height=480, width=640, fps=30),
        "wrist": CameraConfig(height=480, width=640, fps=30)
    })

# ==============================================================================
# 2. CORE ROBOT CLASS (Multi-Camera Compliant)
# ==============================================================================

class MuJoCoMockBus:
    """Udaje sprzętową płytkę sterującą dla środowiska HIL-SERL."""
    def __init__(self, robot_instance):
        self._robot = robot_instance
        # Skrypt HIL-SERL oczekuje nazw silników bez końcówki '.pos'
        self.motors = {key.replace('.pos', ''): None for key in robot_instance._motors_ft.keys()}
        self.is_connected = True
        self.is_calibrated = True

    def sync_read(self, command: str) -> dict:
        if command == "Present_Position":
            obs = self._robot.get_observation()
            return {k: float(obs.get(f"{k}.pos", 0.0)) for k in self.motors.keys()}
        return {}

    def sync_write(self, command: str, target_dict: dict):
        if command == "Goal_Position":
            action = {f"{k}.pos": float(v) for k, v in target_dict.items()}
            self._robot.send_action(action)
            
    def connect(self): pass
    def disconnect(self, *args): pass


# ZMODYFIKUJ OrcaRobot:
class OrcaRobot(Robot):
    name = "mujoco_orca_robot"
    config_class = OrcaRobotConfig
    hand_dof_count = 17

    def __init__(self, config: OrcaRobotConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._is_calibrated = False
        
        self.model = None
        self.data = None
        self.viewer = None
        self.renderers = {}
        self.camera_name_to_id = {}

        # 👇 1. ZAINICJUJ MOCK BUS
        self.bus = MuJoCoMockBus(self)

    # 👇 2. DODAJ WŁAŚCIWOŚĆ KAMER DLA HIL-SERL
    @property
    def cameras(self):
        return self.config.cameras

    # --------------------------------------------------------------------------
    # MOTOR & CAMERA FEATURES SCHEMA
    # --------------------------------------------------------------------------

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Generates explicit named features for all 20 degrees of freedom."""
        hand_names = [f"hand_joint_{i:02d}.pos" for i in range(1, 18)]
        aux_names = ["x.pos", "y.pos", "z.pos"]
        
        all_joint_names = tuple(hand_names + aux_names)
        return dict.fromkeys(all_joint_names, float)

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Dynamically generates camera feature shapes in native HWC format."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) 
            for cam in self.config.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Combines named motor states and native HWC camera shapes."""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Actions mirror the named motor joint states exactly."""
        return self._motors_ft

    # --------------------------------------------------------------------------
    # HARDWARE LIFECYCLE MANAGEMENT
    # --------------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._is_connected and (len(self.renderers) > 0)

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            return

        if not os.path.exists(self.config.xml_path):
            raise FileNotFoundError(f"MuJoCo scene file not found at: {self.config.xml_path}")

        # 1. Initialize MuJoCo model and execution structures
        self.model = mujoco.MjModel.from_xml_path(self.config.xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 2. Set up dedicated render contexts for each configured camera stream
        for cam_name, cam_cfg in self.config.cameras.items():
            # Resolve the string camera name from XML to its internal numerical ID
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id == -1:
                raise ValueError(f"Camera identifier '{cam_name}' not found in MJCF scene layout xml.")
            
            self.camera_name_to_id[cam_name] = cam_id
            
            # Each renderer instance allocates separate resolution-optimized frame buffers
            self.renderers[cam_name] = mujoco.Renderer(
                self.model, 
                height=cam_cfg.height, 
                width=cam_cfg.width
            )

        # 3. Launch interactive visualization window using "base" view as default viewport
        if self.config.show_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            if "base" in self.camera_name_to_id:
                self.viewer.cam.fixedcamid = self.camera_name_to_id["base"]

        # 4. Cache initial structural positions for physics calibration resets
        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()
        
        self._is_connected = True
        
        if calibrate:
            self.calibrate()

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def calibrate(self) -> None:
        if not self._is_connected:
            raise RuntimeError("Cannot calibrate environment before connecting to simulation context.")
            
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        mujoco.mj_forward(self.model, self.data)
        
        if self.config.show_viewer and self.viewer is not None:
            self.viewer.sync()
            
        self._is_calibrated = True

    def configure(self) -> None:
        pass

    # --------------------------------------------------------------------------
    # DATA STEERING (Input / Output Pipelines)
    # --------------------------------------------------------------------------

    def get_observation(self) -> dict[str, float | np.ndarray]:
        """Polls current joint telemetry and returns native HWC camera images."""
        if not self._is_connected:
            raise RuntimeError("Robot instance must be connected to poll system telemetry.")
        
        obs_dict = {}
        available_qpos = len(self.data.qpos)
        
        # 1. Map flat MuJoCo joint arrays to explicit named float keys
        for i, joint_name in enumerate(self._motors_ft.keys()):
            obs_dict[joint_name] = float(self.data.qpos[i]) if i < available_qpos else 0.0
        
        # 2. Sequentially update and extract matrices from all active renderer instances
        for cam_key, cam_id in self.camera_name_to_id.items():
            renderer = self.renderers[cam_key]
            renderer.update_scene(self.data, camera=cam_id)
            raw_image = renderer.render()  # Returns native HWC (H, W, 3) matrix
            
            # Populate the observation dictionary directly with raw data
            obs_dict[cam_key] = raw_image
            
            # 3. Handle live rendering preview window via OpenCV conditional flag
            if self.config.render:
                opencv_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
                cv2.imshow(f"LeRobot Viewport - {cam_key}", opencv_image)
        
        if self.config.render:
            cv2.waitKey(1)
        
        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Reassembles flat motor array from named dictionary updates and ticks physics."""
        if not self._is_connected:
            raise RuntimeError("Robot instance must be connected to dispatch control targets.")

        # 1. Rebuild flat control topology array from active feature dictionaries
        action_vector = np.zeros(20, dtype=np.float32)
        for i, joint_name in enumerate(self._motors_ft.keys()):
            action_vector[i] = action.get(joint_name, 0.0)

        # 2. Assign control outputs directly into MuJoCo actuator cells
        available_actuators = min(20, self.model.nu)
        if available_actuators > 0:
            self.data.ctrl[:available_actuators] = action_vector[:available_actuators]
            
        # 3. Advance physics world metrics by one simulation step unit
        mujoco.mj_step(self.model, self.data)
        
        if self.config.show_viewer and self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()
            
        return {joint_name: action.get(joint_name, 0.0) for joint_name in self._motors_ft.keys()}

    def disconnect(self) -> None:
        self._is_connected = False
        self._is_calibrated = False
        
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            
        # Clean up all allocated renderer memory instances safely
        for renderer in self.renderers.values():
            renderer.close()
        self.renderers.clear()
        self.camera_name_to_id.clear()
            
        cv2.destroyAllWindows()