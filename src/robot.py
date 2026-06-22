import os
import dataclasses
import numpy as np
import mujoco
import mujoco.viewer

from lerobot.robots import Robot, RobotConfig
from lerobot.cameras.camera import CameraConfig
from functools import cached_property

# ==============================================================================
# 1. ROBOT CONFIGURATION
# ==============================================================================

@RobotConfig.register_subclass("mujoco_orca")
@dataclasses.dataclass
class OrcaRobotConfig(RobotConfig):
    """Configuration class for the Orca Hand in MuJoCo."""
    show_viewer: bool = False
    render_cv2: bool = True
    xml_path: str = "scene/scene.xml"
    # Camera specs as plain dicts — no CameraConfig, no draccus encoding issue
    camera_configs: dict[str, dict] = dataclasses.field(default_factory=lambda: {
        "base":  {"height": 480, "width": 640, "fps": 30},
        "wrist": {"height": 480, "width": 640, "fps": 30},
    })

# ==============================================================================
# 2. VIRTUAL HARDWARE BUS
# ==============================================================================

class MuJoCoMockBus:
    """
    A virtual hardware bus that tricks the HIL-SERL framework into treating 
    the MuJoCo simulation as a physical, hardware-connected robot.
    """
    def __init__(self, robot_instance):
        self._robot = robot_instance
        # Extract base motor names by removing the '.pos' suffix expected by LeRobot
        self.motors = {
            key.replace('.pos', ''): None 
            for key in robot_instance._motors_ft.keys()
        }
        self.is_connected = True
        self.is_calibrated = True

    def sync_read(self, command: str) -> dict:
        """Intercepts hardware read requests and fetches normalized simulation state."""
        if command == "Present_Position":
            obs = self._robot.get_observation()
            return {k: float(obs.get(f"{k}.pos", 0.0)) for k in self.motors.keys()}
        return {}

    def sync_write(self, command: str, target_dict: dict):
        """Intercepts hardware write requests and routes them as delta actions."""
        if command == "Goal_Position":
            action = {f"{k}.pos": float(v) for k, v in target_dict.items()}
            self._robot.send_action(action)
            
    def connect(self): 
        pass
        
    def disconnect(self, *args): 
        pass

# ==============================================================================
# 3. CORE ROBOT CLASS
# ==============================================================================
class OrcaRobot(Robot):
    """
    Hardware driver for the Orca Hand. Interfaces with the MuJoCo physics engine,
    normalizes observations for RL, and handles velocity-based (delta) control
    by correctly targeting actuators instead of directly teleporting joints.
    """
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
        
        # Mappings built during connection
        self.actuator_name_to_id = {}
        self.actuator_id_to_joint_id = {}

        self.bus = MuJoCoMockBus(self)

    # --------------------------------------------------------------------------
    # FEATURE SCHEMAS
    # --------------------------------------------------------------------------

    @property
    def cameras(self):
        # Return a simple namespace so existing code using .height/.width/.fps still works
        return {
            name: type("Cam", (), cfg)()
            for name, cfg in self.config.camera_configs.items()
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            name: (cfg["height"], cfg["width"], 3)
            for name, cfg in self.config.camera_configs.items()
        }
    
    @property
    def _motors_ft(self) -> dict[str, type]:
        """
        Defines the names of the ACTUATORS exposed to the RL system.
        These must exactly match the 'name' attributes in <actuator> tags.
        """
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
        # Based on your XML, the actuators are named "x", "y", "z"
        aux_names = ["x.pos", "y.pos", "z.pos"] 
        
        all_actuator_names = tuple(hand_names + aux_names)
        return dict.fromkeys(all_actuator_names, float)

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) 
            for cam in self.config.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    # --------------------------------------------------------------------------
    # LIFECYCLE MANAGEMENT
    # --------------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._is_connected and (len(self.renderers) > 0)

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            return

        if not os.path.exists(self.config.xml_path):
            raise FileNotFoundError(f"MuJoCo scene file not found at: {self.config.xml_path}")

        self.model = mujoco.MjModel.from_xml_path(self.config.xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Build Camera Mappings
        for cam_name, cam_cfg in self.config.camera_configs.items():
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id == -1:
                raise ValueError(f"Camera '{cam_name}' not found in XML.")
            self.camera_name_to_id[cam_name] = cam_id
            self.renderers[cam_name] = mujoco.Renderer(
                self.model, height=cam_cfg["height"], width=cam_cfg["width"]
            )

        # Build Actuator to Joint Mappings
        for motor_feature in self._motors_ft.keys():
            actuator_name = motor_feature.replace('.pos', '_actuator')  # Match the <actuator> names in XML
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            
            if act_id != -1:
                self.actuator_name_to_id[actuator_name] = act_id
                # MuJoCo stores the target joint ID for each actuator in trnid (transmission ID)
                # For position actuators, trnid[0] is the joint ID.
                self.actuator_id_to_joint_id[act_id] = self.model.actuator_trnid[act_id][0]
            else:
                print(f"Warning: Actuator '{actuator_name}' not found in MuJoCo model!")

        if self.config.show_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            if "base" in self.camera_name_to_id:
                self.viewer.cam.fixedcamid = self.camera_name_to_id["base"]

        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()
        self.initial_ctrl = self.data.ctrl.copy()
        
        self._is_connected = True
        
        if calibrate:
            self.calibrate()

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def calibrate(self) -> None:
        if not self._is_connected:
            raise RuntimeError("Cannot calibrate before connecting.")
            
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        self.data.ctrl[:] = self.initial_ctrl
        mujoco.mj_forward(self.model, self.data)
        
        if self.config.show_viewer and self.viewer is not None:
            self.viewer.sync()
            
        self._is_calibrated = True

    def configure(self) -> None:
        pass

    # --------------------------------------------------------------------------
    # DATA STEERING & PHYSICS
    # --------------------------------------------------------------------------

    def get_observation(self) -> dict[str, float | np.ndarray]:
        if not self._is_connected:
            raise RuntimeError("Robot must be connected to poll telemetry.")
        
        obs_dict = {}
        
        for cam_key, cam_id in self.camera_name_to_id.items():
            renderer = self.renderers[cam_key]
            renderer.update_scene(self.data, camera=cam_id)
            img_rgb = np.array(renderer.render().copy(), dtype=np.uint8)
            obs_dict[cam_key] = img_rgb
            
            # Optional GUI preview
            if getattr(self.config, "render_cv2", False):
                import cv2
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow(f"Orca Vision - {cam_key}", img_bgr)
                cv2.waitKey(1)
            
        # Fetch and normalize actuator target states
        for motor_feature in self._motors_ft.keys():
            act_name = motor_feature.replace('.pos', '_actuator')
            act_id = self.actuator_name_to_id.get(act_name, -1)
            
            if act_id != -1:
                joint_id = self.actuator_id_to_joint_id[act_id]
                qpos_idx = self.model.jnt_qposadr[joint_id]
                
                # We read the ACTUAL physical position of the joint
                current_physical_pos = self.data.qpos[qpos_idx]
                
                # But we normalize it relative to what the ACTUATOR can control (ctrlrange)
                ctrl_min = self.model.actuator_ctrlrange[act_id][0]
                ctrl_max = self.model.actuator_ctrlrange[act_id][1]
                range_span = ctrl_max - ctrl_min
                
                if range_span > 0:
                    normalized_pos = 2.0 * ((current_physical_pos - ctrl_min) / range_span) - 1.0
                else:
                    normalized_pos = 0.0
                    
                obs_dict[motor_feature] = float(np.clip(normalized_pos, -1.0, 1.0))
            else:
                obs_dict[motor_feature] = 0.0
        
        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self._is_connected:
            raise RuntimeError("Robot must be connected to dispatch control targets.")

        BASE_SPEED = 0.05   
        FINGER_SPEED = 0.05 
        BASE_ACTUATORS = ["x_actuator", "y_actuator", "z_actuator"] 

        for feature_name, action_val in action.items():
            act_name = feature_name.replace('.pos', '_actuator')
            act_id = self.actuator_name_to_id.get(act_name, -1)
            
            if act_id != -1:
                speed_scale = BASE_SPEED if act_name in BASE_ACTUATORS else FINGER_SPEED
                delta = float(action_val) * speed_scale

                # The new target is derived from the LAST sent command (data.ctrl), 
                # ensuring smooth integration rather than jittering around the physical state.
                joint_id = self.actuator_id_to_joint_id[act_id]
                qpos_idx = self.model.jnt_qposadr[joint_id]
                current_physical_pos = self.data.qpos[qpos_idx]
                new_target_pos = current_physical_pos + delta

                # Clip using the actuator's specific control range
                ctrl_min = self.model.actuator_ctrlrange[act_id][0]
                ctrl_max = self.model.actuator_ctrlrange[act_id][1]
                new_target_pos = np.clip(new_target_pos, ctrl_min, ctrl_max)

                self.data.ctrl[act_id] = new_target_pos

        mujoco.mj_step(self.model, self.data)
        
        if self.config.show_viewer and self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()
            
        return action

    def disconnect(self) -> None:
        self._is_connected = False
        self._is_calibrated = False
        
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            
        for renderer in self.renderers.values():
            renderer.close()
            
        self.renderers.clear()
        self.camera_name_to_id.clear()
        self.actuator_name_to_id.clear()
        self.actuator_id_to_joint_id.clear()