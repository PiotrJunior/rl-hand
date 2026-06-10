import sys
import time
import threading
import cv2

sys.path.insert(0, "orca_teleop/src")

import mujoco
import mediapipe as mp
import numpy as np
from orca_teleop.ingress.mediapipe.mediapipe_ingress import MediaPipeIngress
from orca_teleop.retargeting.retargeter import Retargeter, TargetPose

from robot import OrcaRobot, OrcaRobotConfig

MODEL_PATH  = ".venv/lib/python3.13/site-packages/orca_core/models/v1/orcahand_right/config.yaml"
URDF_PATH   = "orcahand_description/v1/models/urdf/orcahand_right.urdf"
CONFIG_PATH = "orca_teleop/src/orca_teleop/retargeting/configs/adaptive_analytical_orca.yaml"
HAND_TYPE   = "right"
FPS         = 30

# Maps retargeter joint names → data.ctrl indices
# Indices 0-2 are the XYZ slide actuators; hand joints start at 3
JOINT_TO_CTRL = {
    "thumb_pip":  0,  # right_t-cmc (aliased)
    "thumb_mcp":  1,  # right_t-mcp
    "thumb_abd":  2,  # right_t-abd
    "thumb_dip":  3,  # right_t-pip
    "index_abd":  4,  # right_i-abd
    "index_mcp":  5,  # right_i-mcp
    "index_pip":  6,  # right_i-pip
    "middle_abd": 7,  # right_m-abd
    "middle_mcp": 8,  # right_m-mcp
    "middle_pip": 9,  # right_m-pip
    "ring_abd":   10,  # right_r-abd
    "ring_mcp":   11,  # right_r-mcp
    "ring_pip":   12,  # right_r-pip
    "pinky_abd":  13,  # right_p-abd
    "pinky_mcp":  14,  # right_p-mcp
    "pinky_pip":  15,  # right_p-pip
    "wrist":      16,  # right_wrist
}


class SimMediaPipeIngress(MediaPipeIngress):
    """MediaPipeIngress with OrcaHand removed — for sim/visualization use."""

    def __init__(self, hand_type: str = "right", callback=None):
        import os
        self.callback = callback
        self.hand_type = hand_type

        task_path = os.path.join(
            os.path.dirname(os.path.abspath(
                sys.modules["orca_teleop.ingress.mediapipe.mediapipe_ingress"].__file__
            )),
            "hand_landmarker.task",
        )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(task_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            result_callback=self._result_callback,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open webcam")

        self.latest_frame = None
        self.latest_image_landmarks = None
        self.orientation_good = True
        self.frame_lock = threading.Lock()
        self.running = False
        self.livestream_thread = None


def main():
    # --- Init retargeter ---
    print("Loading retargeter (this may take a moment)...")
    retargeter = Retargeter.from_paths(
        model_path=MODEL_PATH,
        urdf_path=URDF_PATH,
        backend="adaptive_analytical",
        config_path=CONFIG_PATH,
    )
    print("Retargeter ready.")

    # --- Init sim ---
    print("Initializing simulation...")
    robot_config = OrcaRobotConfig(show_viewer=True)
    robot = OrcaRobot(robot_config)

    # --- Init ingress ---
    latest_joints = {"data": None}
    joints_lock = threading.Lock()

    def on_landmarks(world_landmarks: np.ndarray):
        with joints_lock:
            latest_joints["data"] = world_landmarks

    ingress = SimMediaPipeIngress(hand_type=HAND_TYPE, callback=on_landmarks)

    calibrated = False
    print("Calibrating — hold your hand in view, palm facing down...")
    print("Press 'q' to quit.")

    with robot:
        robot.calibrate()
        ingress.start()

        try:
            while True:
                start_time = time.perf_counter()

                # Show webcam with hand landmarks
                ingress.display_frame()

                # Get latest landmarks from ingress callback
                with joints_lock:
                    landmarks = latest_joints["data"]

                if landmarks is not None:
                    pose = TargetPose(joint_positions=landmarks)
                    result = retargeter.retarget(pose)

                    if result is not None:
                        if not calibrated:
                            print("Calibration done! Retargeting active.")
                            calibrated = True

                        # Build action vector: indices 0-2 stay 0 (XYZ slides),
                        # indices 3-19 are hand joint angles in radians
                        action = np.zeros(37, dtype=np.float32)
                        for joint_name, ctrl_idx in JOINT_TO_CTRL.items():
                            if joint_name in result.data:
                                action[ctrl_idx] = result.data[joint_name]
                                # action[ctrl_idx] = np.deg2rad(result.data[joint_name])
                        robot.send_action({"action": action})

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                # Pace the loop to FPS
                elapsed = time.perf_counter() - start_time
                wait = (1.0 / FPS) - elapsed
                if wait > 0:
                    time.sleep(wait)

        except KeyboardInterrupt:
            pass
        finally:
            ingress.cleanup()


if __name__ == "__main__":
    main()