import time
import numpy as np

# Import our custom environment components
from robot import OrcaRobot, OrcaRobotConfig
from teleop import KeyboardTeleoperator, KeyboardTeleoperatorConfig

def test_manual_control():
    print("=" * 55)
    print("🚀 ORCA HAND TELEOPERATION SANDBOX INITIALIZATION")
    print("=" * 55)

    # 1. Initialize Robot Configuration
    # Enable custom OpenCV windows and turn off standard MuJoCo 3D window
    robot_config = OrcaRobotConfig(
        xml_path="scene/scene.xml",
        show_viewer=False, 
        render_cv2=True
    )
    robot = OrcaRobot(robot_config)

    # 2. Initialize Teleoperator Configuration
    teleop_config = KeyboardTeleoperatorConfig()
    teleop = KeyboardTeleoperator(teleop_config)

    try:
        print("[SANDBOX] Connecting to MuJoCo physics driver...")
        robot.connect(calibrate=True)
        
        print("[SANDBOX] Registering global keyboard hook...")
        teleop.connect()

        print("\n[SANDBOX] System ready! Click your Terminal window to focus input.")
        print("[SANDBOX] Press CTRL+C in the terminal to terminate safely.\n")

        # Maintain a clean 30 FPS control loop matching LeRobot's frequency
        dt = 1.0 / 30.0
        
        while True:
            start_time = time.perf_counter()

            # A. Fetch the current 20-DoF action vector array from the teleop listener
            action_vector = teleop.get_action()
            print(f"[DEBUG] Active Action Vector (indices 17-19): {action_vector[17:20]}", end="\r")
            
            # B. Package the flat numpy array into the expected feature dictionary
            action_dict = {}
            for i, motor_name in enumerate(robot._motors_ft.keys()):
                action_dict[motor_name] = action_vector[i]

            # C. Dispatch velocity updates to the simulation (ticks step and handles OpenCV windows)
            robot.send_action(action_dict)
            robot.get_observation()  # Update the robot's internal state and render OpenCV windows
            
            # D. Maintain target frequency pacing
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0.0, dt - elapsed)
            # time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[SANDBOX] Execution halted via KeyboardInterrupt (CTRL+C).")
        
    finally:
        print("[SANDBOX] Revoking hooks and closing hardware contexts...")
        teleop.disconnect()
        robot.disconnect()
        print("[SANDBOX] Cleanup sequence finished.")

if __name__ == "__main__":
    test_manual_control()