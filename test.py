#!/usr/bin/env python3
"""
Simple terminal UI to test AI car hardware:

- Motors: left / right / both
- Motion patterns: forward, spin in place
- Camera: backend + live FPS readout

Run on the Pi inside your project folder:

    python3 diagnostics_cli.py
"""

import time
import sys

# Import your existing classes from app.py
from app import CameraManager, MotorController, CONFIG  # type: ignore


def make_camera():
    cam_cfg = CONFIG.get("camera", {})
    cam = CameraManager(
        width=cam_cfg.get("width", 320),
        height=cam_cfg.get("height", 240),
        fps=cam_cfg.get("fps", 15),
    )
    return cam


def make_motors():
    motor_cfg = CONFIG.get("motor", {})
    m = MotorController(cfg=motor_cfg)
    print(f"[INFO] Motor backend: {m.backend_name}")
    return m


def pause(msg="Press ENTER to continue..."):
    input(msg)


# ---------------- Motor tests ----------------


def test_motor_single_side(motors):
    """
    Test left and right channels separately.
    """
    try:
        power = float(input("Enter test power (0.0–1.0, e.g. 0.4): ").strip() or "0.4")
    except ValueError:
        print("Invalid, using 0.4")
        power = 0.4

    duration = 1.5
    print("\n[TEST] LEFT motor only...")
    motors.open_loop(power, 0.0)
    time.sleep(duration)
    motors.open_loop(0.0, 0.0)
    print("[INFO] LEFT test done.\n")
    pause()

    print("\n[TEST] RIGHT motor only...")
    motors.open_loop(0.0, power)
    time.sleep(duration)
    motors.open_loop(0.0, 0.0)
    print("[INFO] RIGHT test done.\n")
    pause()

    print("\n[TEST] BOTH motors...")
    motors.open_loop(power, power)
    time.sleep(duration)
    motors.open_loop(0.0, 0.0)
    print("[INFO] BOTH test done.\n")
    pause()


def test_motion_patterns(motors):
    """
    Simple patterns to see if the car moves correctly:
      - forward
      - spin left
      - spin right
    """
    try:
        power = float(input("Enter base power (0.0–1.0, e.g. 0.4): ").strip() or "0.4")
    except ValueError:
        print("Invalid, using 0.4")
        power = 0.4

    dur = 1.5

    print("\n[PATTERN] Forward:")
    motors.open_loop(power, power)
    time.sleep(dur)
    motors.open_loop(0.0, 0.0)
    pause("Forward done. Press ENTER for spin-left...")

    print("\n[PATTERN] Spin LEFT (right wheel forward, left off):")
    motors.open_loop(0.0, power)
    time.sleep(dur)
    motors.open_loop(0.0, 0.0)
    pause("Spin-left done. Press ENTER for spin-right...")

    print("\n[PATTERN] Spin RIGHT (left wheel forward, right off):")
    motors.open_loop(power, 0.0)
    time.sleep(dur)
    motors.open_loop(0.0, 0.0)
    pause("Spin-right done. Press ENTER to continue...")


def manual_motor_control(motors):
    """
    Manual numeric control loop.
    Useful to debug asymmetry or offsets.
    """
    print("\n[MANUAL] Enter left/right values in [0.0, 1.0]. Empty to stop.")
    print("Example: 0.3 0.3  (both forward)\n")

    while True:
        line = input("left right > ").strip()
        if not line:
            break
        try:
            parts = line.split()
            if len(parts) != 2:
                print("Please enter two numbers: left right")
                continue
            left = float(parts[0])
            right = float(parts[1])
        except ValueError:
            print("Invalid numbers.")
            continue

        # clamp
        left = max(0.0, min(1.0, left))
        right = max(0.0, min(1.0, right))

        print(f"[CMD] open_loop(left={left:.2f}, right={right:.2f})")
        motors.open_loop(left, right)

    motors.open_loop(0.0, 0.0)
    print("[MANUAL] Stopped motors.")


# ---------------- Camera tests ----------------


def test_camera_fps(camera):
    """
    Start camera and print backend + FPS for a few seconds.
    """
    print("\n[CAM] Starting camera...")
    try:
        camera.start()
    except Exception as e:
        print(f"[ERROR] Failed to start camera: {e}")
        pause()
        return

    print(f"[CAM] Backend: {camera.backend_name}")
    print("[CAM] Measuring FPS for ~10 seconds...\n")
    t0 = time.time()
    last_print = t0

    try:
        while time.time() - t0 < 10.0:
            # just grab frame to keep pipeline active
            _ = camera.get_frame()
            now = time.time()
            if now - last_print >= 1.0:
                fps = camera.get_fps()
                print(f"  [{now - t0:4.1f}s] camera_fps = {fps:.2f}")
                last_print = now
            time.sleep(0.01)
    finally:
        print("\n[CAM] Stopping camera...")
        camera.stop()
        pause()


def camera_single_snapshot(camera):
    """
    Grab a single frame and print its shape and current FPS.
    """
    print("\n[CAM] Starting camera for one snapshot...")
    try:
        camera.start()
    except Exception as e:
        print(f"[ERROR] Failed to start camera: {e}")
        pause()
        return

    time.sleep(0.5)  # let it warm up a bit
    frame = camera.get_frame()
    fps = camera.get_fps()
    if frame is None:
        print("[CAM] No frame received.")
    else:
        h, w = frame.shape[:2]
        print(f"[CAM] Got frame: {w}x{h}, fps={fps:.2f}")

    camera.stop()
    pause()


# ---------------- Main menu ----------------


def main():
    print("=== AI Car Diagnostics (Terminal UI) ===")
    print("Using CONFIG from app.py")

    motors = make_motors()
    camera = make_camera()

    try:
        while True:
            print("\n--- Main Menu ---")
            print("1) Motor test: left / right / both")
            print("2) Motion patterns (forward + spins)")
            print("3) Manual motor control")
            print("4) Camera FPS test (10s)")
            print("5) Camera single snapshot (shape + FPS)")
            print("0) Exit")
            choice = input("Select option: ").strip()

            if choice == "1":
                test_motor_single_side(motors)
            elif choice == "2":
                test_motion_patterns(motors)
            elif choice == "3":
                manual_motor_control(motors)
            elif choice == "4":
                test_camera_fps(camera)
            elif choice == "5":
                camera_single_snapshot(camera)
            elif choice == "0":
                break
            else:
                print("Unknown option.")
    finally:
        print("\n[EXIT] Stopping motors and camera...")
        try:
            motors.open_loop(0.0, 0.0)
        except Exception:
            pass
        try:
            motors.shutdown()
        except Exception:
            pass
        try:
            camera.stop()
        except Exception:
            pass
        print("[EXIT] Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[CTRL-C] Exiting.")
        sys.exit(0)
