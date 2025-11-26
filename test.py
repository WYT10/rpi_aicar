#!/usr/bin/env python3
"""
Very simple validation script for AI car:

- Motor tests: left / right / both
- Manual motor values
- Camera FPS for a short period

No extra helper functions, everything is inline.
"""

import time
import sys

# Use your existing project classes / config
from app import MotorController, CameraManager, CONFIG  # type: ignore

print("=== AI CAR VALIDATION SCRIPT ===")

# ----- Create motor controller -----
print("\n[STEP] Initializing motors from CONFIG...")
motor_cfg = CONFIG.get("motor", {})
motors = MotorController(cfg=motor_cfg)
print(f"[INFO] Motor backend: {motors.backend_name}")
print(f"[INFO] Motor pins from config: {motor_cfg.get('pins')}")

# Ask test power
try:
    power_str = input("\nEnter test power for motors (0.0–1.0, default 0.4): ").strip()
    if power_str == "":
        power = 0.4
    else:
        power = float(power_str)
except Exception:
    print("[WARN] Invalid input, using 0.4")
    power = 0.4

# Clamp
if power < 0.0:
    power = 0.0
if power > 1.0:
    power = 1.0

duration = 1.5

print("\n========================================")
print("[TEST 1] LEFT MOTOR ONLY")
print("Expected: ONLY left wheel should move forward.")
input("Press ENTER to start left motor test...")

motors.open_loop(power, 0.0)
print(f"[CMD] open_loop(left={power:.2f}, right=0.00)")
time.sleep(duration)
motors.open_loop(0.0, 0.0)
print("[INFO] Left motor test done.")
input("If behavior was correct, press ENTER to continue...")

print("\n========================================")
print("[TEST 2] RIGHT MOTOR ONLY")
print("Expected: ONLY right wheel should move forward.")
input("Press ENTER to start right motor test...")

motors.open_loop(0.0, power)
print(f"[CMD] open_loop(left=0.00, right={power:.2f})")
time.sleep(duration)
motors.open_loop(0.0, 0.0)
print("[INFO] Right motor test done.")
input("If behavior was correct, press ENTER to continue...")

print("\n========================================")
print("[TEST 3] BOTH MOTORS")
print("Expected: both wheels move forward, car goes straight (if balanced).")
input("Press ENTER to start both-motor test...")

motors.open_loop(power, power)
print(f"[CMD] open_loop(left={power:.2f}, right={power:.2f})")
time.sleep(duration)
motors.open_loop(0.0, 0.0)
print("[INFO] Both motors test done.")
input("Press ENTER to continue to manual control...")

# ----- Manual motor control -----
print("\n========================================")
print("[TEST 4] MANUAL MOTOR CONTROL")
print("Enter left and right values in [0.0, 1.0].")
print("Example: 0.3 0.3   (both forward)")
print("Empty line to stop manual test.\n")

while True:
    line = input("left right > ").strip()
    if line == "":
        break
    parts = line.split()
    if len(parts) != 2:
        print("Please enter two numbers: left right")
        continue
    try:
        left = float(parts[0])
        right = float(parts[1])
    except Exception:
        print("Invalid numbers.")
        continue

    # clamp to [0, 1]
    if left < 0.0:
        left = 0.0
    if left > 1.0:
        left = 1.0
    if right < 0.0:
        right = 0.0
    if right > 1.0:
        right = 1.0

    print(f"[CMD] open_loop(left={left:.2f}, right={right:.2f})")
    motors.open_loop(left, right)

# stop motors after manual mode
motors.open_loop(0.0, 0.0)
print("[INFO] Manual motor test finished.")

# ----- Camera tests -----
print("\n========================================")
print("[STEP] Initializing camera from CONFIG...")

cam_cfg = CONFIG.get("camera", {})
camera = CameraManager(
    width=cam_cfg.get("width", 320),
    height=cam_cfg.get("height", 240),
    fps=cam_cfg.get("fps", 15),
)

print(f"[INFO] Camera config: {cam_cfg}")
input("Press ENTER to start camera FPS test (~8 seconds)...")

try:
    print("[CAM] Starting camera...")
    camera.start()
except Exception as e:
    print(f"[ERROR] Failed to start camera: {e}")
    print("[EXIT] Stopping motors and exiting.")
    try:
        motors.open_loop(0.0, 0.0)
        motors.shutdown()
    except Exception:
        pass
    sys.exit(1)

start_time = time.time()
last_print = start_time
print(f"[CAM] Backend: {camera.backend_name}")
print("[CAM] Reading FPS, watch the values below:\n")

while True:
    now = time.time()
    if now - start_time > 8.0:
        break

    # grab frame just to keep pipeline active
    frame = camera.get_frame()
    if (now - last_print) >= 1.0:
        fps = camera.get_fps()
        if frame is None:
            print(f"  [{now - start_time:4.1f}s] FPS={fps:.2f} (no frame)")
        else:
            h = frame.shape[0]
            w = frame.shape[1]
            print(f"  [{now - start_time:4.1f}s] FPS={fps:.2f}, frame={w}x{h}")
        last_print = now

    time.sleep(0.01)

print("\n[CAM] Stopping camera...")
camera.stop()

print("\n========================================")
print("[DONE] Validation script finished.")
print("Motors stopped, camera stopped.")
try:
    motors.open_loop(0.0, 0.0)
    motors.shutdown()
except Exception:
    pass
print("Good luck with further debugging and AI driving!")
