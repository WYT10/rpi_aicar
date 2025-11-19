#!/usr/bin/env python3
import os
import json
import time
import glob
import logging
import threading
import atexit
from collections import deque

import cv2
import numpy as np
from flask import Flask, jsonify, request, Response, render_template
from flask_cors import CORS

# ============================================================
# Config + Logging
# ============================================================

ROOT = os.path.abspath(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(ROOT, "config.json")


def load_config(path=CONFIG_PATH):
    if not os.path.exists(path):
        # minimal default config
        cfg = {
            "server": {"host": "0.0.0.0", "port": 8888},
            "camera": {"width": 640, "height": 480, "fps": 20},
            "motor": {
                "pins": {
                    "left_forward": 5,
                    "left_backward": 6,
                    "right_forward": 20,
                    "right_backward": 21,
                    "left_pwm": None,
                    "right_pwm": None,
                    "stby": None,
                },
                "invert_left": True,
                "invert_right": False,
                "pwm_freq": 800,
                "max_duty_pct": 90,
                "deadzone": 0.03,
            },
            "paths": {
                "logs_root": "data/logs",
                "datasets_root": "data/datasets",
                "models_root": "data/models",
            },
            "joystick": {
                "max_speed": 0.5,    # cap |v|
                "max_rot": 0.8,      # cap |w|
                "speed_gain": 1.0,   # shaping gain on v (server side)
                "steer_gain": 1.0,   # shaping gain on w (server side)
            },
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        return cfg
    with open(path, "r") as f:
        return json.load(f)


CONFIG = load_config()

DATA_ROOT = os.path.join(ROOT, "data")
LOG_ROOT = os.path.join(DATA_ROOT, "logs")
DATASETS_ROOT = os.path.join(DATA_ROOT, "datasets")
MODELS_ROOT = os.path.join(DATA_ROOT, "models")
TMP_ROOT = os.path.join(DATA_ROOT, "tmp")

for d in (DATA_ROOT, LOG_ROOT, DATASETS_ROOT, MODELS_ROOT, TMP_ROOT):
    os.makedirs(d, exist_ok=True)

log_path = os.path.join(LOG_ROOT, "app.log")
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.getLogger("").addHandler(console)

logger = logging.getLogger("aicar")
logger.info("AI Car server starting up...")

# ============================================================
# Camera (hybrid backend, threaded + simple processing)
# ============================================================


class CameraManager:
    """
    Hybrid camera manager:
      1. Try Picamera2 (BGR888)
      2. Try /dev/video* via OpenCV V4L2
      3. Try GStreamer libcamerasrc → BGR

    Provides:
      - start()
      - stop()
      - get_frame()  -> latest processed BGR frame or None
      - get_raw()    -> latest raw BGR frame or None
      - get_fps()
      - backend_name
      - update_settings(...)
    """

    def __init__(self, width=640, height=480, fps=20):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        self._backend = None
        self._picam2 = None
        self._cap = None

        self._frame_raw = None
        self._frame_proc = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        self._fps_hist = deque(maxlen=60)

        # processing settings
        self.vflip = False
        self.hflip = False
        self.gray = False
        self.mode = "raw"  # raw / gray / edges
        self.gamma = 1.0

    # -------- Backend detection helpers --------
    @staticmethod
    def _opencv_supports_gst():
        try:
            info = cv2.getBuildInformation()
            return "GStreamer" in info and "YES" in info
        except Exception:
            return False

    @staticmethod
    def _list_video_devices():
        return sorted(glob.glob("/dev/video*"))

    # -------- Open/close backends --------
    def _open_backend(self):
        # Try Picamera2 first
        try:
            from picamera2 import Picamera2  # type: ignore

            cam = Picamera2()
            # Use BGR888 so frames are already BGR for OpenCV
            cfg = cam.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "BGR888"},
                buffer_count=3,
            )
            cam.configure(cfg)
            cam.start()
            self._picam2 = cam
            self._backend = "picamera2"
            logger.info("Camera: using Picamera2 backend (BGR888)")
            return True
        except Exception as e:
            logger.info(f"Camera: Picamera2 not available: {e!r}")
            self._picam2 = None

        # Try V4L2 devices via OpenCV
        for dev in self._list_video_devices():
            try:
                cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                ok, frame = cap.read()
                if ok and frame is not None:
                    self._cap = cap
                    self._backend = f"v4l2:{dev}"
                    logger.info(f"Camera: using V4L2 backend on {dev}")
                    return True
                cap.release()
            except Exception as e:
                logger.info(f"Camera: V4L2 check failed on {dev}: {e!r}")

        # Try GStreamer pipeline if OpenCV has GStreamer
        if self._opencv_supports_gst():
            try:
                pipeline = (
                    f"libcamerasrc ! "
                    f"video/x-raw,width={self.width},height={self.height},"
                    f"framerate={self.fps}/1,format=RGB ! "
                    f"videoconvert ! video/x-raw,format=BGR ! "
                    f"appsink drop=1 max-buffers=1 sync=false"
                )
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                ok, frame = cap.read()
                if ok and frame is not None:
                    self._cap = cap
                    self._backend = "gstreamer"
                    logger.info("Camera: using GStreamer backend")
                    return True
                cap.release()
            except Exception as e:
                logger.info(f"Camera: GStreamer backend failed: {e!r}")

        logger.error("Camera: failed to open any backend")
        return False

    def _close_backend(self):
        if self._picam2 is not None:
            try:
                self._picam2.stop()
                self._picam2.close()
            except Exception:
                pass
            self._picam2 = None

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        self._backend = None
        logger.info("Camera: backend closed")

    # -------- Processing --------
    def _apply_processing(self, frame_bgr):
        img = frame_bgr

        if self.vflip:
            img = cv2.flip(img, 0)
        if self.hflip:
            img = cv2.flip(img, 1)

        if abs(self.gamma - 1.0) > 1e-3:
            inv = 1.0 / max(self.gamma, 1e-6)
            table = (np.arange(256) / 255.0) ** inv * 255.0
            lut = np.clip(table, 0, 255).astype(np.uint8)
            img = cv2.LUT(img, lut)

        mode = self.mode
        if mode == "gray" or (self.gray and mode == "raw"):
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        elif mode == "edges":
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(g, 80, 160)
            img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return img

    def update_settings(self, vflip=None, hflip=None, gray=None, mode=None, gamma=None):
        if vflip is not None:
            self.vflip = bool(vflip)
        if hflip is not None:
            self.hflip = bool(hflip)
        if gray is not None:
            self.gray = bool(gray)
        if mode is not None:
            self.mode = str(mode)
        if gamma is not None:
            try:
                self.gamma = float(gamma)
            except Exception:
                pass

    # -------- Grabbing frames --------
    def _grab_frame_bgr(self):
        if self._backend == "picamera2" and self._picam2 is not None:
            # Picamera2 already gives us BGR888 now
            arr = self._picam2.capture_array("main")
            if arr is None:
                return None
            return arr

        if self._cap is not None:
            ok, frame = self._cap.read()
            return frame if ok else None

        return None

    # -------- Public API --------
    def start(self):
        if self._running:
            return
        if not self._open_backend():
            raise RuntimeError("Camera: unable to open any backend")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Camera: capture loop started")

    def _loop(self):
        prev = time.time()
        while self._running:
            frame = self._grab_frame_bgr()
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            dt = now - prev
            prev = now
            if dt > 0:
                self._fps_hist.append(1.0 / dt)

            proc = self._apply_processing(frame)

            with self._lock:
                self._frame_raw = frame
                self._frame_proc = proc

        logger.info("Camera: capture loop exiting")

    def stop(self):
        self._running = False
        time.sleep(0.05)
        self._close_backend()

    def get_frame(self):
        with self._lock:
            if self._frame_proc is None:
                return None
            return self._frame_proc.copy()

    def get_raw(self):
        with self._lock:
            if self._frame_raw is None:
                return None
            return self._frame_raw.copy()

    def get_fps(self):
        if not self._fps_hist:
            return 0.0
        return round(sum(self._fps_hist) / len(self._fps_hist), 2)

    @property
    def backend_name(self):
        return self._backend or "none"


# ============================================================
# Motors (pigpio / RPi.GPIO / Dummy)
# ============================================================

class MotorBase:
    def set_openloop(self, left: float, right: float):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def shutdown(self):
        self.stop()


class MotorDummy(MotorBase):
    def set_openloop(self, left: float, right: float):
        logger.info(f"[MOTOR dummy] L={left:+.2f} R={right:+.2f}")

    def stop(self):
        logger.info("[MOTOR dummy] stop")

    def shutdown(self):
        logger.info("[MOTOR dummy] shutdown")


class MotorPigpio(MotorBase):
    def __init__(self, cfg: dict):
        import pigpio  # type: ignore

        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpio daemon not running (sudo pigpiod)")

        pins = cfg.get("pins", {})
        self.LF = int(pins.get("left_forward", 5))
        self.LB = int(pins.get("left_backward", 6))
        self.RF = int(pins.get("right_forward", 20))
        self.RB = int(pins.get("right_backward", 21))
        self.LP = pins.get("left_pwm")
        self.RP = pins.get("right_pwm")
        self.STBY = pins.get("stby")

        self.invL = bool(cfg.get("invert_left", False))
        self.invR = bool(cfg.get("invert_right", False))
        self.freq = int(cfg.get("pwm_freq", 800))
        self.max_duty = max(0, min(100, int(cfg.get("max_duty_pct", 90))))
        self.deadzone = float(cfg.get("deadzone", 0.03))

        for p in [self.LF, self.LB, self.RF, self.RB]:
            self.pi.set_mode(p, pigpio.OUTPUT)

        if self.LP is not None:
            self.pi.set_mode(self.LP, pigpio.OUTPUT)
            self.pi.set_PWM_frequency(self.LP, self.freq)
        if self.RP is not None:
            self.pi.set_mode(self.RP, pigpio.OUTPUT)
            self.pi.set_PWM_frequency(self.RP, self.freq)

        if self.STBY is not None:
            self.pi.set_mode(self.STBY, pigpio.OUTPUT)
            self.pi.write(self.STBY, 1)

        self.stop()
        logger.info("MotorPigpio initialized")

    def _set_pwm(self, pin, duty_pct):
        if pin is None:
            return
        duty_pct = max(0, min(100, duty_pct))
        self.pi.set_PWM_dutycycle(int(pin), int(255 * duty_pct / 100.0))

    def _drive_channel(self, pwm_pin, fwd_pin, rev_pin, value, invert):
        v = -value if invert else value
        if abs(v) < self.deadzone:
            v = 0.0
        v = max(-1.0, min(1.0, v))

        self.pi.write(int(fwd_pin), 1 if v > 0 else 0)
        self.pi.write(int(rev_pin), 1 if v < 0 else 0)
        self._set_pwm(pwm_pin, abs(v) * self.max_duty)

    def set_openloop(self, left: float, right: float):
        self._drive_channel(self.LP, self.LF, self.LB, left, self.invL)
        self._drive_channel(self.RP, self.RF, self.RB, right, self.invR)

    def stop(self):
        self._set_pwm(self.LP, 0)
        self._set_pwm(self.RP, 0)
        for p in [self.LF, self.LB, self.RF, self.RB]:
            self.pi.write(int(p), 0)

    def shutdown(self):
        self.stop()
        self.pi.stop()
        logger.info("MotorPigpio shutdown")


class MotorRPiGPIO(MotorBase):
    def __init__(self, cfg: dict):
        import RPi.GPIO as GPIO  # type: ignore
        self.GPIO = GPIO

        pins = cfg.get("pins", {})
        self.LF = int(pins.get("left_forward", 5))
        self.LB = int(pins.get("left_backward", 6))
        self.RF = int(pins.get("right_forward", 20))
        self.RB = int(pins.get("right_backward", 21))
        self.LP = pins.get("left_pwm")
        self.RP = pins.get("right_pwm")
        self.STBY = pins.get("stby")

        self.invL = bool(cfg.get("invert_left", False))
        self.invR = bool(cfg.get("invert_right", False))
        self.freq = int(cfg.get("pwm_freq", 800))
        self.max_duty = max(0, min(100, int(cfg.get("max_duty_pct", 90))))
        self.deadzone = float(cfg.get("deadzone", 0.03))

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        for p in [self.LF, self.LB, self.RF, self.RB]:
            GPIO.setup(p, GPIO.OUT)

        if self.LP is not None:
            GPIO.setup(self.LP, GPIO.OUT)
            self.pwmL = GPIO.PWM(self.LP, self.freq)
            self.pwmL.start(0)
        else:
            self.pwmL = None

        if self.RP is not None:
            GPIO.setup(self.RP, GPIO.OUT)
            self.pwmR = GPIO.PWM(self.RP, self.freq)
            self.pwmR.start(0)
        else:
            self.pwmR = None

        if self.STBY is not None:
            GPIO.setup(self.STBY, GPIO.OUT)
            GPIO.output(self.STBY, GPIO.HIGH)

        self.stop()
        logger.info("MotorRPiGPIO initialized")

    def _set_pwm(self, pwm, duty_pct):
        if pwm is None:
            return
        duty_pct = max(0, min(100, duty_pct))
        pwm.ChangeDutyCycle(duty_pct)

    def _drive_channel(self, pwm, fwd_pin, rev_pin, value, invert):
        v = -value if invert else value
        if abs(v) < self.deadzone:
            v = 0.0
        v = max(-1.0, min(1.0, v))

        self.GPIO.output(fwd_pin, self.GPIO.HIGH if v > 0 else self.GPIO.LOW)
        self.GPIO.output(rev_pin, self.GPIO.HIGH if v < 0 else self.GPIO.LOW)
        self._set_pwm(pwm, abs(v) * self.max_duty)

    def set_openloop(self, left: float, right: float):
        self._drive_channel(self.pwmL, self.LF, self.LB, left, self.invL)
        self._drive_channel(self.pwmR, self.RF, self.RB, right, self.invR)

    def stop(self):
        self.GPIO.output(self.LF, self.GPIO.LOW)
        self.GPIO.output(self.LB, self.GPIO.LOW)
        self.GPIO.output(self.RF, self.GPIO.LOW)
        self.GPIO.output(self.RB, self.GPIO.LOW)
        self._set_pwm(self.pwmL, 0)
        self._set_pwm(self.pwmR, 0)

    def shutdown(self):
        self.stop()
        if self.pwmL:
            self.pwmL.stop()
        if self.pwmR:
            self.pwmR.stop()
        self.GPIO.cleanup()
        logger.info("MotorRPiGPIO shutdown")


class MotorController:
    """
    Wrapper that auto-selects backend:
      - pigpio if available
      - RPi.GPIO if available
      - Dummy otherwise
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self.backend = None
        self.backend_name = "dummy"

        try:
            import pigpio  # noqa: F401
            self.backend = MotorPigpio(self.cfg)
            self.backend_name = "pigpio"
            logger.info("MotorController using pigpio backend")
            return
        except Exception as e:
            logger.info(f"MotorController: pigpio unavailable: {e!r}")

        try:
            import RPi.GPIO  # noqa: F401
            self.backend = MotorRPiGPIO(self.cfg)
            self.backend_name = "RPi.GPIO"
            logger.info("MotorController using RPi.GPIO backend")
            return
        except Exception as e:
            logger.info(f"MotorController: RPi.GPIO unavailable: {e!r}")

        self.backend = MotorDummy()
        self.backend_name = "dummy"
        logger.info("MotorController using Dummy backend")

    def open_loop(self, left: float, right: float):
        self.backend.set_openloop(float(left), float(right))

    def stop(self):
        self.backend.stop()

    def shutdown(self):
        self.backend.shutdown()


# ============================================================
# Smooth joystick / autopilot (v, w) control loop
# ============================================================

# Target and current commands in "JetBot" space:
# v = linear velocity [-1,1], w = angular velocity [-1,1]
target_v = 0.0
target_w = 0.0
current_v = 0.0
current_w = 0.0

# Loop rate and rate limits
JOY_LOOP_HZ = 80.0              # faster control loop
JOY_MAX_DV = 0.035              # max delta per tick in v (accel)
JOY_MAX_DW = 0.06               # max delta per tick in w (accel)

# global caps from config
JOY_MAX_SPEED = float(CONFIG.get("joystick", {}).get("max_speed", 0.5))
JOY_MAX_ROT = float(CONFIG.get("joystick", {}).get("max_rot", 0.8))

# joystick timeout: if no new cmd for this long (and no autopilot) → stop
JOY_CMD_TIMEOUT = 0.25  # seconds
last_cmd_time = 0.0

_control_thread = None
_control_running = True


def vw_to_lr(v: float, w: float, turn_gain: float = 1.0):
    """
    Map JetBot-style (v, w) into normalized left/right wheel speeds.

    v: linear velocity  [-1, 1]
    w: angular velocity [-1, 1]
    turn_gain: how aggressive rotation is relative to forward speed

    IMPORTANT: we DO NOT renormalize both wheels together; instead we
    clamp each wheel independently, so small v/w are preserved.
    """
    left = v - turn_gain * w
    right = v + turn_gain * w

    # clamp each wheel individually
    left = max(-1.0, min(1.0, left))
    right = max(-1.0, min(1.0, right))
    return left, right


def _rate_step(current: float, target: float,
               accel_lim: float, decel_lim: float, brake_lim: float) -> float:
    """
    Asymmetric rate limiter:

    - Small accel limit when speeding up
    - Slightly larger decel limit when slowing down
    - Much larger limit when crossing zero (braking / reversing)
    """
    dv = target - current

    if current * target < 0.0:
        # crossing zero → brake harder
        lim = brake_lim
    elif abs(target) > abs(current):
        # speeding up
        lim = accel_lim
    else:
        # slowing down (same sign)
        lim = decel_lim

    if dv > lim:
        dv = lim
    elif dv < -lim:
        dv = -lim

    return current + dv


def _apply_expo(x: float, expo: float) -> float:
    """
    Exponential sensitivity shaping:
      expo=0 -> linear
      expo>0 -> finer near center, steeper at edges
    """
    sign = 1.0 if x >= 0.0 else -1.0
    ax = abs(x)
    y = expo * (ax ** 3) + (1.0 - expo) * ax
    return sign * y


def _control_loop():
    """
    Runs in the background and smoothly moves current_v/current_w
    towards target_v/target_w, then drives the motors.
    Also enforces a timeout on joystick commands when autopilot is off.
    """
    global current_v, current_w, target_v, target_w, last_cmd_time
    period = 1.0 / JOY_LOOP_HZ

    logger.info("[control] loop started @ %.1f Hz", JOY_LOOP_HZ)
    while _control_running:
        try:
            now = time.time()

            # Joystick timeout: if no command recently and no autopilot → decay target to 0
            if not deploy_status.get("autopilot", False):
                if last_cmd_time > 0.0 and (now - last_cmd_time) > JOY_CMD_TIMEOUT:
                    target_v = 0.0
                    target_w = 0.0

            # Rate-limit v,w with aggressive braking
            current_v = _rate_step(
                current_v,
                target_v,
                accel_lim=JOY_MAX_DV,
                decel_lim=JOY_MAX_DV * 1.2,
                brake_lim=JOY_MAX_DV * 3.0,
            )
            current_w = _rate_step(
                current_w,
                target_w,
                accel_lim=JOY_MAX_DW,
                decel_lim=JOY_MAX_DW * 1.2,
                brake_lim=JOY_MAX_DW * 3.0,
            )

            # Apply caps on v, w BEFORE mixing
            v = max(-JOY_MAX_SPEED, min(JOY_MAX_SPEED, current_v))
            w = max(-JOY_MAX_ROT, min(JOY_MAX_ROT, current_w))

            left, right = vw_to_lr(v, w, turn_gain=1.0)
            motors.open_loop(left, right)
        except Exception:
            logger.exception("[control] error in loop")

        time.sleep(period)

    logger.info("[control] loop stopped")


# ============================================================
# Flask app + globals
# ============================================================

app = Flask(__name__, template_folder="templates", static_folder=None)
CORS(app)

cam_cfg = CONFIG.get("camera", {})
camera = CameraManager(
    width=cam_cfg.get("width", 640),
    height=cam_cfg.get("height", 480),
    fps=cam_cfg.get("fps", 20),
)

motor_cfg = CONFIG.get("motor", {})
motors = MotorController(cfg=motor_cfg)

camera_on = False

# datasets
current_dataset = "default"
os.makedirs(os.path.join(DATASETS_ROOT, current_dataset), exist_ok=True)

# train / deploy stubs
train_status = {"running": False, "epoch": 0, "epochs": 0, "loss": None, "note": ""}
deploy_status = {
    "autopilot": False,
    "gains": {"throttle_gain": 1.0, "steering_gain": 1.0, "expo": 1.0},
    "drive_motors": False,
}
_autopilot_thread = None

# start control loop
_control_thread = threading.Thread(target=_control_loop, daemon=True)
_control_thread.start()


def ok(data=None):
    return jsonify({"ok": True, "data": data})


def err(message, code=400):
    return jsonify({"ok": False, "error": str(message)}), code


# ============================================================
# File system API (simple, rooted at project folder)
# ============================================================

FS_ROOT = ROOT  # you can point this to a subdir if you want


def _safe_path(rel_path: str) -> str:
    rel = os.path.normpath(rel_path).lstrip(os.sep)
    return os.path.join(FS_ROOT, rel)


@app.route("/api/fs/list")
def api_fs_list():
    rel = request.args.get("path", ".")
    base = _safe_path(rel)
    if not os.path.isdir(base):
        return err("not a directory", 400)
    entries = []
    for name in sorted(os.listdir(base)):
        full = os.path.join(base, name)
        st = os.stat(full)
        entries.append({
            "name": name,
            "is_dir": os.path.isdir(full),
            "size": None if os.path.isdir(full) else st.st_size,
        })
    return ok({"path": rel, "entries": entries})


@app.route("/api/fs/read")
def api_fs_read():
    rel = request.args.get("path", "")
    full = _safe_path(rel)
    if not os.path.isfile(full):
        return err("not a file", 400)
    with open(full, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return ok({"path": rel, "content": content})


@app.route("/api/fs/write", methods=["POST"])
def api_fs_write():
    body = request.get_json(silent=True) or {}
    path = body.get("path")
    content = body.get("content", "")
    if not path:
        return err("path required", 400)
    full = _safe_path(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return ok({"path": path})


@app.route("/api/fs/mkdir", methods=["POST"])
def api_fs_mkdir():
    body = request.get_json(silent=True) or {}
    path = body.get("path")
    if not path:
        return err("path required", 400)
    full = _safe_path(path)
    os.makedirs(full, exist_ok=True)
    return ok({"path": path})


@app.route("/api/fs/delete", methods=["POST"])
def api_fs_delete():
    body = request.get_json(silent=True) or {}
    path = body.get("path")
    if not path:
        return err("path required", 400)
    full = _safe_path(path)
    try:
        if os.path.isdir(full):
            os.rmdir(full)
        elif os.path.isfile(full):
            os.remove(full)
        else:
            return err("not found", 404)
    except OSError as e:
        return err(f"delete failed: {e}", 400)
    return ok({"deleted": path})


# ============================================================
# Camera routes
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def api_health():
    return ok({
        "camera_on": camera_on,
        "camera_fps": camera.get_fps(),
        "camera_backend": camera.backend_name,
        "motor_backend": motors.backend_name,
        "time": time.time(),
        "log_path": log_path,
        "current_dataset": current_dataset,
    })


@app.route("/api/camera/start", methods=["POST"])
def api_camera_start():
    global camera_on
    if not camera_on:
        try:
            camera.start()
            camera_on = True
        except Exception as e:
            logger.exception("Failed to start camera")
            return err(str(e), 500)
    return ok({"camera_on": camera_on, "backend": camera.backend_name})


@app.route("/api/camera/stop", methods=["POST"])
def api_camera_stop():
    global camera_on
    if camera_on:
        camera.stop()
        camera_on = False
    return ok({"camera_on": camera_on})


@app.route("/api/camera/frame")
def api_camera_frame():
    frame = camera.get_frame()
    if frame is None:
        return err("no frame", 409)
    ret, jpeg = cv2.imencode(".jpg", frame)
    if not ret:
        return err("encode failed", 500)
    return Response(jpeg.tobytes(), mimetype="image/jpeg")


@app.route("/api/video")
def api_video():
    if not camera_on:
        return err("camera is off", 409)

    def generate():
        while camera_on:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            ret, jpeg = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpeg.tobytes() +
                b"\r\n"
            )

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/camera/settings", methods=["POST"])
def api_camera_settings():
    body = request.get_json(silent=True) or {}
    camera.update_settings(
        vflip=body.get("vflip"),
        hflip=body.get("hflip"),
        gray=body.get("gray"),
        mode=body.get("mode"),
        gamma=body.get("gamma"),
    )
    return ok({
        "vflip": camera.vflip,
        "hflip": camera.hflip,
        "gray": camera.gray,
        "mode": camera.mode,
        "gamma": camera.gamma,
    })


# ============================================================
# Data: labeled clicks → dataset images with X,Y in filename
# ============================================================

def _ensure_dataset(name: str):
    path = os.path.join(DATASETS_ROOT, name)
    os.makedirs(path, exist_ok=True)
    return path


@app.route("/api/datasets")
def api_datasets():
    names = []
    for entry in sorted(os.listdir(DATASETS_ROOT)):
        full = os.path.join(DATASETS_ROOT, entry)
        if os.path.isdir(full):
            names.append(entry)
    return ok({"datasets": names, "current": current_dataset})


@app.route("/api/datasets/select", methods=["POST"])
def api_datasets_select():
    global current_dataset
    body = request.get_json(silent=True) or {}
    name = body.get("name")
    if not name:
        return err("name required", 400)
    _ensure_dataset(name)
    current_dataset = name
    return ok({"current": current_dataset})


@app.route("/api/datasets/summary")
def api_datasets_summary():
    name = request.args.get("name", current_dataset)
    path = _ensure_dataset(name)
    files = sorted(glob.glob(os.path.join(path, "*.jpg")))
    count = len(files)
    return ok({"name": name, "count": count})


@app.route("/api/data/label_click", methods=["POST"])
def api_label_click():
    body = request.get_json(silent=True) or {}
    try:
        x = float(body["x"])
        y = float(body["y"])
        iw = float(body["image_width"])
        ih = float(body["image_height"])
    except Exception:
        return err("x, y, image_width, image_height required", 400)

    if not camera_on:
        return err("camera off", 409)

    raw = camera.get_raw()
    if raw is None:
        return err("no frame", 409)

    h, w = raw.shape[:2]
    # normalize click to [0,1], then to 0..149 label space
    sx = x / max(iw, 1.0)
    sy = y / max(ih, 1.0)
    X = int(np.clip(round(sx * 149), 0, 149))
    Y = int(np.clip(round(sy * 149), 0, 149))

    ds_path = _ensure_dataset(current_dataset)
    ts = int(time.time() * 1000)
    tag = str(body.get("tag", "click"))
    fname = f"x{X:03d}_y{Y:03d}_{tag}_{ts}.jpg"
    full = os.path.join(ds_path, fname)
    cv2.imwrite(full, raw)
    logger.info("Saved labeled frame: %s", full)

    return ok({"file": os.path.relpath(full, ROOT), "X": X, "Y": Y})


# ============================================================
# Logging (stub)
# ============================================================

@app.route("/api/log/start", methods=["POST"])
def api_log_start():
    body = request.get_json(silent=True) or {}
    rate_hz = float(body.get("rate_hz", 5.0))
    tag = str(body.get("tag", "auto"))
    # stub for now
    return ok({"msg": "logging stub", "rate_hz": rate_hz, "tag": tag})


@app.route("/api/log/stop", methods=["POST"])
def api_log_stop():
    return ok({"msg": "logging stopped (stub)"})


# ============================================================
# Motor routes (openloop + smooth v,w command)
# ============================================================

@app.route("/api/motors/openloop", methods=["POST"])
def api_motors_openloop():
    body = request.get_json(silent=True) or {}
    try:
        left = float(body.get("left", 0.0))
        right = float(body.get("right", 0.0))
    except Exception:
        return err("invalid left/right", 400)
    # manual override: send directly
    motors.open_loop(left, right)
    return ok({"left": left, "right": right})


@app.route("/api/motors/stop", methods=["POST"])
def api_motors_stop():
    global target_v, target_w, current_v, current_w, last_cmd_time
    try:
        target_v = 0.0
        target_w = 0.0
        current_v = 0.0
        current_w = 0.0
        last_cmd_time = 0.0
        motors.stop()
        return ok({"stopped": True})
    except Exception as e:
        logger.exception("stop failed")
        return err(str(e), 500)


@app.route("/api/motors/cmd", methods=["POST"])
def api_motors_cmd():
    """
    Set target linear/angular velocity from joystick (raw):
      body: { "v": float, "w": float }

    These are raw [-1,1] joystick values. We do expo + gains + clamping here.
    """
    global target_v, target_w, last_cmd_time
    body = request.get_json(silent=True) or {}
    try:
        v_raw = float(body.get("v", 0.0))
        w_raw = float(body.get("w", 0.0))
    except Exception:
        return err("invalid v/w", 400)

    # clamp raw
    v_raw = max(-1.0, min(1.0, v_raw))
    w_raw = max(-1.0, min(1.0, w_raw))

    # local gains / expo for joystick
    expo_trans = 0.3
    expo_rot = 0.4
    joy_cfg = CONFIG.get("joystick", {})
    speed_gain = float(joy_cfg.get("speed_gain", 1.0))
    steer_gain = float(joy_cfg.get("steer_gain", 1.0))

    v = _apply_expo(v_raw, expo_trans) * speed_gain
    w = _apply_expo(w_raw, expo_rot) * steer_gain

    # clamp shaped commands
    v = max(-1.0, min(1.0, v))
    w = max(-1.0, min(1.0, w))

    target_v = v
    target_w = w
    last_cmd_time = time.time()

    return ok({"v": v, "w": w})


# ============================================================
# Train (stub) + Deploy (stub + autopilot using v,w)
# ============================================================

@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    global train_status
    if train_status["running"]:
        return err("training already running", 409)

    body = request.get_json(silent=True) or {}
    dataset = body.get("dataset", current_dataset)
    epochs = int(body.get("epochs", 10))

    def _run():
        global train_status
        train_status = {
            "running": True,
            "epoch": 0,
            "epochs": epochs,
            "loss": None,
            "note": f"stub train on {dataset}",
        }
        for e in range(epochs):
            time.sleep(0.3)
            train_status["epoch"] = e + 1
            train_status["loss"] = float(np.exp(-0.2 * e))  # fake curve
        train_status["running"] = False
        train_status["note"] = "done (stub)"

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return ok({"status": "started", "dataset": dataset, "epochs": epochs})


@app.route("/api/train/status")
def api_train_status():
    return ok(train_status)


def _autopilot_loop():
    logger.info("[autopilot] loop started")
    global target_v, target_w
    while deploy_status["autopilot"]:
        # Very dumb autopilot: small forward motion only if drive_motors is True
        if deploy_status["drive_motors"]:
            target_v = 0.2 * deploy_status["gains"].get("throttle_gain", 1.0)
            target_w = 0.0
        else:
            target_v = 0.0
            target_w = 0.0
        time.sleep(0.1)
    # stop when leaving
    target_v = 0.0
    target_w = 0.0
    logger.info("[autopilot] loop stopped")


@app.route("/api/deploy/start", methods=["POST"])
def api_deploy_start():
    global _autopilot_thread
    if deploy_status["autopilot"]:
        return err("autopilot already running", 409)
    deploy_status["autopilot"] = True
    _autopilot_thread = threading.Thread(target=_autopilot_loop, daemon=True)
    _autopilot_thread.start()
    return ok({"autopilot": True})


@app.route("/api/deploy/stop", methods=["POST"])
def api_deploy_stop():
    deploy_status["autopilot"] = False
    return ok({"autopilot": False})


@app.route("/api/deploy/status")
def api_deploy_status():
    return ok(deploy_status)


@app.route("/api/deploy/gains", methods=["POST"])
def api_deploy_gains():
    body = request.get_json(silent=True) or {}
    throttle_gain = float(body.get("throttle_gain", deploy_status["gains"]["throttle_gain"]))
    steering_gain = float(body.get("steering_gain", deploy_status["gains"]["steering_gain"]))
    expo = float(body.get("expo", deploy_status["gains"]["expo"]))
    drive_motors = bool(body.get("drive_motors", deploy_status["drive_motors"]))

    deploy_status["gains"].update(
        throttle_gain=throttle_gain,
        steering_gain=steering_gain,
        expo=expo,
    )
    deploy_status["drive_motors"] = drive_motors
    return ok(deploy_status)


# ============================================================
# Cleanup
# ============================================================

@atexit.register
def _cleanup():
    logger.info("Cleanup: shutting down motors and camera...")
    global _control_running
    try:
        _control_running = False
    except Exception:
        pass
    try:
        deploy_status["autopilot"] = False
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
    logger.info("Cleanup complete.")


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    server_cfg = CONFIG.get("server", {})
    host = server_cfg.get("host", "0.0.0.0")
    port = int(server_cfg.get("port", 8888))
    logger.info("Starting AI Car server on %s:%d ...", host, port)
    app.run(host=host, port=port, threaded=True)