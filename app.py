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
        # minimal default config (smaller camera + fps to reduce load)
        cfg = {
            "server": {"host": "0.0.0.0", "port": 8888},
            "camera": {"width": 320, "height": 240, "fps": 15},
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
                # default: no inversion, easier to reason about
                "invert_left": False,
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

for d in (DATA_ROOT, LOG_ROOT, DATASETS_ROOT, MODELS_ROOT):
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

    def __init__(self, width=320, height=240, fps=15):
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

        # gamma LUT cache (for speed)
        self._gamma_lut = None
        self._gamma_lut_for = None

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
        # Fast path: absolutely raw, no processing
        if (
            not self.vflip
            and not self.hflip
            and not self.gray
            and self.mode == "raw"
            and abs(self.gamma - 1.0) < 1e-3
        ):
            return frame_bgr

        img = frame_bgr

        # flips
        if self.vflip:
            img = cv2.flip(img, 0)
        if self.hflip:
            img = cv2.flip(img, 1)

        # gamma (with cached LUT)
        if abs(self.gamma - 1.0) > 1e-3:
            if self._gamma_lut_for != self.gamma or self._gamma_lut is None:
                inv = 1.0 / max(self.gamma, 1e-6)
                table = (np.arange(256, dtype=np.float32) / 255.0) ** inv * 255.0
                self._gamma_lut = np.clip(table, 0, 255).astype(np.uint8)
                self._gamma_lut_for = self.gamma
            img = cv2.LUT(img, self._gamma_lut)

        # mode / gray / edges
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
            arr = self._picam2.capture_array("main")
            if arr is None:
                return None
            # NOTE: if Picamera2 already returns BGR, you can drop the next line.
            arr = arr[..., ::-1]
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
        pass

    def stop(self):
        logger.info("[MOTOR dummy] stop")

    def shutdown(self):
        logger.info("MotorDummy shutdown")


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
        """
        Bidirectional control:
          - value in [-1.0, 1.0]
          - sign = direction (forward / reverse)
          - magnitude = speed
          - invert flips the sign
        """
        v = float(value)
        # clamp to [-1, 1]
        if v > 1.0:
            v = 1.0
        if v < -1.0:
            v = -1.0

        if invert:
            v = -v

        # deadzone
        if abs(v) < self.deadzone:
            v = 0.0

        if v > 0:
            # forward
            self.pi.write(int(fwd_pin), 1)
            self.pi.write(int(rev_pin), 0)
            duty = v * self.max_duty
        elif v < 0:
            # reverse
            self.pi.write(int(fwd_pin), 0)
            self.pi.write(int(rev_pin), 1)
            duty = (-v) * self.max_duty  # use magnitude
        else:
            # stop
            self.pi.write(int(fwd_pin), 0)
            self.pi.write(int(rev_pin), 0)
            duty = 0.0

        self._set_pwm(pwm_pin, duty)

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

    def _drive_channel(self, pwm_pin, fwd_pin, rev_pin, value, invert):
        """
        Bidirectional control, same behavior as pigpio:
          - value in [-1.0, 1.0]
          - sign = direction
          - invert flips sign
        """
        GPIO = self.GPIO
        v = float(value)

        # clamp to [-1, 1]
        if v > 1.0:
            v = 1.0
        if v < -1.0:
            v = -1.0

        if invert:
            v = -v

        # deadzone
        if abs(v) < self.deadzone:
            v = 0.0

        if v > 0:
            # forward
            GPIO.output(fwd_pin, GPIO.HIGH)
            GPIO.output(rev_pin, GPIO.LOW)
            duty = v * self.max_duty
        elif v < 0:
            # reverse
            GPIO.output(fwd_pin, GPIO.LOW)
            GPIO.output(rev_pin, GPIO.HIGH)
            duty = (-v) * self.max_duty
        else:
            # stop
            GPIO.output(fwd_pin, GPIO.LOW)
            GPIO.output(rev_pin, GPIO.LOW)
            duty = 0.0

        self._set_pwm(pwm_pin, duty)

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
# Flask app + globals
# ============================================================

app = Flask(__name__, template_folder="templates", static_folder=None)
CORS(app)

cam_cfg = CONFIG.get("camera", {})
camera = CameraManager(
    width=cam_cfg.get("width", 320),
    height=cam_cfg.get("height", 240),
    fps=cam_cfg.get("fps", 15),
)

motor_cfg = CONFIG.get("motor", {})
motors = MotorController(cfg=motor_cfg)

camera_on = False

# --- motor control state (smooth, prioritized loop) ---
CONTROL_DEFAULT_HZ = 100.0       # faster loop for less lag
CONTROL_DEFAULT_ALPHA = 0.35     # smoothing factor (0=very smooth, 1=no smoothing)
TRIM_DEFAULT_LEFT = 1.0
TRIM_DEFAULT_RIGHT = 1.0
EXPO_DEFAULT_THROTTLE = 1.3      # >1 = softer near center

control_state = {
    "left_target": 0.0,   # logical [-1,1]
    "right_target": 0.0,  # logical [-1,1]
    "left_actual": 0.0,   # smoothed logical
    "right_actual": 0.0,
}
control_lock = threading.Lock()

# tunable control parameters (exposed to web UI)
control_params = {
    "hz": CONTROL_DEFAULT_HZ,
    "alpha": CONTROL_DEFAULT_ALPHA,
    "left_trim": TRIM_DEFAULT_LEFT,
    "right_trim": TRIM_DEFAULT_RIGHT,
    "throttle_expo": EXPO_DEFAULT_THROTTLE,
}

def _apply_expo(x: float, expo: float) -> float:
    """
    Simple expo curve: sign(x) * |x|^expo
    expo > 1.0  -> softer near center (more gentle control)
    expo = 1.0  -> linear
    expo < 1.0  -> aggressive near center (not recommended here)
    """
    v = float(x)
    expo = max(0.1, min(5.0, float(expo)))
    s = 1.0 if v >= 0 else -1.0
    return s * (abs(v) ** expo)


def motor_loop():
    logger.info("[motor_loop] started (tunable Hz/alpha/trim/expo)")
    while True:
        # snapshot under lock
        with control_lock:
            hz = float(control_params.get("hz", CONTROL_DEFAULT_HZ))
            alpha = float(control_params.get("alpha", CONTROL_DEFAULT_ALPHA))
            left_trim = float(control_params.get("left_trim", TRIM_DEFAULT_LEFT))
            right_trim = float(control_params.get("right_trim", TRIM_DEFAULT_RIGHT))
            throttle_expo = float(control_params.get("throttle_expo", EXPO_DEFAULT_THROTTLE))

            # clamp to safe ranges
            hz = max(5.0, min(200.0, hz))           # 5..200 Hz
            alpha = max(0.0, min(1.0, alpha))       # 0..1
            left_trim = max(0.5, min(1.5, left_trim))
            right_trim = max(0.5, min(1.5, right_trim))
            throttle_expo = max(0.5, min(3.0, throttle_expo))

            lt = control_state["left_target"]
            rt = control_state["right_target"]
            la = control_state["left_actual"]
            ra = control_state["right_actual"]

        dt = 1.0 / hz

        # smooth transitions in logical space
        la = la + alpha * (lt - la)
        ra = ra + alpha * (rt - ra)

        # apply expo + trim just before sending to motors
        la_cmd = _apply_expo(la, throttle_expo) * left_trim
        ra_cmd = _apply_expo(ra, throttle_expo) * right_trim

        # clamp final outputs to [-1,1]
        la_cmd = max(-1.0, min(1.0, la_cmd))
        ra_cmd = max(-1.0, min(1.0, ra_cmd))

        motors.open_loop(la_cmd, ra_cmd)

        # write back smoothed logical values (not trimmed)
        with control_lock:
            control_state["left_actual"] = la
            control_state["right_actual"] = ra

        time.sleep(dt)

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


def ok(data=None):
    return jsonify({"ok": True, "data": data})


def err(message, code=400):
    return jsonify({"ok": False, "error": str(message)}), code


# ============================================================
# File system API
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
        entries.append(
            {
                "name": name,
                "is_dir": os.path.isdir(full),
                "size": None if os.path.isdir(full) else st.st_size,
            }
        )
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
    return ok(
        {
            "camera_on": camera_on,
            "camera_fps": camera.get_fps(),
            "camera_backend": camera.backend_name,
            "motor_backend": motors.backend_name,
            "time": time.time(),
            "log_path": log_path,
            "current_dataset": current_dataset,
        }
    )


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
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 70]  # lower quality for smaller size
    ret, jpeg = cv2.imencode(".jpg", frame, encode_params)
    if not ret:
        return err("encode failed", 500)
    return Response(jpeg.tobytes(), mimetype="image/jpeg")


@app.route("/api/video")
def api_video():
    if not camera_on:
        return err("camera is off", 409)

    def generate():
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        while camera_on:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            ret, jpeg = cv2.imencode(".jpg", frame, encode_params)
            if not ret:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
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
    return ok(
        {
            "vflip": camera.vflip,
            "hflip": camera.hflip,
            "gray": camera.gray,
            "mode": camera.mode,
            "gamma": camera.gamma,
        }
    )

@app.route("/api/camera/config", methods=["GET", "POST"])
def api_camera_config():
    global camera_on

    if request.method == "GET":
        return ok(
            {
                "width": camera.width,
                "height": camera.height,
                "fps": camera.fps,
                "camera_on": camera_on,
            }
        )

    # POST: update (fps and optionally width/height)
    body = request.get_json(silent=True) or {}

    width = body.get("width", camera.width)
    height = body.get("height", camera.height)
    fps = body.get("fps", camera.fps)

    try:
        width = int(width)
        height = int(height)
        fps = int(fps)
    except Exception:
        return err("invalid width/height/fps", 400)

    # clamp a bit to keep Pi happy
    width = max(80, min(1280, width))
    height = max(60, min(720, height))
    fps = max(1, min(60, fps))

    # if camera is running, restart with new settings
    was_on = camera_on
    if was_on:
        camera.stop()
        camera_on = False

    camera.width = width
    camera.height = height
    camera.fps = fps

    if was_on:
        try:
            camera.start()
            camera_on = True
        except Exception as e:
            logger.exception("Failed to restart camera with new config")
            return err(str(e), 500)

    return ok(
        {
            "width": camera.width,
            "height": camera.height,
            "fps": camera.fps,
            "camera_on": camera_on,
        }
    )

# ============================================================
# Data: labeled clicks
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
# Motor routes (openloop)
# ============================================================

@app.route("/api/control/params", methods=["GET", "POST"])
def api_control_params():
    global control_params
    if request.method == "GET":
        with control_lock:
            return ok(
                {
                    "hz": control_params.get("hz", CONTROL_DEFAULT_HZ),
                    "alpha": control_params.get("alpha", CONTROL_DEFAULT_ALPHA),
                    "left_trim": control_params.get("left_trim", TRIM_DEFAULT_LEFT),
                    "right_trim": control_params.get("right_trim", TRIM_DEFAULT_RIGHT),
                    "throttle_expo": control_params.get("throttle_expo", EXPO_DEFAULT_THROTTLE),
                }
            )

    # POST: update
    body = request.get_json(silent=True) or {}

    hz = body.get("hz", control_params.get("hz", CONTROL_DEFAULT_HZ))
    alpha = body.get("alpha", control_params.get("alpha", CONTROL_DEFAULT_ALPHA))
    left_trim = body.get("left_trim", control_params.get("left_trim", TRIM_DEFAULT_LEFT))
    right_trim = body.get("right_trim", control_params.get("right_trim", TRIM_DEFAULT_RIGHT))
    throttle_expo = body.get("throttle_expo", control_params.get("throttle_expo", EXPO_DEFAULT_THROTTLE))

    try:
        hz = float(hz)
        alpha = float(alpha)
        left_trim = float(left_trim)
        right_trim = float(right_trim)
        throttle_expo = float(throttle_expo)
    except Exception:
        return err("invalid control params", 400)

    # clamp
    hz = max(5.0, min(200.0, hz))
    alpha = max(0.0, min(1.0, alpha))
    left_trim = max(0.5, min(1.5, left_trim))
    right_trim = max(0.5, min(1.5, right_trim))
    throttle_expo = max(0.5, min(3.0, throttle_expo))

    with control_lock:
        control_params["hz"] = hz
        control_params["alpha"] = alpha
        control_params["left_trim"] = left_trim
        control_params["right_trim"] = right_trim
        control_params["throttle_expo"] = throttle_expo

    return ok(
        {
            "hz": hz,
            "alpha": alpha,
            "left_trim": left_trim,
            "right_trim": right_trim,
            "throttle_expo": throttle_expo,
        }
    )

@app.route("/api/motors/openloop", methods=["POST"])
def api_motors_openloop():
    body = request.get_json(silent=True) or {}
    try:
        left = float(body.get("left", 0.0))
        right = float(body.get("right", 0.0))
    except Exception:
        return err("invalid left/right", 400)

    # bidirectional: clamp to [-1.0, 1.0]
    left = max(-1.0, min(1.0, left))
    right = max(-1.0, min(1.0, right))

    with control_lock:
        control_state["left_target"] = left
        control_state["right_target"] = right

    return ok({"left": left, "right": right})


@app.route("/api/motors/stop", methods=["POST"])
def api_motors_stop():
    try:
        with control_lock:
            control_state["left_target"] = 0.0
            control_state["right_target"] = 0.0
        motors.stop()
        return ok({"stopped": True})
    except Exception as e:
        logger.exception("stop failed")
        return err(str(e), 500)


# ============================================================
# Train (stub) + Deploy (stub autopilot)
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
    while deploy_status["autopilot"]:
        if deploy_status["drive_motors"]:
            throttle_gain = deploy_status["gains"].get("throttle_gain", 1.0)
            throttle = 0.3 * throttle_gain
            throttle = max(0.0, min(1.0, throttle))

            # Straight for now; later plug in (x,y) → steering here
            left_cmd = throttle
            right_cmd = throttle

            with control_lock:
                control_state["left_target"] = left_cmd
                control_state["right_target"] = right_cmd
        else:
            with control_lock:
                control_state["left_target"] = 0.0
                control_state["right_target"] = 0.0

        time.sleep(0.1)

    # ensure stop at end
    with control_lock:
        control_state["left_target"] = 0.0
        control_state["right_target"] = 0.0
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
    try:
        deploy_status["autopilot"] = False
    except Exception:
        pass
    try:
        with control_lock:
            control_state["left_target"] = 0.0
            control_state["right_target"] = 0.0
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

    # start motor control loop
    t_motor = threading.Thread(target=motor_loop, daemon=True)
    t_motor.start()

    logger.info("Starting AI Car server on %s:%d ...", host, port)
    app.run(host=host, port=port, threaded=True)