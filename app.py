#!/usr/bin/env python3
"""
AI CAR SERVER – SECTION MAP

[1] Config & logging
[2] Camera subsystem
[3] Motor backends
[4] Motor control loop (Hz/alpha/trim/expo)
[5] Flask app setup & globals
[6] File-system API (editor backend)
[7] Camera routes
[8] Datasets & labeling
[9] Logging stub
[10] Motor routes
[11] Train stub
[12] Deploy stub (autopilot stub)
[13] Cleanup & main entry
"""

# ============================================================
# [1] Config & logging
# ============================================================

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


ROOT = os.path.abspath(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(ROOT, "config.json")


def _default_config():
    return {
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


def load_config(path: str = CONFIG_PATH):
    if not os.path.exists(path):
        cfg = _default_config()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        return cfg
    with open(path, "r") as f:
        return json.load(f)


CONFIG = load_config()

DATA_ROOT = os.path.join(ROOT, "data")
LOGS_ROOT = os.path.join(ROOT, CONFIG["paths"]["logs_root"])
DATASETS_ROOT = os.path.join(ROOT, CONFIG["paths"]["datasets_root"])
MODELS_ROOT = os.path.join(ROOT, CONFIG["paths"]["models_root"])

for d in (DATA_ROOT, LOGS_ROOT, DATASETS_ROOT, MODELS_ROOT):
    os.makedirs(d, exist_ok=True)

LOG_PATH = os.path.join(LOGS_ROOT, "app.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("aicar")
logger.info("AI Car server starting...")


def ok(data=None):
    return jsonify({"ok": True, "data": data})


def err(message, code=400):
    return jsonify({"ok": False, "error": str(message)}), code


# ============================================================
# [2] Camera subsystem
# ============================================================


class CameraManager:
    """
    CameraManager:
      1. Try Picamera2 (BGR888)
      2. Try /dev/video* via OpenCV V4L2
      3. Try GStreamer via OpenCV if available

    API:
      - start()
      - stop()
      - get_frame() -> processed BGR frame or None
      - get_raw()   -> raw BGR frame or None
      - get_fps()   -> moving average FPS
      - update_settings(...)
      - backend_name property
    """

    def __init__(self, width: int, height: int, fps: int):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        self._backend = None  # "picamera2" / "v4l2:/dev/videoX" / "gstreamer" / None
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
        self.mode = "raw"  # "raw", "gray", "edges"
        self.gamma = 1.0

        self._gamma_lut = None
        self._gamma_lut_for = None

    # ---------- backend helpers ----------

    @staticmethod
    def _opencv_supports_gst() -> bool:
        try:
            info = cv2.getBuildInformation()
            return "GStreamer" in info and "YES" in info
        except Exception:
            return False

    @staticmethod
    def _list_video_devices():
        return sorted(glob.glob("/dev/video*"))

    # ---------- open / close ----------

    def _open_backend(self) -> bool:
        # Picamera2
        try:
            from picamera2 import Picamera2  # type: ignore

            cam = Picamera2()
            cfg = cam.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "BGR888"},
                buffer_count=3,
            )
            cam.configure(cfg)
            cam.start()
            self._picam2 = cam
            self._backend = "picamera2"
            logger.info("Camera: using Picamera2 backend")
            return True
        except Exception as e:
            logger.info("Camera: Picamera2 not available: %r", e)
            self._picam2 = None

        # V4L2
        for dev in self._list_video_devices():
            try:
                cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                ok_, frame = cap.read()
                if ok_ and frame is not None:
                    self._cap = cap
                    self._backend = f"v4l2:{dev}"
                    logger.info("Camera: using V4L2 backend on %s", dev)
                    return True
                cap.release()
            except Exception as e:
                logger.info("Camera: V4L2 probe failed on %s: %r", dev, e)

        # GStreamer
        if self._opencv_supports_gst():
            try:
                pipeline = (
                    "libcamerasrc ! "
                    f"video/x-raw,width={self.width},height={self.height},"
                    f"framerate={self.fps}/1,format=RGB ! "
                    "videoconvert ! video/x-raw,format=BGR ! "
                    "appsink drop=1 max-buffers=1 sync=false"
                )
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                ok_, frame = cap.read()
                if ok_ and frame is not None:
                    self._cap = cap
                    self._backend = "gstreamer"
                    logger.info("Camera: using GStreamer backend")
                    return True
                cap.release()
            except Exception as e:
                logger.info("Camera: GStreamer backend failed: %r", e)

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
        logger.info("Camera: backend closed")
        self._backend = None

    # ---------- processing ----------

    def _apply_processing(self, frame_bgr):

        frame_bgr = frame_bgr[:, :, ::-1]
        
        if (
            not self.vflip
            and not self.hflip
            and not self.gray
            and self.mode == "raw"
            and abs(self.gamma - 1.0) < 1e-3
        ):
            return frame_bgr

        img = frame_bgr

        if self.vflip:
            img = cv2.flip(img, 0)
        if self.hflip:
            img = cv2.flip(img, 1)

        if abs(self.gamma - 1.0) > 1e-3:
            if self._gamma_lut_for != self.gamma or self._gamma_lut is None:
                inv = 1.0 / max(self.gamma, 1e-6)
                table = (np.arange(256, dtype=np.float32) / 255.0) ** inv * 255.0
                lut = np.clip(table, 0, 255).astype(np.uint8)
                self._gamma_lut = lut
                self._gamma_lut_for = self.gamma
            img = cv2.LUT(img, self._gamma_lut)

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

    # ---------- grabbing ----------

    def _grab_frame_bgr(self):
        if self._backend == "picamera2" and self._picam2 is not None:
            arr = self._picam2.capture_array("main")
            if arr is None:
                return None
            # Picamera2 BGR888 is actually BGR already; if you see swapped colors,
            # flip channels here.
            return arr

        if self._cap is not None:
            ok_, frame = self._cap.read()
            return frame if ok_ else None

        return None

    # ---------- public API ----------

    def start(self):
        if self._running:
            return
        if not self._open_backend():
            raise RuntimeError("Camera: unable to open backend")
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

        logger.info("Camera: capture loop stopped")

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

    def get_fps(self) -> float:
        if not self._fps_hist:
            return 0.0
        return round(sum(self._fps_hist) / len(self._fps_hist), 2)

    @property
    def backend_name(self):
        return self._backend or "none"


# ============================================================
# [3] Motor backends
# ============================================================


class MotorBase:
    def set_openloop(self, left: float, right: float):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def shutdown(self):
        try:
            self.stop()
        except Exception:
            pass


class MotorDummy(MotorBase):
    def __init__(self):
        logger.info("MotorDummy initialized")

    def set_openloop(self, left: float, right: float):
        # Silent dummy; you can log if you want.
        pass

    def stop(self):
        logger.info("MotorDummy stop")

    def shutdown(self):
        logger.info("MotorDummy shutdown")


class MotorPigpio(MotorBase):
    """
    Forward-only: set_openloop(left, right) expects 0..1.
    Direction is handled by which pin is driven HIGH depending on invert flags.
    """

    def __init__(self, cfg: dict):
        import pigpio  # type: ignore

        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpio daemon not running; try 'sudo pigpiod'")

        pins = cfg.get("pins", {})
        self.LF = int(pins.get("left_forward", 5))
        self.LB = int(pins.get("left_backward", 6))
        self.RF = int(pins.get("right_forward", 20))
        self.RB = int(pins.get("right_backward", 21))
        self.LP = pins.get("left_pwm")
        self.RP = pins.get("right_pwm")
        self.STBY = pins.get("stby")

        self.invert_left = bool(cfg.get("invert_left", False))
        self.invert_right = bool(cfg.get("invert_right", False))
        self.freq = int(cfg.get("pwm_freq", 800))
        self.max_duty = max(0, min(100, int(cfg.get("max_duty_pct", 90))))
        self.deadzone = float(cfg.get("deadzone", 0.03))

        for p in (self.LF, self.LB, self.RF, self.RB):
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

    def _set_pwm(self, pwm_pin, duty_pct):
        if pwm_pin is None:
            return
        duty_pct = max(0.0, min(100.0, float(duty_pct)))
        self.pi.set_PWM_dutycycle(int(pwm_pin), int(255 * duty_pct / 100.0))

    def _drive_channel(self, fwd_pin, rev_pin, pwm_pin, value, invert):
        # value is logical forward 0..1
        v = max(0.0, min(1.0, float(value)))

        if v < self.deadzone:
            v = 0.0

        if v <= 0.0:
            # stop
            self.pi.write(int(fwd_pin), 0)
            self.pi.write(int(rev_pin), 0)
            self._set_pwm(pwm_pin, 0)
            return

        duty = v * self.max_duty

        if not invert:
            self.pi.write(int(fwd_pin), 1)
            self.pi.write(int(rev_pin), 0)
        else:
            # wiring inverted: use "reverse" pin as physical forward
            self.pi.write(int(fwd_pin), 0)
            self.pi.write(int(rev_pin), 1)

        self._set_pwm(pwm_pin, duty)

    def set_openloop(self, left: float, right: float):
        self._drive_channel(self.LF, self.LB, self.LP, left, self.invert_left)
        self._drive_channel(self.RF, self.RB, self.RP, right, self.invert_right)

    def stop(self):
        for p in (self.LF, self.LB, self.RF, self.RB):
            self.pi.write(int(p), 0)
        self._set_pwm(self.LP, 0)
        self._set_pwm(self.RP, 0)

    def shutdown(self):
        self.stop()
        self.pi.stop()
        logger.info("MotorPigpio shutdown")


class MotorRPiGPIO(MotorBase):
    def __init__(self, cfg: dict):
        import RPi.GPIO as GPIO  # type: ignore

        self.GPIO = GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        pins = cfg.get("pins", {})
        self.LF = int(pins.get("left_forward", 5))
        self.LB = int(pins.get("left_backward", 6))
        self.RF = int(pins.get("right_forward", 20))
        self.RB = int(pins.get("right_backward", 21))
        self.LP = pins.get("left_pwm")
        self.RP = pins.get("right_pwm")
        self.STBY = pins.get("stby")

        self.invert_left = bool(cfg.get("invert_left", False))
        self.invert_right = bool(cfg.get("invert_right", False))
        self.freq = int(cfg.get("pwm_freq", 800))
        self.max_duty = max(0, min(100, int(cfg.get("max_duty_pct", 90))))
        self.deadzone = float(cfg.get("deadzone", 0.03))

        for p in (self.LF, self.LB, self.RF, self.RB):
            GPIO.setup(p, GPIO.OUT)

        self.pwmL = None
        self.pwmR = None
        if self.LP is not None:
            GPIO.setup(self.LP, GPIO.OUT)
            self.pwmL = GPIO.PWM(self.LP, self.freq)
            self.pwmL.start(0)
        if self.RP is not None:
            GPIO.setup(self.RP, GPIO.OUT)
            self.pwmR = GPIO.PWM(self.RP, self.freq)
            self.pwmR.start(0)

        if self.STBY is not None:
            GPIO.setup(self.STBY, GPIO.OUT)
            GPIO.output(self.STBY, GPIO.HIGH)

        self.stop()
        logger.info("MotorRPiGPIO initialized")

    def _set_pwm(self, pwm, duty_pct):
        if pwm is None:
            return
        duty_pct = max(0.0, min(100.0, float(duty_pct)))
        pwm.ChangeDutyCycle(duty_pct)

    def _drive_channel(self, fwd_pin, rev_pin, pwm, value, invert):
        GPIO = self.GPIO
        v = max(0.0, min(1.0, float(value)))

        if v < self.deadzone:
            v = 0.0

        if v <= 0.0:
            GPIO.output(fwd_pin, GPIO.LOW)
            GPIO.output(rev_pin, GPIO.LOW)
            self._set_pwm(pwm, 0)
            return

        duty = v * self.max_duty

        if not invert:
            GPIO.output(fwd_pin, GPIO.HIGH)
            GPIO.output(rev_pin, GPIO.LOW)
        else:
            GPIO.output(fwd_pin, GPIO.LOW)
            GPIO.output(rev_pin, GPIO.HIGH)

        self._set_pwm(pwm, duty)

    def set_openloop(self, left: float, right: float):
        self._drive_channel(self.LF, self.LB, self.pwmL, left, self.invert_left)
        self._drive_channel(self.RF, self.RB, self.pwmR, right, self.invert_right)

    def stop(self):
        G = self.GPIO
        G.output(self.LF, G.LOW)
        G.output(self.LB, G.LOW)
        G.output(self.RF, G.LOW)
        G.output(self.RB, G.LOW)
        self._set_pwm(self.pwmL, 0)
        self._set_pwm(self.pwmR, 0)

    def shutdown(self):
        self.stop()
        if self.pwmL is not None:
            self.pwmL.stop()
        if self.pwmR is not None:
            self.pwmR.stop()
        self.GPIO.cleanup()
        logger.info("MotorRPiGPIO shutdown")


class MotorController:
    """
    Tries pigpio -> RPi.GPIO -> Dummy.
    Exposes open_loop(left,right), stop(), shutdown(), backend_name.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self.backend = None
        self.backend_name = "dummy"

        # pigpio
        try:
            import pigpio  # noqa: F401

            self.backend = MotorPigpio(self.cfg)
            self.backend_name = "pigpio"
            logger.info("MotorController backend: pigpio")
            return
        except Exception as e:
            logger.info("MotorController: pigpio unavailable: %r", e)

        # RPi.GPIO
        try:
            import RPi.GPIO  # noqa: F401

            self.backend = MotorRPiGPIO(self.cfg)
            self.backend_name = "RPi.GPIO"
            logger.info("MotorController backend: RPi.GPIO")
            return
        except Exception as e:
            logger.info("MotorController: RPi.GPIO unavailable: %r", e)

        # Dummy
        self.backend = MotorDummy()
        self.backend_name = "dummy"
        logger.info("MotorController backend: dummy")

    def open_loop(self, left: float, right: float):
        # clamp to 0..1 before hardware
        l = max(0.0, min(1.0, float(left)))
        r = max(0.0, min(1.0, float(right)))
        self.backend.set_openloop(l, r)

    def stop(self):
        self.backend.stop()

    def shutdown(self):
        self.backend.shutdown()


# ============================================================
# [4] Motor control loop (Hz/alpha/trim/expo)
# ============================================================

CONTROL_DEFAULT_HZ = 100.0
CONTROL_DEFAULT_ALPHA = 0.35
TRIM_DEFAULT_LEFT = 1.0
TRIM_DEFAULT_RIGHT = 1.0
EXPO_DEFAULT_THROTTLE = 1.3

control_state = {
    "left_target": 0.0,   # logical 0..1
    "right_target": 0.0,
    "left_actual": 0.0,   # smoothed logical 0..1
    "right_actual": 0.0,
}
control_params = {
    "hz": CONTROL_DEFAULT_HZ,
    "alpha": CONTROL_DEFAULT_ALPHA,
    "left_trim": TRIM_DEFAULT_LEFT,
    "right_trim": TRIM_DEFAULT_RIGHT,
    "throttle_expo": EXPO_DEFAULT_THROTTLE,
}
control_lock = threading.Lock()

motor_loop_running = True

def _apply_expo01(x: float, expo: float) -> float:
    """
    Expo on [0,1]: returns x**expo, clamped.
    expo > 1 = softer near 0, still reaches 1 at 1.
    """
    expo = max(0.1, min(5.0, float(expo)))
    v = max(0.0, min(1.0, float(x)))
    return v ** expo

def motor_loop(motors: MotorController):
    global motor_loop_running
    logger.info("[motor_loop] started")
    while motor_loop_running:
        with control_lock:
            hz = float(control_params.get("hz", CONTROL_DEFAULT_HZ))
            alpha = float(control_params.get("alpha", CONTROL_DEFAULT_ALPHA))
            left_trim = float(control_params.get("left_trim", TRIM_DEFAULT_LEFT))
            right_trim = float(control_params.get("right_trim", TRIM_DEFAULT_RIGHT))
            throttle_expo = float(control_params.get("throttle_expo", EXPO_DEFAULT_THROTTLE))

            hz = max(5.0, min(200.0, hz))
            alpha = max(0.0, min(1.0, alpha))
            left_trim = max(0.5, min(1.5, left_trim))
            right_trim = max(0.5, min(1.5, right_trim))
            throttle_expo = max(0.5, min(3.0, throttle_expo))

            lt = max(0.0, min(1.0, control_state["left_target"]))
            rt = max(0.0, min(1.0, control_state["right_target"]))
            la = max(0.0, min(1.0, control_state["left_actual"]))
            ra = max(0.0, min(1.0, control_state["right_actual"]))

        dt = 1.0 / hz

        # logical smoothing
        la = la + alpha * (lt - la)
        ra = ra + alpha * (rt - ra)

        # expo + trim
        la_cmd = _apply_expo01(la, throttle_expo) * left_trim
        ra_cmd = _apply_expo01(ra, throttle_expo) * right_trim

        la_cmd = max(0.0, min(1.0, la_cmd))
        ra_cmd = max(0.0, min(1.0, ra_cmd))

        try:
            motors.open_loop(la_cmd, ra_cmd)
        except Exception as e:
            logger.exception("motor_loop: open_loop failed, stopping loop")
            break

        with control_lock:
            control_state["left_actual"] = la
            control_state["right_actual"] = ra

        time.sleep(dt)

    logger.info("[motor_loop] exiting")


# ============================================================
# [5] Flask app setup & globals
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

# datasets
current_dataset = "default"
os.makedirs(os.path.join(DATASETS_ROOT, current_dataset), exist_ok=True)

# train / deploy stubs
train_status = {
    "running": False,
    "epoch": 0,
    "epochs": 0,
    "loss": None,
    "note": "stub trainer",
}
deploy_status = {
    "autopilot": False,
    "gains": {"throttle_gain": 1.0, "steering_gain": 1.0, "expo": 1.0},
    "drive_motors": False,
}

_autopilot_thread = None

# start motor control loop once
_motor_thread = threading.Thread(target=motor_loop, args=(motors,), daemon=True)
_motor_thread.start()


@app.route("/api/control/params", methods=["GET", "POST"])
def api_control_params():
    if request.method == "GET":
        with control_lock:
            return ok(control_params.copy())

    body = request.get_json(silent=True) or {}
    with control_lock:
        if "hz" in body:
            hz = float(body["hz"])
            control_params["hz"] = max(5.0, min(200.0, hz))
        if "alpha" in body:
            a = float(body["alpha"])
            control_params["alpha"] = max(0.0, min(1.0, a))
        if "left_trim" in body:
            lt = float(body["left_trim"])
            control_params["left_trim"] = max(0.5, min(1.5, lt))
        if "right_trim" in body:
            rt = float(body["right_trim"])
            control_params["right_trim"] = max(0.5, min(1.5, rt))
        if "throttle_expo" in body:
            te = float(body["throttle_expo"])
            control_params["throttle_expo"] = max(0.5, min(3.0, te))
        return ok(control_params.copy())


# ============================================================
# [6] File-system API (editor backend)
# ============================================================

FS_ROOT = ROOT  # project root; change if you want a subdirectory


def _safe_path(rel: str) -> str:
    rel_norm = os.path.normpath(rel).lstrip(os.sep)
    full = os.path.join(FS_ROOT, rel_norm)
    if not full.startswith(FS_ROOT):
        raise ValueError("path escapes root")
    return full


@app.route("/api/fs/list")
def api_fs_list():
    rel = request.args.get("path", ".")
    try:
        base = _safe_path(rel)
    except ValueError:
        return err("invalid path", 400)
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
    try:
        full = _safe_path(rel)
    except ValueError:
        return err("invalid path", 400)
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
    try:
        full = _safe_path(path)
    except ValueError:
        return err("invalid path", 400)
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
    try:
        full = _safe_path(path)
    except ValueError:
        return err("invalid path", 400)
    os.makedirs(full, exist_ok=True)
    return ok({"path": path})


@app.route("/api/fs/delete", methods=["POST"])
def api_fs_delete():
    body = request.get_json(silent=True) or {}
    path = body.get("path")
    if not path:
        return err("path required", 400)
    try:
        full = _safe_path(path)
    except ValueError:
        return err("invalid path", 400)
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
# [7] Camera routes
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
            "current_dataset": current_dataset,
            "time": time.time(),
            "log_path": LOG_PATH,
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
    ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ret:
        return err("encode failed", 500)
    return Response(jpeg.tobytes(), mimetype="image/jpeg")


@app.route("/api/video")
def api_video():
    if not camera_on:
        return err("camera is off", 409)

    def gen():
        while camera_on:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ret:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


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

    width = max(80, min(1280, width))
    height = max(60, min(720, height))
    fps = max(1, min(60, fps))

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
            logger.exception("Failed to restart camera")
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
# [8] Datasets & labeling
# ============================================================

@app.route("/api/datasets")
def api_datasets():
    ds = []
    for name in sorted(os.listdir(DATASETS_ROOT)):
        full = os.path.join(DATASETS_ROOT, name)
        if os.path.isdir(full):
            ds.append(name)
    return ok({"datasets": ds, "current": current_dataset})


@app.route("/api/datasets/select", methods=["POST"])
def api_datasets_select():
    global current_dataset
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        return err("name required", 400)
    full = os.path.join(DATASETS_ROOT, name)
    os.makedirs(full, exist_ok=True)
    current_dataset = name
    return ok({"current": current_dataset})


@app.route("/api/datasets/summary")
def api_datasets_summary():
    name = request.args.get("name", current_dataset)
    full = os.path.join(DATASETS_ROOT, name)
    if not os.path.isdir(full):
        return err("dataset not found", 404)
    count = len([f for f in os.listdir(full) if f.lower().endswith(".jpg")])
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
        return err("x,y,image_width,image_height required", 400)

    tag = str(body.get("tag", "click"))

    frame = camera.get_raw()
    if frame is None:
        return err("no camera frame", 409)

    h, w = frame.shape[:2]
    sx = max(0.0, min(1.0, x / max(1.0, iw)))
    sy = max(0.0, min(1.0, y / max(1.0, ih)))

    X = int(round(sx * 149))
    Y = int(round(sy * 149))
    X = max(0, min(149, X))
    Y = max(0, min(149, Y))

    ds_dir = os.path.join(DATASETS_ROOT, current_dataset)
    os.makedirs(ds_dir, exist_ok=True)
    ts = int(time.time() * 1000)
    fname = f"x{X:03d}_y{Y:03d}_{tag}_{ts}.jpg"
    path = os.path.join(ds_dir, fname)
    cv2.imwrite(path, frame)

    return ok({"file": fname, "X": X, "Y": Y, "dataset": current_dataset})


# ============================================================
# [9] Logging stub
# ============================================================

log_session = {"active": False, "rate_hz": 0.0, "tag": ""}


@app.route("/api/log/start", methods=["POST"])
def api_log_start():
    body = request.get_json(silent=True) or {}
    rate = float(body.get("rate_hz", 5.0))
    tag = str(body.get("tag", "auto"))
    log_session["active"] = True
    log_session["rate_hz"] = rate
    log_session["tag"] = tag
    return ok(log_session.copy())


@app.route("/api/log/stop", methods=["POST"])
def api_log_stop():
    log_session["active"] = False
    return ok(log_session.copy())


# ============================================================
# [10] Motor routes
# ============================================================

@app.route("/api/motors/openloop", methods=["POST"])
def api_motors_openloop():
    body = request.get_json(silent=True) or {}
    try:
        left = float(body.get("left", 0.0))
        right = float(body.get("right", 0.0))
    except Exception:
        return err("invalid left/right", 400)

    left = max(0.0, min(1.0, left))
    right = max(0.0, min(1.0, right))

    with control_lock:
        control_state["left_target"] = left
        control_state["right_target"] = right

    return ok({"left": left, "right": right})


@app.route("/api/motors/stop", methods=["POST"])
def api_motors_stop():
    with control_lock:
        control_state["left_target"] = 0.0
        control_state["right_target"] = 0.0
    motors.stop()
    return ok({"stopped": True})


# ============================================================
# [11] Train stub
# ============================================================

def _train_loop(dataset: str, epochs: int, lr: float, batch_size: int):
    train_status["running"] = True
    train_status["epoch"] = 0
    train_status["epochs"] = epochs
    train_status["note"] = f"stub training on {dataset}"
    losses = []

    for e in range(1, epochs + 1):
        if not train_status["running"]:
            break
        time.sleep(0.5)
        loss = max(0.01, 1.0 / e)
        losses.append(loss)
        train_status["epoch"] = e
        train_status["loss"] = loss

    train_status["running"] = False
    train_status["note"] = "finished stub training"
    logger.info("Stub training finished: %s", losses)


@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    if train_status["running"]:
        return err("training already running", 409)
    body = request.get_json(silent=True) or {}
    dataset = (body.get("dataset") or current_dataset).strip()
    epochs = int(body.get("epochs", 10))
    lr = float(body.get("lr", 0.001))
    batch_size = int(body.get("batch_size", 32))
    t = threading.Thread(
        target=_train_loop,
        args=(dataset, epochs, lr, batch_size),
        daemon=True,
    )
    t.start()
    return ok(
        {
            "dataset": dataset,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "running": True,
        }
    )


@app.route("/api/train/status")
def api_train_status():
    return ok(train_status.copy())


# ============================================================
# [12] Deploy stub (autopilot stub)
# ============================================================

def _autopilot_loop():
    logger.info("Autopilot stub started")
    while deploy_status["autopilot"]:
        time.sleep(0.2)
        # could update some fake internal state here
    logger.info("Autopilot stub stopped")


@app.route("/api/deploy/start", methods=["POST"])
def api_deploy_start():
    global _autopilot_thread
    if deploy_status["autopilot"]:
        return err("autopilot already running", 409)
    deploy_status["autopilot"] = True
    _autopilot_thread = threading.Thread(target=_autopilot_loop, daemon=True)
    _autopilot_thread.start()
    return ok(deploy_status.copy())


@app.route("/api/deploy/stop", methods=["POST"])
def api_deploy_stop():
    deploy_status["autopilot"] = False
    return ok(deploy_status.copy())


@app.route("/api/deploy/status")
def api_deploy_status():
    return ok(deploy_status.copy())


@app.route("/api/deploy/gains", methods=["POST"])
def api_deploy_gains():
    body = request.get_json(silent=True) or {}
    g = deploy_status["gains"]
    if "throttle_gain" in body:
        g["throttle_gain"] = float(body["throttle_gain"])
    if "steering_gain" in body:
        g["steering_gain"] = float(body["steering_gain"])
    if "expo" in body:
        g["expo"] = float(body["expo"])
    if "drive_motors" in body:
        deploy_status["drive_motors"] = bool(body["drive_motors"])
    return ok(deploy_status.copy())


# ============================================================
# [13] Cleanup & main entry
# ============================================================

@atexit.register
def _cleanup():
    logger.info("Shutting down AI Car server...")
    global motor_loop_running
    try:
        deploy_status["autopilot"] = False
    except Exception:
        pass

    # Stop motor loop before killing pigpio / GPIO
    motor_loop_running = False
    time.sleep(0.05)

    try:
        motors.shutdown()
    except Exception:
        logger.exception("Error during motors.shutdown()")
    try:
        camera.stop()
    except Exception:
        logger.exception("Error during camera.stop()")
    logger.info("Cleanup complete")

if __name__ == "__main__":
    host = CONFIG["server"].get("host", "0.0.0.0")
    port = int(CONFIG["server"].get("port", 8888))
    logger.info("Starting Flask on %s:%d", host, port)
    app.run(host=host, port=port, threaded=True)
