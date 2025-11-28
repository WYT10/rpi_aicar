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
from flask import Flask, jsonify, request, Response, render_template, send_file
from flask_cors import CORS
import re

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    Dataset = None
    DataLoader = None

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
train_cfg = CONFIG.get("train", {})
autopilot_cfg = CONFIG.get("autopilot", {})

DATA_ROOT = os.path.join(ROOT, "data")
LOGS_ROOT = os.path.join(ROOT, CONFIG["paths"]["logs_root"])
DATASETS_ROOT = os.path.join(ROOT, CONFIG["paths"]["datasets_root"])
MODELS_ROOT = os.path.join(ROOT, CONFIG["paths"]["models_root"])

for d in (DATA_ROOT, LOGS_ROOT, DATASETS_ROOT, MODELS_ROOT):
    os.makedirs(d, exist_ok=True)

if TORCH_AVAILABLE:
    class TinySteerNet(nn.Module):
        """
        Small CNN that maps an image -> [left,right] in [0,1].
        """
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 16, 5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 48, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )
            self.fc = nn.Sequential(
                nn.Linear(48 * 8 * 12, 64),  # based on 96x64 input
                nn.ReLU(inplace=True),
                nn.Linear(64, 2),
                nn.Sigmoid(),  # outputs in [0,1]
            )

        def forward(self, x):
            z = self.conv(x)
            z = z.view(z.size(0), -1)
            return self.fc(z)
else:
    TinySteerNet = None

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

# train / deploy state
train_status = {
    "running": False,
    "epoch": 0,
    "epochs": 0,
    "loss": None,
    "note": "",
    "dataset": None,
    "model_path": None,
}

deploy_status = {
    "autopilot": False,
    "model": None,
    "gains": {"throttle_gain": 1.0, "steering_gain": 1.0, "expo": 1.0},
    "drive_motors": False,
}
_autopilot_thread = None

# autopilot model cache
autopilot_model = None
autopilot_device = "cpu"
autopilot_image_size = (
    int(train_cfg.get("image_width", 96)),
    int(train_cfg.get("image_height", 64)),
)

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

@app.route("/api/fs/raw")
def api_fs_raw():
    """
    Return a raw file (useful for image preview in the File tab).
    """
    rel = request.args.get("path", "")
    try:
        full = _safe_path(rel)
    except ValueError:
        return err("invalid path", 400)
    if not os.path.isfile(full):
        return err("not a file", 404)
    # Let Flask guess mimetype (jpg/png/etc.)
    return send_file(full)


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

def _ensure_dataset(name: str) -> str:
    """
    Ensure dataset directory exists and return its absolute path.
    """
    name = (name or "").strip() or "default"
    ds_dir = os.path.join(DATASETS_ROOT, name)
    os.makedirs(ds_dir, exist_ok=True)
    return ds_dir


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
    _ensure_dataset(name)
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
    """
    Click-to-label.

    We encode the label into the filename:

      x{X:03d}_y{Y:03d}_{tag}_{ts}.jpg

    where X,Y are on a 150x150 grid (0..149). Training can recover
    normalized sx,sy and derive left/right later.
    """
    body = request.get_json(silent=True) or {}
    try:
        x = float(body["x"])
        y = float(body["y"])
        iw = float(body["image_width"])
        ih = float(body["image_height"])
    except Exception:
        return err("x,y,image_width,image_height required", 400)

    if not camera_on:
        return err("camera off", 409)

    frame = camera.get_frame()
    if frame is None:
        return err("no camera frame", 409)

    # normalize click to [0,1] in UI image space
    sx = max(0.0, min(1.0, x / max(1.0, iw)))
    sy = max(0.0, min(1.0, y / max(1.0, ih)))

    # map to 0..149 grid
    X = int(round(sx * 149))
    Y = int(round(sy * 149))
    X = max(0, min(149, X))
    Y = max(0, min(149, Y))

    ds_dir = os.path.join(DATASETS_ROOT, current_dataset)
    os.makedirs(ds_dir, exist_ok=True)
    ts = int(time.time() * 1000)
    tag = str(body.get("tag", "click"))

    fname = f"x{X:03d}_y{Y:03d}_{tag}_{ts}.jpg"
    path = os.path.join(ds_dir, fname)
    cv2.imwrite(path, frame)
    logger.info("Saved click-labeled frame: %s", path)

    return ok(
        {
            "file": os.path.relpath(path, ROOT),
            "X": X,
            "Y": Y,
            "dataset": current_dataset,
        }
    )


# ============================================================
# [9] Logging (teleop recording using filenames)
# ============================================================


# Teleop logger state
log_state = {
    "thread": None,
    "running": False,
    "rate_hz": 5.0,
    "tag": "auto",
    "dataset": None,
}


def _teleop_log_loop():
    """
    Periodically capture frames + current motor outputs and save them as:

      t_l{L:03d}_r{R:03d}_{tag}_{ts}.jpg

    L,R are 0..255 (quantized left/right in [0,1]).
    """
    logger.info("teleop logger: starting")
    rate = float(log_state["rate_hz"])
    tag = log_state["tag"]
    ds_name = log_state["dataset"] or current_dataset

    ds_dir = os.path.join(DATASETS_ROOT, ds_name)
    os.makedirs(ds_dir, exist_ok=True)

    dt = 1.0 / max(0.1, rate)

    while log_state["running"]:
        # grab raw frame
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        # read smoothed logical motor outputs from control loop
        with control_lock:
            left = max(0.0, min(1.0, float(control_state["left_actual"])))
            right = max(0.0, min(1.0, float(control_state["right_actual"])))

        # quantize to 0..255 for filename
        L = int(round(left * 255.0))
        R = int(round(right * 255.0))
        L = max(0, min(255, L))
        R = max(0, min(255, R))

        ts = int(time.time() * 1000)
        fname = f"t_l{L:03d}_r{R:03d}_{tag}_{ts}.jpg"
        path = os.path.join(ds_dir, fname)
        cv2.imwrite(path, frame)

        time.sleep(dt)

    logger.info("teleop logger: stopped")


@app.route("/api/log/start", methods=["POST"])
def api_log_start():
    """
    Start teleop logging.

    Body:
      {
        "rate_hz": 5.0,          # optional
        "tag": "auto",           # optional (goes into filename)
        "dataset": "teleop01"    # optional (defaults to current_dataset)
      }
    """
    body = request.get_json(silent=True) or {}
    if log_state["running"]:
        return err("logging already running", 409)

    rate_hz = float(body.get("rate_hz", 5.0))
    tag = str(body.get("tag", "auto"))
    dataset = (body.get("dataset") or current_dataset).strip() or current_dataset

    log_state["rate_hz"] = max(0.5, min(30.0, rate_hz))
    log_state["tag"] = tag
    log_state["dataset"] = dataset
    log_state["running"] = True

    t = threading.Thread(target=_teleop_log_loop, daemon=True)
    log_state["thread"] = t
    t.start()

    logger.info(
        "teleop logger: started (dataset=%s, rate=%.1f Hz, tag=%s)",
        dataset,
        log_state["rate_hz"],
        tag,
    )

    return ok({"rate_hz": log_state["rate_hz"], "tag": tag, "dataset": dataset})


@app.route("/api/log/stop", methods=["POST"])
def api_log_stop():
    if not log_state["running"]:
        return ok({"running": False})
    log_state["running"] = False
    logger.info("teleop logger: stop requested")
    return ok({"running": False})


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
# [11+12] Train + Deploy (autopilot using Torch model)
# ============================================================

_CLICK_RE = re.compile(r"x(\d{3})_y(\d{3})_")
_TELEOP_RE = re.compile(r"t_l(\d{3})_r(\d{3})_")


def _filename_to_lr(fname: str):
    """
    Parse left,right ∈ [0,1] from filename.

    Supports:
      xXXX_yYYY_...jpg  (click)
      t_lLLL_rRRR_...jpg (teleop)
    """
    m = _CLICK_RE.search(fname)
    if m:
        X = int(m.group(1))
        Y = int(m.group(2))
        sx = X / 149.0
        sy = Y / 149.0

        # click → forward/steer
        steer = (sx - 0.5) * 2.0        # [-1,1]
        forward = max(0.0, min(1.0, 1.0 - sy))  # [0,1]

        left = forward * (1.0 - steer)
        right = forward * (1.0 + steer)
        maxv = max(left, right, 1e-6)
        if maxv > 1.0:
            left /= maxv
            right /= maxv
        left = max(0.0, min(1.0, left))
        right = max(0.0, min(1.0, right))
        return left, right

    m = _TELEOP_RE.search(fname)
    if m:
        L = int(m.group(1))
        R = int(m.group(2))
        left = max(0.0, min(1.0, L / 255.0))
        right = max(0.0, min(1.0, R / 255.0))
        return left, right

    raise ValueError(f"Unrecognized label pattern: {fname}")


@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    global train_status

    if not TORCH_AVAILABLE or TinySteerNet is None:
        return err("PyTorch not available on server (training disabled)", 500)

    if train_status["running"]:
        return err("training already running", 409)

    from PIL import Image

    body = request.get_json(silent=True) or {}
    dataset = body.get("dataset", current_dataset)
    epochs = int(body.get("epochs", train_cfg.get("default_epochs", 10)))
    lr = float(body.get("learning_rate", train_cfg.get("default_learning_rate", 1e-3)))
    batch_size = int(body.get("batch_size", train_cfg.get("default_batch_size", 32)))

    w = int(body.get("image_width", train_cfg.get("image_width", 96)))
    h = int(body.get("image_height", train_cfg.get("image_height", 64)))
    image_size = (w, h)

    class FilenameDataset(Dataset):
        def __init__(self, ds_name: str, image_size=(96, 64)):
            self.ds_name = ds_name
            self.image_size = image_size
            ds_dir = _ensure_dataset(ds_name)
            files = sorted(
                f for f in os.listdir(ds_dir) if f.lower().endswith(".jpg")
            )
            self.samples = []
            for fname in files:
                try:
                    left, right = _filename_to_lr(fname)
                except ValueError:
                    continue
                full = os.path.join(ds_dir, fname)
                self.samples.append((full, left, right))
            if not self.samples:
                raise RuntimeError(f"No labeled samples in dataset {ds_name}")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, left, right = self.samples[idx]
            img = Image.open(path).convert("RGB")
            img = img.resize(self.image_size, Image.BILINEAR)
            x = np.asarray(img, dtype=np.float32) / 255.0
            x = np.transpose(x, (2, 0, 1))
            x = torch.from_numpy(x)
            y = torch.tensor([left, right], dtype=torch.float32)
            return x, y

    def _run_train():
        global train_status
        train_status = {
            "running": True,
            "epoch": 0,
            "epochs": epochs,
            "loss": None,
            "note": f"training on {dataset}",
            "dataset": dataset,
            "model_path": None,
        }
        try:
            ds = FilenameDataset(dataset, image_size=image_size)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = TinySteerNet().to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            logger.info(
                "Train: dataset=%s, samples=%d, epochs=%d, batch=%d, lr=%.4f, device=%s",
                dataset,
                len(ds),
                epochs,
                batch_size,
                lr,
                device,
            )

            for e in range(epochs):
                model.train()
                running_loss = 0.0
                n = 0
                for x, y in loader:
                    x = x.to(device)
                    y = y.to(device)
                    optimizer.zero_grad()
                    y_hat = model(x)
                    loss = criterion(y_hat, y)
                    loss.backward()
                    optimizer.step()
                    running_loss += float(loss.item()) * x.size(0)
                    n += x.size(0)

                avg = running_loss / max(1, n)
                train_status["epoch"] = e + 1
                train_status["loss"] = float(avg)
                train_status["note"] = f"epoch {e+1}/{epochs}"
                logger.info("[train] epoch %d/%d loss=%.6f", e + 1, epochs, avg)

            ts = int(time.time())
            model_fname = f"aicar_{dataset}_{ts}.pt"
            model_path = os.path.join(MODELS_ROOT, model_fname)
            latest_path = os.path.join(MODELS_ROOT, "aicar_latest.pt")
            os.makedirs(MODELS_ROOT, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "image_size": image_size}, model_path)
            torch.save({"model_state": model.state_dict(), "image_size": image_size}, latest_path)
            train_status["model_path"] = os.path.relpath(model_path, ROOT)
            train_status["note"] = "done"
            logger.info("Train: saved model to %s (and aicar_latest.pt)", model_path)
        except Exception as e:
            logger.exception("Training failed")
            train_status["note"] = f"error: {e}"
        finally:
            train_status["running"] = False

    t = threading.Thread(target=_run_train, daemon=True)
    t.start()
    return ok(
        {
            "status": "started",
            "dataset": dataset,
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
            "image_size": image_size,
        }
    )


@app.route("/api/train/status")
def api_train_status():
    return ok(train_status)


@app.route("/api/models")
def api_models():
    items = []
    if os.path.isdir(MODELS_ROOT):
        for name in sorted(os.listdir(MODELS_ROOT)):
            if not name.lower().endswith(".pt"):
                continue
            full = os.path.join(MODELS_ROOT, name)
            st = os.stat(full)
            kind = "fp32"
            lname = name.lower()
            if "int8" in lname:
                kind = "int8"
            elif "quant" in lname or "qnn" in lname:
                kind = "quantized"

            items.append(
                {
                    "name": name,
                    "size": st.st_size,
                    "mtime": st.st_mtime,
                    "kind": kind,
                }
            )

    return ok(
        {
            "models": items,
            "default": autopilot_cfg.get("default_model", "aicar_latest.pt"),
            "current": deploy_status.get("model"),
        }
    )


def load_policy_model(model_name: str = None):
    """
    Load trained model into globals for autopilot.
    """
    global autopilot_model, autopilot_device, autopilot_image_size

    if not TORCH_AVAILABLE or TinySteerNet is None:
        return False, "PyTorch not available"

    model_name = model_name or autopilot_cfg.get("default_model", "aicar_latest.pt")
    model_path = os.path.join(MODELS_ROOT, model_name)
    if not os.path.exists(model_path):
        return False, f"model not found: {model_name}"

    try:
        ckpt = torch.load(model_path, map_location="cpu")
        image_size = tuple(ckpt.get("image_size", autopilot_image_size))
        model = TinySteerNet()
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        autopilot_model = model
        autopilot_device = device
        autopilot_image_size = image_size

        logger.info(
            "Autopilot: loaded model %s on %s (image_size=%s)",
            model_path,
            device,
            image_size,
        )
        return True, None
    except Exception as e:
        logger.exception("Failed to load policy model")
        autopilot_model = None
        return False, str(e)


@app.route("/api/train/validate")
def api_train_validate():
    """
    Simple validation stub.

    For now:
      - counts how many .jpg files exist in the chosen dataset
      - returns basic info + a fake metric so the UI can show something

    Later:
      - replace "metrics" with real loss / MAE / etc. from your trained model.
    """
    name = request.args.get("dataset", current_dataset).strip()
    ds_dir = os.path.join(DATASETS_ROOT, name)
    if not os.path.isdir(ds_dir):
        return err(f"dataset not found: {name}", 404)

    files = sorted(
        f for f in os.listdir(ds_dir)
        if f.lower().endswith(".jpg")
    )
    n = len(files)

    # stub metrics – you will replace these with real model evaluation later
    metrics = {
        "samples": n,
        "mse": None,
        "mae": None,
        "note": "stub validation – no real model yet",
    }

    return ok({
        "dataset": name,
        "num_images": n,
        "metrics": metrics,
    })


def run_policy_on_frame(frame_bgr):
    """
    Run current policy model on a BGR frame, return (left,right) ∈ [0,1].
    """
    if not TORCH_AVAILABLE or autopilot_model is None:
        return 0.0, 0.0

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    w, h = autopilot_image_size
    img = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    x = img.astype("float32") / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, 0)

    with torch.no_grad():
        t = torch.from_numpy(x).to(autopilot_device)
        out = autopilot_model(t)
        left, right = out[0].cpu().numpy().tolist()

    left = float(max(0.0, min(1.0, left)))
    right = float(max(0.0, min(1.0, right)))
    return left, right


def _autopilot_loop():
    logger.info("[autopilot] loop started")
    if not TORCH_AVAILABLE or autopilot_model is None:
        logger.error("Autopilot: no model / torch; exiting loop")
        deploy_status["autopilot"] = False
        return

    fps = float(autopilot_cfg.get("fps", 20))
    dt = 1.0 / max(1.0, fps)

    while deploy_status["autopilot"]:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.02)
            continue

        base_left, base_right = run_policy_on_frame(frame)

        gains = deploy_status.get("gains", {})
        throttle_gain = float(gains.get("throttle_gain", 1.0))
        steering_gain = float(gains.get("steering_gain", 1.0))
        expo = float(gains.get("expo", 1.0))

        # apply throttle gain and expo per wheel
        left = max(0.0, min(1.0, base_left * throttle_gain))
        right = max(0.0, min(1.0, base_right * throttle_gain))

        expo = max(0.5, min(3.0, expo))
        left = left ** expo
        right = right ** expo

        if steering_gain != 1.0:
            mid = 0.5 * (left + right)
            dl = (left - mid) * steering_gain
            dr = (right - mid) * steering_gain
            left = max(0.0, min(1.0, mid + dl))
            right = max(0.0, min(1.0, mid + dr))

        with control_lock:
            if deploy_status.get("drive_motors", False):
                control_state["left_target"] = left
                control_state["right_target"] = right
            else:
                control_state["left_target"] = 0.0
                control_state["right_target"] = 0.0

        time.sleep(dt)

    with control_lock:
        control_state["left_target"] = 0.0
        control_state["right_target"] = 0.0
    logger.info("[autopilot] loop stopped")


@app.route("/api/deploy/start", methods=["POST"])
def api_deploy_start():
    global _autopilot_thread
    if deploy_status["autopilot"]:
        return err("autopilot already running", 409)

    body = request.get_json(silent=True) or {}
    model_name = body.get("model") or autopilot_cfg.get(
        "default_model", "aicar_latest.pt"
    )

    ok_model, msg = load_policy_model(model_name)
    if not ok_model:
        return err(f"model load failed: {msg}", 500)

    deploy_status["autopilot"] = True
    deploy_status["model"] = model_name
    _autopilot_thread = threading.Thread(target=_autopilot_loop, daemon=True)
    _autopilot_thread.start()
    return ok(deploy_status)


@app.route("/api/deploy/stop", methods=["POST"])
def api_deploy_stop():
    deploy_status["autopilot"] = False
    return ok(deploy_status)


@app.route("/api/deploy/status")
def api_deploy_status():
    return ok(deploy_status)


@app.route("/api/deploy/gains", methods=["POST"])
def api_deploy_gains():
    body = request.get_json(silent=True) or {}
    throttle_gain = float(
        body.get("throttle_gain", deploy_status["gains"]["throttle_gain"])
    )
    steering_gain = float(
        body.get("steering_gain", deploy_status["gains"]["steering_gain"])
    )
    expo = float(body.get("expo", deploy_status["gains"]["expo"]))
    drive_motors = bool(body.get("drive_motors", deploy_status["drive_motors"]))

    deploy_status["gains"].update(
        throttle_gain=throttle_gain,
        steering_gain=steering_gain,
        expo=expo,
    )
    deploy_status["drive_motors"] = drive_motors
    return ok(deploy_status)

@app.route("/api/deploy/test_frame")
def api_deploy_test_frame():
    """
    Real-time model test.

    - Requires camera to be ON.
    - Optional query: ?model=NAME to test a specific .pt file.
    - Uses TinySteerNet via run_policy_on_frame() to predict (left,right).
    """
    if not camera_on:
        return err("camera is OFF", 409)

    frame = camera.get_frame() or camera.get_raw()
    if frame is None:
        return err("no frame available", 409)

    # Decide which model to use
    model_name = (
        request.args.get("model")
        or deploy_status.get("model")
        or autopilot_cfg.get("default_model", "aicar_latest.pt")
    )

    # Load / reload model if needed
    if autopilot_model is None or model_name != deploy_status.get("model"):
        ok_model, msg = load_policy_model(model_name)
        if not ok_model:
            return err(f"model load failed: {msg}", 500)
        deploy_status["model"] = model_name

    # Run policy
    left, right = run_policy_on_frame(frame)
    h, w = frame.shape[:2]

    # For now we still use a stub click in the middle of the 150x150 grid.
    # Later you can map (left,right) -> (X,Y) if you want a "look-ahead" point.
    X = 75
    Y = 75

    return ok(
        {
            "model": deploy_status.get("model"),
            "image_width": int(w),
            "image_height": int(h),
            "click_X": int(X),
            "click_Y": int(Y),
            "left": float(left),
            "right": float(right),
            "note": "TinySteerNet inference; X/Y are still stubbed.",
        }
    )



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
