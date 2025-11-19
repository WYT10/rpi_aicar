#!/usr/bin/env python3
import os
import json
import time
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

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_config(path="config.json"):
    cfg_path = os.path.join(ROOT_DIR, path)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        return json.load(f)

CONFIG = load_config()

paths_cfg = CONFIG.get("paths", {})
LOG_ROOT = os.path.join(ROOT_DIR, paths_cfg.get("logs_root", "data/logs"))
DATASET_ROOT = os.path.join(ROOT_DIR, paths_cfg.get("dataset_root", "data/datasets"))
MODELS_ROOT = os.path.join(ROOT_DIR, paths_cfg.get("models_root", "data/models"))
FS_ROOT = os.path.realpath(os.path.join(ROOT_DIR, paths_cfg.get("fs_root", ".")))

os.makedirs(LOG_ROOT, exist_ok=True)
os.makedirs(DATASET_ROOT, exist_ok=True)
os.makedirs(MODELS_ROOT, exist_ok=True)

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
# Camera (hybrid backend, threaded) + simple processing
# ============================================================

class ProcConfig:
    """Very simple processing config for the Image tab."""
    def __init__(self):
        self.vflip = False
        self.hflip = False
        self.gray = False

    def to_dict(self):
        return {"vflip": self.vflip, "hflip": self.hflip, "gray": self.gray}

    def update_from_dict(self, d):
        if "vflip" in d:
            self.vflip = bool(d["vflip"])
        if "hflip" in d:
            self.hflip = bool(d["hflip"])
        if "gray" in d:
            self.gray = bool(d["gray"])

PROC_CFG = ProcConfig()


def apply_processing(frame_bgr):
    """Apply simple operations based on PROC_CFG."""
    if PROC_CFG.vflip:
        frame_bgr = cv2.flip(frame_bgr, 0)
    if PROC_CFG.hflip:
        frame_bgr = cv2.flip(frame_bgr, 1)
    if PROC_CFG.gray:
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return frame_bgr


class CameraManager:
    """
    Hybrid camera manager:
      1. Try Picamera2 (RGB888)
      2. Try /dev/video* via OpenCV V4L2
      3. Try GStreamer libcamerasrc → BGR
    """

    def __init__(self, width=640, height=480, fps=20):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        self._backend = None
        self._picam2 = None
        self._cap = None

        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        self._fps_hist = deque(maxlen=60)

    @staticmethod
    def _opencv_supports_gst():
        try:
            info = cv2.getBuildInformation()
            return "GStreamer" in info and "YES" in info
        except Exception:
            return False

    @staticmethod
    def _list_video_devices():
        import glob
        return sorted(glob.glob("/dev/video*"))

    def _open_backend(self):
        # 1) Picamera2
        try:
            from picamera2 import Picamera2  # type: ignore
            cam = Picamera2()
            cfg = cam.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                buffer_count=3,
            )
            cam.configure(cfg)
            cam.start()
            self._picam2 = cam
            self._backend = "picamera2"
            logger.info("Camera: using Picamera2 backend")
            return True
        except Exception as e:
            logger.info(f"Camera: Picamera2 not available: {e!r}")
            self._picam2 = None

        # 2) V4L2
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

        # 3) GStreamer
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

        logger.info("Camera: backend closed")
        self._backend = None

    def _grab_frame_bgr(self):
        if self._backend == "picamera2" and self._picam2 is not None:
            arr = self._picam2.capture_array("main")  # RGB
            if arr is None:
                return None
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if self._cap is not None:
            ok, frame = self._cap.read()
            return frame if ok else None
        return None

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

            frame = apply_processing(frame)

            now = time.time()
            dt = now - prev
            prev = now
            if dt > 0:
                self._fps_hist.append(1.0 / dt)

            with self._lock:
                self._frame = frame

        logger.info("Camera: capture loop exiting")

    def stop(self):
        self._running = False
        time.sleep(0.05)
        self._close_backend()

    def get_frame(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

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
            raise RuntimeError("pigpio daemon not running (use: sudo pigpiod)")

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
# Flask app + Global state
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

def ok(data=None):
    return jsonify({"ok": True, "data": data})

def err(message, code=400):
    return jsonify({"ok": False, "error": str(message)}), code


# ============================================================
# File tab helpers (safe FS root)
# ============================================================

def fs_safe_join(relpath: str) -> str:
    relpath = relpath.lstrip("/").replace("\\", "/")
    full = os.path.realpath(os.path.join(FS_ROOT, relpath))
    if not full.startswith(FS_ROOT):
        raise ValueError("Path outside allowed root")
    return full


# ============================================================
# Dataset & logging helpers
# ============================================================

CURRENT_DATASET = "default"
os.makedirs(os.path.join(DATASET_ROOT, CURRENT_DATASET), exist_ok=True)

_current_lr = {"left": 0.0, "right": 0.0}
log_event = threading.Event()
log_thread = None
log_rate_hz = 5.0
log_tag = "runA"

def set_current_lr(l, r):
    _current_lr["left"] = float(l)
    _current_lr["right"] = float(r)

def map_lr_to_xy(l, r, size=255):
    """
    Simple mapping: left/right in [-1,1] -> XY in [0..size-1]^2.
    """
    l = max(-1.0, min(1.0, float(l)))
    r = max(-1.0, min(1.0, float(r)))

    sx = ((r - l + 2.0) / 4.0) * (size - 1)
    sy = ((l + r + 2.0) / 4.0) * (size - 1)
    X = int(np.clip(round(sx), 0, size - 1))
    Y = int(np.clip(round(sy), 0, size - 1))
    return X, Y

def save_labeled_frame(frame_bgr, X_label, Y_label, tag="runA"):
    ts = int(time.time() * 1000)
    fname = f"x{X_label}_y{Y_label}_{tag}_{ts}.jpg"
    dpath = os.path.join(DATASET_ROOT, CURRENT_DATASET)
    os.makedirs(dpath, exist_ok=True)
    fpath = os.path.join(dpath, fname)
    cv2.imwrite(fpath, frame_bgr)
    rel = os.path.relpath(fpath, ROOT_DIR)
    logger.info(f"Saved labeled frame: {rel}")
    return rel

def logger_loop():
    logger.info("Logger loop started")
    global log_event
    period = 1.0 / max(1e-3, log_rate_hz)
    while log_event.is_set():
        t0 = time.time()
        frame = camera.get_frame()
        if frame is not None:
            X, Y = map_lr_to_xy(_current_lr["left"], _current_lr["right"])
            save_labeled_frame(frame, X, Y, tag=log_tag)
        dt = time.time() - t0
        time.sleep(max(0.0, period - dt))
    logger.info("Logger loop stopped")


# ============================================================
# Training & deploy stubs
# ============================================================

train_status = {
    "running": False,
    "epoch": 0,
    "epochs": 0,
    "loss": None,
    "note": ""
}
train_thread = None

def train_worker(dataset_name, epochs, lr, batch_size):
    logger.info(f"Training started: dataset={dataset_name}, epochs={epochs}, lr={lr}, batch_size={batch_size}")
    train_status.update({"running": True, "epoch": 0, "epochs": epochs, "note": "starting"})
    dpath = os.path.join(DATASET_ROOT, dataset_name)
    if not os.path.isdir(dpath):
        train_status.update({"running": False, "note": f"dataset not found: {dataset_name}"})
        return

    images = [f for f in os.listdir(dpath) if f.lower().endswith(".jpg")]
    if not images:
        train_status.update({"running": False, "note": "no images in dataset"})
        return

    for e in range(1, epochs + 1):
        time.sleep(0.5)
        loss = max(0.01, 1.0 / e)
        train_status.update({
            "epoch": e,
            "loss": loss,
            "note": "training"
        })
    train_status.update({"running": False, "note": "done"})
    logger.info("Training finished")


autopilot_event = threading.Event()
autopilot_thread = None
autopilot_state = {
    "running": False,
    "X": None,
    "Y": None,
    "throttle": 0.0,
    "steering": 0.0,
    "note": "stub - no model yet"
}

def autopilot_loop():
    logger.info("Autopilot loop started")
    autopilot_state["running"] = True
    while autopilot_event.is_set():
        frame = camera.get_frame()
        if frame is not None:
            X = 127
            Y = 127
            autopilot_state.update({
                "X": X,
                "Y": Y,
                "throttle": 0.0,
                "steering": 0.0,
                "note": "stub prediction"
            })
        time.sleep(0.1)
    autopilot_state["running"] = False
    logger.info("Autopilot loop stopped")


# ============================================================
# Routes
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")

# ---------- Health ----------

@app.route("/api/health", methods=["GET"])
def api_health():
    return ok({
        "camera_on": camera_on,
        "camera_fps": camera.get_fps(),
        "camera_backend": camera.backend_name,
        "motor_backend": motors.backend_name,
        "time": time.time(),
        "log_path": log_path,
        "current_dataset": CURRENT_DATASET
    })

# ---------- Camera ----------

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

@app.route("/api/camera/frame", methods=["GET"])
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
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   jpeg.tobytes() + b"\r\n")
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/camera/settings", methods=["POST"])
def api_camera_settings():
    body = request.get_json(silent=True) or {}
    PROC_CFG.update_from_dict(body)
    return ok(PROC_CFG.to_dict())

# ---------- Motors ----------

@app.route("/api/motors/openloop", methods=["POST"])
def api_motors_openloop():
    body = request.get_json(silent=True) or {}
    try:
        left = float(body.get("left", 0.0))
        right = float(body.get("right", 0.0))
    except ValueError:
        return err("invalid left/right", 400)
    motors.open_loop(left, right)
    set_current_lr(left, right)
    return ok({"left": left, "right": right})

@app.route("/api/motors/stop", methods=["POST"])
def api_motors_stop():
    motors.stop()
    set_current_lr(0.0, 0.0)
    return ok({"stopped": True})

# ---------- Datasets & logging ----------

@app.route("/api/datasets", methods=["GET"])
def api_list_datasets():
    names = []
    if os.path.isdir(DATASET_ROOT):
        for name in sorted(os.listdir(DATASET_ROOT)):
            if os.path.isdir(os.path.join(DATASET_ROOT, name)):
                names.append(name)
    return ok({"datasets": names, "current": CURRENT_DATASET})

@app.route("/api/datasets/select", methods=["POST"])
def api_select_dataset():
    global CURRENT_DATASET
    body = request.get_json(silent=True) or {}
    name = body.get("name", "").strip()
    if not name:
        return err("dataset name required", 400)
    dpath = os.path.join(DATASET_ROOT, name)
    os.makedirs(dpath, exist_ok=True)
    CURRENT_DATASET = name
    return ok({"current": CURRENT_DATASET})

@app.route("/api/data/label_click", methods=["POST"])
def api_label_click():
    body = request.get_json(silent=True) or {}
    try:
        px = float(body["x"])
        py = float(body["y"])
        iw = float(body["image_width"])
        ih = float(body["image_height"])
    except KeyError as e:
        return err(f"missing field: {e}", 400)
    except ValueError:
        return err("invalid coordinates", 400)

    if not camera_on:
        return err("camera off", 409)
    frame = camera.get_frame()
    if frame is None:
        return err("no frame", 409)

    X_label = int(np.clip(round(px / max(iw, 1e-6) * 255), 0, 255))
    Y_label = int(np.clip(round(py / max(ih, 1e-6) * 255), 0, 255))
    tag = body.get("tag", "click")

    rel = save_labeled_frame(frame, X_label, Y_label, tag=tag)
    return ok({"file": rel, "X": X_label, "Y": Y_label})

@app.route("/api/log/start", methods=["POST"])
def api_log_start():
    global log_thread, log_event, log_rate_hz, log_tag
    body = request.get_json(silent=True) or {}
    if not camera_on:
        return err("camera off", 409)
    rate = float(body.get("rate_hz", 5.0))
    tag = str(body.get("tag", "auto"))
    log_rate_hz = max(0.5, min(60.0, rate))
    log_tag = tag

    if log_event.is_set():
        return ok({"msg": "already logging", "rate_hz": log_rate_hz, "tag": log_tag})

    log_event.set()
    log_thread = threading.Thread(target=logger_loop, daemon=True)
    log_thread.start()
    return ok({"msg": "logging started", "rate_hz": log_rate_hz, "tag": log_tag})

@app.route("/api/log/stop", methods=["POST"])
def api_log_stop():
    global log_event
    log_event.clear()
    time.sleep(0.05)
    return ok({"msg": "logging stopped"})

# ---------- Train tab ----------

@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    global train_thread, train_status
    if train_status.get("running", False):
        return err("training already running", 409)
    body = request.get_json(silent=True) or {}
    dataset = body.get("dataset", CURRENT_DATASET)
    epochs = int(body.get("epochs", 10))
    lr = float(body.get("learning_rate", 1e-3))
    batch_size = int(body.get("batch_size", 32))

    train_thread = threading.Thread(
        target=train_worker,
        args=(dataset, epochs, lr, batch_size),
        daemon=True
    )
    train_thread.start()
    return ok({"status": "started", "dataset": dataset, "epochs": epochs})

@app.route("/api/train/status", methods=["GET"])
def api_train_status():
    return ok(train_status)

# ---------- Deploy tab ----------

@app.route("/api/deploy/start", methods=["POST"])
def api_deploy_start():
    global autopilot_thread, autopilot_event
    if autopilot_event.is_set():
        return ok({"msg": "autopilot already running"})
    autopilot_event.set()
    autopilot_thread = threading.Thread(target=autopilot_loop, daemon=True)
    autopilot_thread.start()
    return ok({"msg": "autopilot started"})

@app.route("/api/deploy/stop", methods=["POST"])
def api_deploy_stop():
    global autopilot_event
    autopilot_event.clear()
    time.sleep(0.05)
    return ok({"msg": "autopilot stopped"})

@app.route("/api/deploy/status", methods=["GET"])
def api_deploy_status():
    return ok(autopilot_state)

# ---------- File tab ----------

@app.route("/api/fs/list", methods=["GET"])
def api_fs_list():
    rel = request.args.get("path", "").strip()
    try:
        full = fs_safe_join(rel)
    except ValueError as e:
        return err(str(e), 400)
    if not os.path.exists(full):
        return err("path not found", 404)

    if os.path.isfile(full):
        return ok({
            "path": rel,
            "type": "file",
            "size": os.path.getsize(full)
        })

    entries = []
    for name in sorted(os.listdir(full)):
        p = os.path.join(full, name)
        entries.append({
            "name": name,
            "is_dir": os.path.isdir(p),
            "size": os.path.getsize(p) if os.path.isfile(p) else None
        })
    return ok({"path": rel, "entries": entries})

@app.route("/api/fs/read", methods=["GET"])
def api_fs_read():
    rel = request.args.get("path", "").strip()
    try:
        full = fs_safe_join(rel)
    except ValueError as e:
        return err(str(e), 400)
    if not os.path.isfile(full):
        return err("file not found", 404)
    try:
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return err(str(e), 500)
    return ok({"path": rel, "content": content})

@app.route("/api/fs/write", methods=["POST"])
def api_fs_write():
    body = request.get_json(silent=True) or {}
    rel = body.get("path", "").strip()
    content = body.get("content", "")
    if not rel:
        return err("path required", 400)
    try:
        full = fs_safe_join(rel)
    except ValueError as e:
        return err(str(e), 400)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    try:
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        return err(str(e), 500)
    return ok({"path": rel})

@app.route("/api/fs/mkdir", methods=["POST"])
def api_fs_mkdir():
    body = request.get_json(silent=True) or {}
    rel = body.get("path", "").strip()
    if not rel:
        return err("path required", 400)
    try:
        full = fs_safe_join(rel)
    except ValueError as e:
        return err(str(e), 400)
    try:
        os.makedirs(full, exist_ok=True)
    except Exception as e:
        return err(str(e), 500)
    return ok({"path": rel})

@app.route("/api/fs/delete", methods=["POST"])
def api_fs_delete():
    body = request.get_json(silent=True) or {}
    rel = body.get("path", "").strip()
    if not rel:
        return err("path required", 400)
    try:
        full = fs_safe_join(rel)
    except ValueError as e:
        return err(str(e), 400)
    try:
        if os.path.isfile(full):
            os.remove(full)
        elif os.path.isdir(full):
            os.rmdir(full)  # non-recursive
        else:
            return err("path not found", 404)
    except OSError as e:
        return err(f"failed to delete: {e}", 400)
    except Exception as e:
        return err(str(e), 500)
    return ok({"path": rel})


# ---------- Cleanup ----------

@atexit.register
def _cleanup():
    logger.info("Cleanup: shutting down motors, camera, logger, autopilot...")
    try:
        log_event.clear()
    except Exception:
        pass
    try:
        autopilot_event.clear()
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


# ---------- Entry point ----------

if __name__ == "__main__":
    server_cfg = CONFIG.get("server", {})
    host = server_cfg.get("host", "0.0.0.0")
    port = int(server_cfg.get("port", 8888))
    logger.info(f"Starting AI Car server on {host}:{port} ...")
    app.run(host=host, port=port, threaded=True)
