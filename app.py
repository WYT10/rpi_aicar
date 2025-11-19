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

def load_config(path="config.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

CONFIG = load_config()

# Prepare logs directory
LOG_ROOT = CONFIG.get("paths", {}).get("logs_root", "data/logs")
os.makedirs(LOG_ROOT, exist_ok=True)
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
# Camera (hybrid backend, threaded)
# ============================================================

class CameraManager:
    """
    Hybrid camera manager:
      1. Try Picamera2 (RGB888)
      2. Try /dev/video* via OpenCV V4L2
      3. Try GStreamer libcamerasrc → BGR

    Provides:
      - start()
      - stop()
      - get_frame() -> latest BGR frame or None
      - get_fps()
      - backend_name
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
        # basic scan for /dev/video*
        import glob
        return sorted(glob.glob("/dev/video*"))

    # -------- Open/close backends --------
    def _open_backend(self):
        # Try Picamera2 first
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

        # If we reach here, no backend worked
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

    # -------- Grabbing frames --------
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

            with self._lock:
                self._frame = frame

        logger.info("Camera: capture loop exiting")

    def stop(self):
        self._running = False
        # give loop a moment to exit
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

        # Setup pins
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
        # deadzone
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
        # zero PWM and direction pins
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

        # Try pigpio
        try:
            import pigpio  # noqa: F401
            self.backend = MotorPigpio(self.cfg)
            self.backend_name = "pigpio"
            logger.info("MotorController using pigpio backend")
            return
        except Exception as e:
            logger.info(f"MotorController: pigpio unavailable: {e!r}")

        # Try RPi.GPIO
        try:
            import RPi.GPIO  # noqa: F401
            self.backend = MotorRPiGPIO(self.cfg)
            self.backend_name = "RPi.GPIO"
            logger.info("MotorController using RPi.GPIO backend")
            return
        except Exception as e:
            logger.info(f"MotorController: RPi.GPIO unavailable: {e!r}")

        # Fallback dummy
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
# Flask app + WebUI
# ============================================================

app = Flask(__name__, template_folder="templates", static_folder=None)
CORS(app)

# Singletons
cam_cfg = CONFIG.get("camera", {})
camera = CameraManager(
    width=cam_cfg.get("width", 640),
    height=cam_cfg.get("height", 480),
    fps=cam_cfg.get("fps", 20),
)

motor_cfg = CONFIG.get("motor", {})
motors = MotorController(cfg=motor_cfg)

camera_on = False  # app-level state


def ok(data=None):
    return jsonify({"ok": True, "data": data})

def err(message, code=400):
    return jsonify({"ok": False, "error": str(message)}), code


# ------------- Routes -------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def api_health():
    return ok({
        "camera_on": camera_on,
        "camera_fps": camera.get_fps(),
        "camera_backend": camera.backend_name,
        "motor_backend": motors.backend_name,
        "time": time.time(),
        "log_path": log_path,
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
    """MJPEG stream endpoint."""

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
            chunk = (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpeg.tobytes() +
                b"\r\n"
            )
            yield chunk

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/motors/openloop", methods=["POST"])
def api_motors_openloop():
    body = request.get_json(silent=True) or {}
    try:
        left = float(body.get("left", 0.0))
        right = float(body.get("right", 0.0))
    except ValueError:
        return err("invalid left/right", 400)

    motors.open_loop(left, right)
    return ok({"left": left, "right": right})


@app.route("/api/motors/stop", methods=["POST"])
def api_motors_stop():
    motors.stop()
    return ok({"stopped": True})


# ---------- Cleanup ----------

@atexit.register
def _cleanup():
    logger.info("Cleanup: shutting down motors and camera...")
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