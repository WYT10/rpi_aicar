#!/usr/bin/env python3
import os
import json
import time
import glob
import logging
import threading
import atexit
from collections import deque
import re

import cv2
import numpy as np
from flask import Flask, jsonify, request, Response, render_template, send_file
from flask_cors import CORS

# --- PYTORCH SETUP ---
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torch.quantization import quantize_dynamic
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    Dataset = None
    DataLoader = None
    quantize_dynamic = None

# --- CONFIG & PATHS ---
ROOT = os.path.abspath(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(ROOT, "config.json")

def load_config(path: str = CONFIG_PATH):
    if not os.path.exists(path):
        # Default fallback
        return {
            "server": {"host": "0.0.0.0", "port": 8888},
            "camera": {"width": 224, "height": 224, "fps": 15},
            "paths": {
                "logs_root": "data/logs",
                "datasets_root": "data/datasets",
                "models_root": "data/models",
            }
        }
    with open(path, "r") as f:
        return json.load(f)

CONFIG = load_config()
train_cfg = CONFIG.get("train", {})
autopilot_cfg = CONFIG.get("autopilot", {})

DATA_ROOT = os.path.join(ROOT, "data")
LOGS_ROOT = os.path.join(ROOT, CONFIG.get("paths", {}).get("logs_root", "data/logs"))
DATASETS_ROOT = os.path.join(ROOT, CONFIG.get("paths", {}).get("datasets_root", "data/datasets"))
MODELS_ROOT = os.path.join(ROOT, CONFIG.get("paths", {}).get("models_root", "data/models"))

for d in (DATA_ROOT, LOGS_ROOT, DATASETS_ROOT, MODELS_ROOT):
    os.makedirs(d, exist_ok=True)

# --- MODEL DEFINITION ---
if TORCH_AVAILABLE:
    class TinySteerNet(nn.Module):
        def __init__(self, image_height=224, image_width=224):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 16, 5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 48, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )
            # Compute dynamic flatten size
            with torch.no_grad():
                dummy = torch.zeros(1, 3, image_height, image_width)
                z = self.conv(dummy)
                self.flat_dim = z.view(1, -1).shape[1]

            self.fc = nn.Sequential(
                nn.Linear(self.flat_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 2),
                nn.Sigmoid(),
            )

        def forward(self, x):
            z = self.conv(x)
            z = z.view(z.size(0), -1)
            return self.fc(z)
else:
    TinySteerNet = None

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("aicar")

# --- CAMERA ---
class CameraManager:
    def __init__(self, width, height, fps):
        self.width = width
        self.height = height
        self.fps = fps
        self._cap = None
        self._frame_proc = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self.backend_name = "none"

        # Settings
        self.vflip = False
        self.hflip = False
        self.gray = False
        self.mode = "raw"
        self.gamma = 1.0

    def start(self):
        if self._running: return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._cap: self._cap.release()
        self._cap = None

    def _loop(self):
        # Simple OpenCV fallback
        self._cap = cv2.VideoCapture(0) 
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.backend_name = "opencv:0"
        
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # Processing
            if self.vflip: frame = cv2.flip(frame, 0)
            if self.hflip: frame = cv2.flip(frame, 1)
            
            # Gamma
            if abs(self.gamma - 1.0) > 0.01:
                inv = 1.0 / self.gamma
                table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
                frame = cv2.LUT(frame, table)

            if self.mode == "gray" or self.gray:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            elif self.mode == "edges":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 80, 160)
                frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            with self._lock:
                self._frame_proc = frame
            
            time.sleep(1.0/self.fps)

    def get_frame(self):
        with self._lock:
            if self._frame_proc is None: return None
            return self._frame_proc.copy()
    
    def get_fps(self): return self.fps

camera = CameraManager(
    CONFIG["camera"].get("width", 224),
    CONFIG["camera"].get("height", 224),
    CONFIG["camera"].get("fps", 15)
)

# --- MOTORS ---
class MotorController:
    # Dummy implementation for brevity - replace with full Pigpio logic if needed
    def __init__(self, cfg):
        self.backend_name = "dummy"
    def open_loop(self, l, r): pass
    def stop(self): pass
    def shutdown(self): pass

motors = MotorController(CONFIG.get("motor", {}))

# --- FLASK ---
app = Flask(__name__, template_folder="templates")
CORS(app)

def ok(data=None): return jsonify({"ok": True, "data": data})
def err(msg, code=400): return jsonify({"ok": False, "error": str(msg)}), code

# Add 404 handler to return JSON instead of HTML
@app.errorhandler(404)
def page_not_found(e):
    return err("Endpoint not found (404)", 404)

@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/health")
def api_health():
    return ok({
        "camera_on": True, # Simplified
        "camera_fps": camera.get_fps(),
        "camera_backend": camera.backend_name,
        "time": time.time()
    })

@app.route("/api/camera/start", methods=["POST"])
def cam_start():
    camera.start()
    return ok({"started": True})

@app.route("/api/camera/stop", methods=["POST"])
def cam_stop():
    camera.stop()
    return ok({"started": False})

@app.route("/api/video")
def video_feed():
    def gen():
        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/camera/frame")
def single_frame():
    frame = camera.get_frame()
    if frame is None: return err("no frame")
    ret, jpeg = cv2.imencode('.jpg', frame)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

# --- DATASETS ---
@app.route("/api/datasets")
def list_datasets():
    ds = [d for d in os.listdir(DATASETS_ROOT) if os.path.isdir(os.path.join(DATASETS_ROOT, d))]
    return ok({"datasets": ds, "current": "default"})

@app.route("/api/datasets/select", methods=["POST"])
def select_dataset():
    # Stub for selection
    return ok({"current": request.json.get("name", "default")})

@app.route("/api/datasets/summary")
def ds_summary():
    name = request.args.get("name", "default")
    path = os.path.join(DATASETS_ROOT, name)
    if not os.path.exists(path): return err("Dataset not found", 404)
    files = glob.glob(os.path.join(path, "*.jpg"))
    return ok({"name": name, "count": len(files)})

@app.route("/api/data/label_click", methods=["POST"])
def label_click():
    # Stub for saving label
    body = request.json
    # In real code: save image to DATASETS_ROOT/current_dataset
    return ok({"X": 75, "Y": 75}) # Dummy return

# --- TRAINING ---
train_status = {"running": False, "epoch": 0, "loss": 0}

@app.route("/api/train/start", methods=["POST"])
def train_start():
    if not TORCH_AVAILABLE: return err("Torch not installed", 500)
    # Stub: start thread
    return ok({"status": "started"})

@app.route("/api/train/status")
def get_train_status():
    return ok(train_status)

@app.route("/api/train/validate")
def validate():
    name = request.args.get("dataset", "default")
    path = os.path.join(DATASETS_ROOT, name)
    count = len(glob.glob(os.path.join(path, "*.jpg")))
    return ok({
        "dataset": name,
        "num_images": count,
        "metrics": {"note": "Validation stub", "samples": count}
    })

# --- MODELS & DEPLOY ---
deploy_status = {
    "autopilot": False,
    "model": None,
    "gains": {"throttle": 1.0, "steering": 1.0},
    "drive_motors": False
}
autopilot_model = None

@app.route("/api/models")
def list_models():
    # Ensure directory exists
    os.makedirs(MODELS_ROOT, exist_ok=True)
    models = []
    for f in os.listdir(MODELS_ROOT):
        if f.endswith(".pt"):
            st = os.stat(os.path.join(MODELS_ROOT, f))
            models.append({"name": f, "size_kb": st.st_size // 1024, "kind": "fp32"})
    return ok({"items": models})

@app.route("/api/deploy/test_frame")
def test_frame():
    # 1. Check Camera
    frame = camera.get_frame()
    if frame is None: return err("Camera OFF or no frame", 409)

    # 2. Check Torch
    if not TORCH_AVAILABLE: return err("PyTorch not installed", 500)

    # 3. Dummy Inference (since we might not have a model loaded)
    # In real code: load model, run inference
    # For now, return dummy center coordinates so UI doesn't error
    h, w = frame.shape[:2]
    
    return ok({
        "model": "dummy",
        "image_width": w,
        "image_height": h,
        "click_X": 75, # Center of 150 grid
        "click_Y": 75, # Center of 150 grid
        "left": 0.0,
        "right": 0.0,
        "note": "Dummy inference response (fix load_policy_model to work)"
    })

@app.route("/api/deploy/start", methods=["POST"])
def deploy_start():
    deploy_status["autopilot"] = True
    return ok(deploy_status)

@app.route("/api/deploy/stop", methods=["POST"])
def deploy_stop():
    deploy_status["autopilot"] = False
    return ok(deploy_status)

@app.route("/api/deploy/status")
def get_deploy_status():
    return ok(deploy_status)

@app.route("/api/deploy/gains", methods=["POST"])
def set_gains():
    return ok(deploy_status)

# --- FS API (for file editor) ---
@app.route("/api/fs/list")
def fs_list():
    path = request.args.get("path", ".")
    abs_path = os.path.abspath(os.path.join(ROOT, path))
    if not abs_path.startswith(ROOT): return err("Access denied")
    
    entries = []
    if os.path.isdir(abs_path):
        for f in os.listdir(abs_path):
            fp = os.path.join(abs_path, f)
            entries.append({"name": f, "is_dir": os.path.isdir(fp), "size": os.path.getsize(fp)})
    return ok({"entries": entries})

@app.route("/api/fs/read")
def fs_read():
    path = request.args.get("path", "")
    abs_path = os.path.abspath(os.path.join(ROOT, path))
    if not os.path.exists(abs_path): return err("File not found")
    with open(abs_path, 'r', errors='ignore') as f:
        return ok({"path": path, "content": f.read()})

@app.route("/api/fs/raw")
def fs_raw():
    path = request.args.get("path", "")
    abs_path = os.path.abspath(os.path.join(ROOT, path))
    return send_file(abs_path)

if __name__ == "__main__":
    print("AI CAR SERVER: V3 (224x224) STARTED")
    app.run(host="0.0.0.0", port=8888, threaded=True)