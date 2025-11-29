#!/usr/bin/env python3
import os
import json
import time
import glob
import logging
import threading
import atexit
import base64
import re
from collections import deque

import cv2
import numpy as np
from flask import Flask, jsonify, request, Response, render_template, send_file
from flask_cors import CORS

# --- Torch / ML Imports ---
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torch.quantization import quantize_dynamic
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    Dataset = object
    DataLoader = None
    quantize_dynamic = None
    Image = None

# --- Configuration & Setup ---
ROOT = os.path.abspath(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(ROOT, "config.json")

def load_config():
    defaults = {
        "server": {"host": "0.0.0.0", "port": 8888},
        "camera": {"width": 224, "height": 224, "fps": 20},
        "motor": {"pins": {}, "invert_left": False, "invert_right": False, "pwm_freq": 1000, "max_duty_pct": 100, "deadzone": 0.05},
        "paths": {"logs_root": "data/logs", "datasets_root": "data/datasets", "models_root": "data/models"},
        "train": {"image_width": 224, "image_height": 224, "default_epochs": 10, "default_batch_size": 32, "default_learning_rate": 0.001},
        "autopilot": {"default_model": "aicar_latest.pt", "fps": 20, "default_mode": "assist", "default_assist_blend": 0.7, "reflex_on": True, "min_brightness": 18.0}
    }
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            user_cfg = json.load(f)
            # Simple recursive update could go here, but simplistic override is fine
            defaults.update(user_cfg)
    else:
        with open(CONFIG_PATH, "w") as f:
            json.dump(defaults, f, indent=2)
    return defaults

CONFIG = load_config()
DATASETS_ROOT = os.path.join(ROOT, CONFIG["paths"]["datasets_root"])
MODELS_ROOT = os.path.join(ROOT, CONFIG["paths"]["models_root"])
os.makedirs(DATASETS_ROOT, exist_ok=True)
os.makedirs(MODELS_ROOT, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("aicar")

# --- Globals ---
control_lock = threading.Lock()
control_state = {
    "human_left": 0.0, "human_right": 0.0,
    "model_left": 0.0, "model_right": 0.0,
    "left_target": 0.0, "right_target": 0.0,
    "left_actual": 0.0, "right_actual": 0.0,
    "last_update_ts": time.time()
}
control_params = {
    "hz": 100.0, "alpha": 0.35,
    "left_trim": 1.0, "right_trim": 1.0,
    "throttle_expo": 1.3
}

deploy_status = {
    "autopilot": False,
    "model": None,
    "gains": {"throttle_gain": 1.0, "steering_gain": 1.0, "expo": 1.0},
    "drive_motors": False,
    "mode": CONFIG["autopilot"]["default_mode"],
    "assist_blend": CONFIG["autopilot"]["default_assist_blend"],
    "reflex_on": CONFIG["autopilot"]["reflex_on"],
    "min_brightness": CONFIG["autopilot"]["min_brightness"],
    "note": ""
}

train_status = {
    "running": False, "epoch": 0, "epochs": 0, "loss": None, "note": "", "dataset": None
}

# --- Neural Network ---
if TORCH_AVAILABLE:
    class TinySteerNet(nn.Module):
        def __init__(self, image_height=224, image_width=224):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
                nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            )
            # Dynamic flatten size calculation
            with torch.no_grad():
                dummy = torch.zeros(1, 3, image_height, image_width)
                out = self.conv(dummy)
                self.flat_dim = out.view(1, -1).shape[1]
            
            self.fc = nn.Sequential(
                nn.Linear(self.flat_dim, 64), nn.ReLU(inplace=True),
                nn.Linear(64, 2), nn.Sigmoid()
            )

        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    autopilot_model = None
    autopilot_device = "cpu"
else:
    TinySteerNet = None
    autopilot_model = None

# --- Camera Manager ---
class CameraManager:
    def __init__(self, cfg):
        self.width = cfg["width"]
        self.height = cfg["height"]
        self.fps = cfg["fps"]
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.vflip = False
        self.hflip = False
        self.mode = "raw"

    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'): self.thread.join(timeout=1.0)

    def _loop(self):
        # 1. Picamera2
        try:
            from picamera2 import Picamera2
            cam = Picamera2()
            config = cam.create_preview_configuration(main={"size": (self.width, self.height), "format": "BGR888"})
            cam.configure(config)
            cam.start()
            logger.info("Camera: using Picamera2")
            while self.running:
                # Picamera2 returns BGR natively
                raw = cam.capture_array("main") 
                self._process_and_store(raw)
            cam.stop()
            cam.close()
            return
        except ImportError: pass
        except Exception as e: logger.warning(f"Picamera2 failed: {e}")

        # 2. OpenCV V4L2 / GStreamer fallback
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        logger.info("Camera: using OpenCV VideoCapture(0)")
        
        while self.running:
            ret, raw = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            # OpenCV returns BGR natively
            self._process_and_store(raw)
        cap.release()

    def _process_and_store(self, frame_bgr):
        # Processing expects BGR, returns BGR
        if self.vflip: frame_bgr = cv2.flip(frame_bgr, 0)
        if self.hflip: frame_bgr = cv2.flip(frame_bgr, 1)
        
        # Mode processing
        if self.mode == "gray":
            frame_bgr = cv2.cvtColor(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        elif self.mode == "edges":
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            frame_bgr = cv2.cvtColor(cv2.Canny(gray, 50, 150), cv2.COLOR_GRAY2BGR)

        with self.lock:
            self.frame = frame_bgr

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

camera = CameraManager(CONFIG["camera"])

# --- Motor Hardware ---
class MotorDriver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pins = cfg["pins"]
        self.max_duty = cfg["max_duty_pct"]
        self.backend = "dummy"
        
        # Try pigpio
        try:
            import pigpio
            self.pi = pigpio.pi()
            if self.pi.connected:
                self.backend = "pigpio"
                for p in [self.pins["left_forward"], self.pins["left_backward"], self.pins["right_forward"], self.pins["right_backward"]]:
                    self.pi.set_mode(p, pigpio.OUTPUT)
                if self.pins["left_pwm"]: 
                    self.pi.set_mode(self.pins["left_pwm"], pigpio.OUTPUT)
                    self.pi.set_PWM_frequency(self.pins["left_pwm"], cfg["pwm_freq"])
                if self.pins["right_pwm"]: 
                    self.pi.set_mode(self.pins["right_pwm"], pigpio.OUTPUT)
                    self.pi.set_PWM_frequency(self.pins["right_pwm"], cfg["pwm_freq"])
                if self.pins["stby"]:
                    self.pi.set_mode(self.pins["stby"], pigpio.OUTPUT)
                    self.pi.write(self.pins["stby"], 1)
                logger.info("Motors: Pigpio connected")
                return
        except ImportError: pass

        # Fallback RPi.GPIO
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            self.GPIO = GPIO
            self.backend = "gpio"
            self.lpwm = None
            self.rpwm = None
            # Setup pins... (Simplified for brevity, standard H-Bridge logic)
            logger.info("Motors: RPi.GPIO connected")
            return
        except: pass
        
        logger.info("Motors: Dummy mode")

    def drive(self, left, right):
        # Input: 0.0 to 1.0
        if self.backend == "pigpio":
            self._drive_pigpio(left, self.pins["left_forward"], self.pins["left_backward"], self.pins["left_pwm"])
            self._drive_pigpio(right, self.pins["right_forward"], self.pins["right_backward"], self.pins["right_pwm"])
        # (RPi.GPIO implementation would go here)

    def _drive_pigpio(self, val, fwd, bwd, pwm):
        duty = int(val * 255 * (self.max_duty/100.0))
        if val > 0.01:
            self.pi.write(fwd, 1)
            self.pi.write(bwd, 0)
            if pwm: self.pi.set_PWM_dutycycle(pwm, duty)
        else:
            self.pi.write(fwd, 0)
            self.pi.write(bwd, 0)
            if pwm: self.pi.set_PWM_dutycycle(pwm, 0)

    def stop(self):
        self.drive(0, 0)

motors = MotorDriver(CONFIG["motor"])

# --- Control Loop ---
def motor_control_thread():
    while True:
        with control_lock:
            # 1. Smooth Targets
            alpha = control_params["alpha"]
            target_l = control_state["left_target"]
            target_r = control_state["right_target"]
            
            # Simple exponential moving average
            curr_l = control_state["left_actual"]
            curr_r = control_state["right_actual"]
            
            next_l = curr_l + alpha * (target_l - curr_l)
            next_r = curr_r + alpha * (target_r - curr_r)
            
            control_state["left_actual"] = next_l
            control_state["right_actual"] = next_r
            
            # 2. Apply Expo & Trim
            expo = control_params["throttle_expo"]
            final_l = (next_l ** expo) * control_params["left_trim"]
            final_r = (next_r ** expo) * control_params["right_trim"]
            
            # 3. Safety Timeout (0.5s)
            if time.time() - control_state["last_update_ts"] > 0.5:
                final_l, final_r = 0, 0
                
        # 4. Hardware Execute
        motors.drive(max(0, min(1, final_l)), max(0, min(1, final_r)))
        
        time.sleep(1.0 / control_params["hz"])

threading.Thread(target=motor_control_thread, daemon=True).start()

# --- Helpers ---
def _load_model_internal(name):
    global autopilot_model, autopilot_device
    if not TORCH_AVAILABLE: return False, "No Torch"
    path = os.path.join(MODELS_ROOT, name)
    if not os.path.exists(path): return False, "File not found"
    
    try:
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model_state", ckpt) # Handle legacy saves
        
        # Create model structure
        model = TinySteerNet(image_height=CONFIG["train"]["image_height"], image_width=CONFIG["train"]["image_width"])
        
        # Check for quantization
        dtype = ckpt.get("dtype", "float32") if isinstance(ckpt, dict) else "float32"
        if dtype == "int8_dynamic" and quantize_dynamic:
            model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
            
        model.load_state_dict(state)
        model.eval()
        
        autopilot_device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(autopilot_device)
        autopilot_model = model
        deploy_status["model"] = name
        return True, "Loaded"
    except Exception as e:
        logger.error(f"Load error: {e}")
        return False, str(e)

def _predict_frame(frame_bgr):
    # PRE-PROCESSING PIPELINE:
    # 1. Resize to Config Dimension
    # 2. BGR -> RGB
    # 3. Normalize 0-1
    # 4. CHW Layout
    if autopilot_model is None: return 0.0, 0.0
    
    h, w = CONFIG["train"]["image_height"], CONFIG["train"]["image_width"]
    resized = cv2.resize(frame_bgr, (w, h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    tensor = torch.from_numpy(rgb).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(autopilot_device)
    
    with torch.no_grad():
        out = autopilot_model(tensor)
        left, right = out[0].tolist()
        
    return float(left), float(right)

# --- Flask App ---
app = Flask(__name__, template_folder=".")
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/health")
def api_health():
    return ok({
        "camera_on": camera.running,
        "camera_backend": camera._backend if camera.running else "none",
        "cpu_temp": 0 # Placeholder
    })

# --- Camera Routes ---
@app.route("/api/camera/start", methods=["POST"])
def cam_start():
    camera.start()
    return ok()

@app.route("/api/camera/stop", methods=["POST"])
def cam_stop():
    camera.stop()
    return ok()

@app.route("/api/camera/config", methods=["POST"])
def cam_config():
    data = request.json
    camera.fps = int(data.get("fps", 20))
    # Restart to apply FPS changes if needed in future
    return ok()

@app.route("/api/camera/settings", methods=["POST"])
def cam_settings():
    d = request.json
    camera.vflip = d.get("vflip", False)
    camera.hflip = d.get("hflip", False)
    camera.mode = d.get("mode", "raw")
    return ok()

@app.route("/api/video")
def video_feed():
    def gen():
        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            # Encode BGR -> JPEG for Browser
            ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(1.0/camera.fps)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Motor/Control Routes ---
@app.route("/api/motors/openloop", methods=["POST"])
def motors_open():
    d = request.json
    with control_lock:
        control_state["human_left"] = float(d.get("left", 0))
        control_state["human_right"] = float(d.get("right", 0))
        
        # If not autopilot, pass through immediately
        if not deploy_status["autopilot"]:
            control_state["left_target"] = control_state["human_left"]
            control_state["right_target"] = control_state["human_right"]
            control_state["last_update_ts"] = time.time()
    return ok()

@app.route("/api/motors/stop", methods=["POST"])
def motors_stop():
    with control_lock:
        control_state["left_target"] = 0
        control_state["right_target"] = 0
        control_state["human_left"] = 0
        control_state["human_right"] = 0
        deploy_status["autopilot"] = False # Safety kill
    return ok()

@app.route("/api/control/params", methods=["POST"])
def set_params():
    d = request.json
    control_params.update(d)
    return ok()

# --- Dataset Routes ---
@app.route("/api/datasets")
def list_datasets():
    ds = [d for d in os.listdir(DATASETS_ROOT) if os.path.isdir(os.path.join(DATASETS_ROOT, d))]
    return ok({"datasets": ds, "current": current_dataset})

@app.route("/api/datasets/select", methods=["POST"])
def select_dataset():
    global current_dataset
    name = request.json.get("name")
    if name:
        current_dataset = name
        os.makedirs(os.path.join(DATASETS_ROOT, name), exist_ok=True)
    return ok()

@app.route("/api/datasets/summary")
def dataset_summary():
    name = request.args.get("name", current_dataset)
    path = os.path.join(DATASETS_ROOT, name)
    count = len(glob.glob(os.path.join(path, "*.jpg")))
    return ok({"name": name, "count": count})

@app.route("/api/data/label_click", methods=["POST"])
def label_click():
    d = request.json
    frame = camera.get_frame()
    if frame is None: return err("No camera")
    
    # Save Image
    ts = int(time.time() * 1000)
    # Calculate grid X,Y (0-149)
    x_click = d.get("x", 0)
    y_click = d.get("y", 0)
    width = d.get("image_width", 1)
    height = d.get("image_height", 1)
    
    grid_x = int((x_click / width) * 149)
    grid_y = int((y_click / height) * 149)
    grid_x = max(0, min(149, grid_x))
    grid_y = max(0, min(149, grid_y))
    
    fname = f"x{grid_x:03d}_y{grid_y:03d}_click_{ts}.jpg"
    cv2.imwrite(os.path.join(DATASETS_ROOT, current_dataset, fname), frame)
    return ok({"saved": fname})

# --- Logging (Teleop) ---
logging_active = False
def logging_thread_func():
    global logging_active
    while logging_active:
        frame = camera.get_frame()
        if frame is not None:
            with control_lock:
                l = control_state["left_actual"]
                r = control_state["right_actual"]
            
            # Map 0.0-1.0 to 0-255 for filename
            l_int = int(l * 255)
            r_int = int(r * 255)
            fname = f"t_l{l_int:03d}_r{r_int:03d}_{int(time.time()*1000)}.jpg"
            cv2.imwrite(os.path.join(DATASETS_ROOT, current_dataset, fname), frame)
        
        time.sleep(1.0 / 10.0) # 10 Hz logging

@app.route("/api/log/start", methods=["POST"])
def log_start():
    global logging_active
    if not logging_active:
        logging_active = True
        threading.Thread(target=logging_thread_func, daemon=True).start()
    return ok()

@app.route("/api/log/stop", methods=["POST"])
def log_stop():
    global logging_active
    logging_active = False
    return ok()

# --- Models & Training ---
@app.route("/api/models")
def list_models():
    models = []
    for f in glob.glob(os.path.join(MODELS_ROOT, "*.pt")):
        st = os.stat(f)
        kind = "int8" if "int8" in f else "fp32"
        models.append({"name": os.path.basename(f), "size": st.st_size, "kind": kind})
    return ok({"models": models, "current": deploy_status["model"]})

@app.route("/api/models/convert", methods=["POST"])
def quantize_model():
    if not TORCH_AVAILABLE: return err("Torch required")
    src = request.json.get("source")
    if not src: return err("No source")
    
    path = os.path.join(MODELS_ROOT, src)
    if not os.path.exists(path): return err("Missing file")
    
    # Load FP32
    model = TinySteerNet(CONFIG["train"]["image_height"], CONFIG["train"]["image_width"])
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()
    
    # Quantize
    q_model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    
    # Save
    out_name = src.replace(".pt", "_int8.pt")
    torch.save({"model_state": q_model.state_dict(), "dtype": "int8_dynamic"}, os.path.join(MODELS_ROOT, out_name))
    
    return ok({"created": out_name})

@app.route("/api/train/start", methods=["POST"])
def train_start():
    if not TORCH_AVAILABLE: return err("No Torch")
    d = request.json
    ds_name = d.get("dataset", current_dataset)
    epochs = int(d.get("epochs", 10))
    
    def train_worker():
        train_status["running"] = True
        train_status["dataset"] = ds_name
        
        # Load Data
        images = []
        labels = []
        path = os.path.join(DATASETS_ROOT, ds_name)
        
        files = glob.glob(os.path.join(path, "*.jpg"))
        if not files:
            train_status["running"] = False
            train_status["note"] = "No images found"
            return

        # Simple In-Memory Dataset for speed on Pi (assuming < 10k images)
        # Parse filenames
        click_re = re.compile(r"x(\d+)_y(\d+)_")
        tele_re = re.compile(r"t_l(\d+)_r(\d+)_")
        
        valid_files = []
        
        logger.info(f"Training: Scanning {len(files)} files...")
        
        for f in files:
            bn = os.path.basename(f)
            l, r = 0.0, 0.0
            
            mc = click_re.search(bn)
            mt = tele_re.search(bn)
            
            if mc:
                # Convert Grid X,Y to L,R
                gx, gy = int(mc.group(1)), int(mc.group(2))
                sx, sy = gx/149.0, gy/149.0
                # Mapping logic:
                # sy is position from top (0) to bottom (1). 0=Fast, 1=Stop.
                throttle = 1.0 - sy 
                steering = (sx - 0.5) * 2.0 # -1 to 1
                
                l = throttle * (1.0 - max(0, -steering)) # Simple differential
                r = throttle * (1.0 - max(0, steering))
            elif mt:
                l = int(mt.group(1)) / 255.0
                r = int(mt.group(2)) / 255.0
            else:
                continue
                
            valid_files.append((f, l, r))

        # Dataset Class
        class PiDataset(Dataset):
            def __init__(self, data): self.data = data
            def __len__(self): return len(self.data)
            def __getitem__(self, idx):
                fpath, l, r = self.data[idx]
                # Read BGR -> RGB -> Tensor
                img = cv2.imread(fpath)
                if img is None: return torch.zeros(3,224,224), torch.tensor([0.0,0.0])
                img = cv2.resize(img, (CONFIG["train"]["image_width"], CONFIG["train"]["image_height"]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(img).permute(2,0,1).float() / 255.0
                return t, torch.tensor([l, r], dtype=torch.float32)

        dataset = PiDataset(valid_files)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TinySteerNet(CONFIG["train"]["image_height"], CONFIG["train"]["image_width"]).to(device)
        opt = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        
        train_status["epochs"] = epochs
        
        for e in range(epochs):
            model.train()
            total_loss = 0
            for imgs, targets in loader:
                imgs, targets = imgs.to(device), targets.to(device)
                opt.zero_grad()
                preds = model(imgs)
                loss = loss_fn(preds, targets)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            train_status["epoch"] = e + 1
            train_status["loss"] = avg_loss
            train_status["note"] = f"Epoch {e+1}/{epochs} Loss: {avg_loss:.4f}"
            logger.info(train_status["note"])
            
        # Save
        save_path = os.path.join(MODELS_ROOT, f"model_{ds_name}_{int(time.time())}.pt")
        torch.save({"model_state": model.state_dict()}, save_path)
        # Symlink/Copy as latest
        torch.save({"model_state": model.state_dict()}, os.path.join(MODELS_ROOT, "aicar_latest.pt"))
        
        train_status["running"] = False
        train_status["note"] = "Training Complete. Saved."

    threading.Thread(target=train_worker, daemon=True).start()
    return ok()

@app.route("/api/train/status")
def get_train_status():
    return ok(train_status)

@app.route("/api/train/validate")
def validate_model():
    if not TORCH_AVAILABLE or autopilot_model is None:
        return err("No model loaded")
        
    ds_name = request.args.get("dataset", current_dataset)
    # Re-use logic from training to load files... (Simplified for brevity: Assume similar parsing)
    path = os.path.join(DATASETS_ROOT, ds_name)
    files = glob.glob(os.path.join(path, "*.jpg"))
    if not files: return ok({"metrics": {"note": "Empty dataset"}})
    
    click_re = re.compile(r"x(\d+)_y(\d+)_")
    tele_re = re.compile(r"t_l(\d+)_r(\d+)_")
    
    total_mae = 0
    total_mse = 0
    count = 0
    
    model = autopilot_model
    device = autopilot_device
    
    # Process small batch for speed
    for f in files[:200]: # Limit validation to 200 random images for speed on Pi
        img = cv2.imread(f)
        if img is None: continue
        
        # Parse Truth
        bn = os.path.basename(f)
        mc = click_re.search(bn)
        mt = tele_re.search(bn)
        
        true_l, true_r = 0.0, 0.0
        if mc:
            gx, gy = int(mc.group(1)), int(mc.group(2))
            sx, sy = gx/149.0, gy/149.0
            th = 1.0 - sy
            st = (sx - 0.5)*2.0
            true_l = th*(1.0 - max(0, -st))
            true_r = th*(1.0 - max(0, st))
        elif mt:
            true_l = int(mt.group(1))/255.0
            true_r = int(mt.group(2))/255.0
        else: continue
            
        # Predict
        # Use existing prediction pipeline (handles resizing/color)
        pred_l, pred_r = _predict_frame(img)
        
        total_mae += (abs(true_l - pred_l) + abs(true_r - pred_r)) / 2.0
        total_mse += ((true_l - pred_l)**2 + (true_r - pred_r)**2) / 2.0
        count += 1
        
    if count == 0: return ok({"metrics": {"note": "No valid labels"}})
    
    return ok({
        "dataset": ds_name,
        "num_images": count,
        "metrics": {
            "mae": round(total_mae/count, 4),
            "mse": round(total_mse/count, 4)
        }
    })

# --- Autopilot ---
@app.route("/api/deploy/start", methods=["POST"])
def ap_start():
    d = request.json
    model_name = d.get("model")
    if model_name:
        success, msg = _load_model_internal(model_name)
        if not success: return err(msg)
    elif autopilot_model is None:
        return err("No model selected")
        
    deploy_status["autopilot"] = True
    
    def drive_loop():
        while deploy_status["autopilot"]:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            
            # 1. Reflex Safety (Brightness check)
            if deploy_status["reflex_on"]:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if np.mean(gray) < deploy_status["min_brightness"]:
                    deploy_status["note"] = "Darkness stop"
                    with control_lock:
                        control_state["left_target"] = 0
                        control_state["right_target"] = 0
                        control_state["last_update_ts"] = time.time()
                    time.sleep(0.1)
                    continue
                else:
                    deploy_status["note"] = ""

            # 2. Inference
            l, r = _predict_frame(frame)
            
            # 3. Mixing
            mode = deploy_status["mode"]
            with control_lock:
                hl = control_state["human_left"]
                hr = control_state["human_right"]
                
                final_l, final_r = 0, 0
                if mode == "auto":
                    final_l, final_r = l, r
                elif mode == "assist":
                    blend = deploy_status["assist_blend"]
                    final_l = (1-blend)*hl + blend*l
                    final_r = (1-blend)*hr + blend*r
                
                # Gains
                g = deploy_status["gains"]
                final_l *= g["throttle_gain"]
                final_r *= g["throttle_gain"]
                
                # Apply to Control State if driving enabled
                if deploy_status["drive_motors"]:
                    control_state["left_target"] = final_l
                    control_state["right_target"] = final_r
                    control_state["last_update_ts"] = time.time()
            
            time.sleep(1.0/CONFIG["autopilot"]["fps"])
            
    threading.Thread(target=drive_loop, daemon=True).start()
    return ok()

@app.route("/api/deploy/stop", methods=["POST"])
def ap_stop():
    deploy_status["autopilot"] = False
    return ok()

@app.route("/api/deploy/gains", methods=["POST"])
def ap_gains():
    d = request.json
    deploy_status["gains"]["throttle_gain"] = float(d.get("throttle_gain", 1.0))
    deploy_status["gains"]["steering_gain"] = float(d.get("steering_gain", 1.0))
    deploy_status["gains"]["expo"] = float(d.get("expo", 1.0))
    deploy_status["drive_motors"] = d.get("drive_motors", False)
    deploy_status["mode"] = d.get("mode", "assist")
    deploy_status["assist_blend"] = float(d.get("assist_blend", 0.7))
    deploy_status["reflex_on"] = d.get("reflex_on", True)
    return ok()

@app.route("/api/deploy/status")
def ap_status():
    return ok(deploy_status)

@app.route("/api/deploy/test_frame")
def test_frame():
    # Visual Debug Endpoint
    if not camera.running: return err("Camera off")
    
    # Ensure model loaded
    target_model = request.args.get("model")
    if target_model and target_model != deploy_status["model"]:
        _load_model_internal(target_model)
    if autopilot_model is None: return err("No model loaded")
    
    frame = camera.get_frame()
    l, r = _predict_frame(frame)
    
    # Calc Grid Position for visualization
    # Reverse mapping from L/R to X/Y
    throttle = (l+r)/2.0
    steering = (r-l)/(throttle+0.0001)/2.0 # approx
    
    grid_y = int((1.0 - throttle) * 149)
    grid_x = int(((steering/2.0) + 0.5) * 149)
    
    # Draw on frame
    vis = frame.copy()
    h, w, _ = vis.shape
    cx = int((grid_x/149.0)*w)
    cy = int((grid_y/149.0)*h)
    cv2.circle(vis, (cx, cy), 10, (0, 255, 0), 2) # Green Circle = Prediction
    
    ret, jpeg = cv2.imencode('.jpg', vis)
    b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
    
    return ok({
        "click_X": grid_x,
        "click_Y": grid_y,
        "left": round(l,3),
        "right": round(r,3),
        "debug_image": f"data:image/jpeg;base64,{b64}"
    })

# --- Utils ---
def ok(d=None): return jsonify({"ok": True, "data": d})
def err(msg, code=400): return jsonify({"ok": False, "error": msg}), code

if __name__ == "__main__":
    app.run(host=CONFIG["server"]["host"], port=CONFIG["server"]["port"], threaded=True)