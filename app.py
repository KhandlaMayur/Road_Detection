# =========================
# IMPORT LIBRARIES
# =========================
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import cv2
import numpy as np
import base64
import os
import json
import threading
import time
from datetime import datetime
from ultralytics import YOLO

app = Flask(__name__)

# =========================
# OUTPUT FOLDER
# =========================
OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =========================
# MODEL PATH
# =========================
BEST_MODEL_PATH = "runs/detect/train/weights/best.pt"

# =========================
# LOAD MODEL (FALLBACK STRATEGY)
# =========================
print("🔍 Checking for trained model...")
if os.path.exists(BEST_MODEL_PATH):
    print("⚡ Loading trained model: best.pt")
    model = YOLO(BEST_MODEL_PATH)
else:
    print("⚠️  Trained model not found. Using pre-trained YOLOv8n...")
    model = YOLO("yolov8n.pt")

# =========================
# WEBCAM STREAM STATE
# =========================
class WebcamStream:
    def __init__(self):
        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_annotated = None
        self.latest_stats = {
            "potholes": 0,
            "cracks": 0,
            "severity": "Low",
            "damage_percentage": 0.0,
            "fps": 0,
            "status": "stopped"
        }
        self.frame_count = 0
        self.start_time = time.time()
        self._thread = None

    def start(self):
        with self.lock:
            if self.running:
                return True
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                self.latest_stats["status"] = "error"
                return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.running = True
            self.latest_stats["status"] = "running"
            self.start_time = time.time()
            self.frame_count = 0
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        with self.lock:
            self.running = False
            self.latest_stats["status"] = "stopped"
        if self._thread:
            self._thread.join(timeout=3)
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.latest_frame = None
            self.latest_annotated = None

    def _capture_loop(self):
        process_every = 3  # run inference every N frames for performance
        frame_idx = 0
        last_annotated = None

        while True:
            with self.lock:
                if not self.running:
                    break
                cap = self.cap

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame_idx += 1
            self.frame_count += 1

            # Calculate FPS
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0

            if frame_idx % process_every == 0:
                # Run YOLO inference
                results = model(frame, verbose=False, conf=0.35)
                annotated = results[0].plot()
                stats = self._compute_stats(results, frame.shape, fps)
                last_annotated = annotated
                with self.lock:
                    self.latest_frame = frame.copy()
                    self.latest_annotated = annotated.copy()
                    self.latest_stats = stats
            else:
                # Use last annotated frame but update FPS
                if last_annotated is not None:
                    with self.lock:
                        if self.latest_stats:
                            self.latest_stats["fps"] = round(fps, 1)
                        self.latest_annotated = last_annotated.copy()
                else:
                    with self.lock:
                        self.latest_frame = frame.copy()
                        self.latest_annotated = frame.copy()

    def _compute_stats(self, results, image_shape, fps):
        boxes = results[0].boxes
        names = model.names
        potholes = 0
        cracks = 0
        total_area = 0
        img_area = image_shape[0] * image_shape[1]

        for box in boxes:
            cls = int(box.cls[0])
            label = names[cls]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)
            total_area += area
            if label == "pothole":
                potholes += 1
            elif label == "crack":
                cracks += 1

        damage_pct = (total_area / img_area * 100) if img_area > 0 else 0
        area_ratio = total_area / img_area if img_area > 0 else 0

        if potholes == 0 and cracks == 0:
            severity = "None"
        elif area_ratio < 0.05 and (potholes + cracks) <= 2:
            severity = "Low"
        elif area_ratio < 0.20 and (potholes + cracks) <= 5:
            severity = "Medium"
        else:
            severity = "High"

        return {
            "potholes": potholes,
            "cracks": cracks,
            "severity": severity,
            "damage_percentage": round(damage_pct, 2),
            "fps": round(fps, 1),
            "status": "running",
            "total_detections": potholes + cracks
        }

    def get_jpeg_frame(self):
        with self.lock:
            frame = self.latest_annotated
        if frame is None:
            # Return a black placeholder frame
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Starting Camera...", (160, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
            _, buf = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buf.tobytes()
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()

webcam = WebcamStream()

# =========================
# ANALYSIS FUNCTION (for image upload)
# =========================
def analyze(results, image_shape):
    boxes = results[0].boxes
    names = model.names

    potholes = 0
    cracks = 0
    total_area = 0
    img_area = image_shape[0] * image_shape[1]

    for box in boxes:
        cls = int(box.cls[0])
        label = names[cls]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1)
        total_area += area
        if label == "pothole":
            potholes += 1
        elif label == "crack":
            cracks += 1

    damage_percentage = float((total_area / img_area) * 100) if img_area > 0 else 0

    if potholes == 0 and cracks == 0:
        return {
            "potholes": 0, "cracks": 0,
            "damage_types": "None", "damage_percentage": 0,
            "severity": "Low", "risk": "Low",
            "labor": 0, "days": 0, "material_cost": 0,
            "total_cost": 0, "machine": "No",
            "machine_type": "No", "decision": "No urgent repair"
        }

    avg_damage_area = total_area / (potholes + cracks) if (potholes + cracks) > 0 else 0
    area_ratio = total_area / img_area

    if area_ratio < 0.05 and (potholes + cracks) <= 2:
        severity = "Low"; risk = "Low"
    elif area_ratio < 0.20 and (potholes + cracks) <= 5:
        severity = "Medium"; risk = "Moderate"
    else:
        severity = "High"; risk = "High"

    base_workers = 2; base_days = 1
    labor_workers = base_workers + int((potholes * 0.5) + (cracks * 0.5))
    days = base_days + int((area_ratio * 5) + (potholes * 0.5) + (cracks * 0.3))
    labor_workers = max(2, min(labor_workers, 10))
    days = max(1, days)

    if severity == "Low":
        machine = "No"; machine_type = "Manual Crack Sealing Tools"
    elif severity == "Medium":
        machine = "Yes"; machine_type = "Patching Machine + Roller"
    else:
        machine = "Yes"; machine_type = "Asphalt Paver + Roller + Compactor"

    material_rate = 0.05
    material_cost = max(int(total_area * material_rate), 200)
    labor_charge_per_day = 300
    total_labor_cost = labor_workers * labor_charge_per_day * days
    total_cost = total_labor_cost + material_cost

    if severity == "Low": decision = "No urgent repair"
    elif severity == "Medium": decision = "Schedule maintenance"
    else: decision = "Immediate repair required"

    damage_types = []
    if potholes > 0: damage_types.append("Pothole")
    if cracks > 0: damage_types.append("Crack")

    return {
        "potholes": potholes, "cracks": cracks,
        "damage_types": ", ".join(damage_types),
        "damage_percentage": round(damage_percentage, 2),
        "severity": severity, "risk": risk,
        "labor": labor_workers, "days": days,
        "material_cost": material_cost, "total_cost": total_cost,
        "machine": machine, "machine_type": machine_type,
        "decision": decision
    }

# =========================
# HOME ROUTE
# =========================
@app.route("/")
def home():
    return render_template("index.html")

# =========================
# WEBCAM CONTROL ROUTES
# =========================
@app.route("/webcam/start", methods=["POST"])
def webcam_start():
    success = webcam.start()
    if success:
        return jsonify({"status": "started", "message": "Webcam started successfully"})
    else:
        return jsonify({"status": "error", "message": "Could not open webcam. Check if camera is connected."}), 500

@app.route("/webcam/stop", methods=["POST"])
def webcam_stop():
    webcam.stop()
    return jsonify({"status": "stopped", "message": "Webcam stopped"})

@app.route("/webcam/status")
def webcam_status():
    with webcam.lock:
        stats = dict(webcam.latest_stats)
    return jsonify(stats)

# =========================
# LIVE VIDEO STREAM (MJPEG)
# =========================
@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with webcam.lock:
                is_running = webcam.running
            if not is_running:
                # Serve a "camera off" frame
                off_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(off_frame, "Camera is OFF", (180, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2)
                cv2.putText(off_frame, "Click 'Start Camera'", (155, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
                _, buf = cv2.imencode('.jpg', off_frame)
                frame_bytes = buf.tobytes()
            else:
                frame_bytes = webcam.get_jpeg_frame()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS target

    return Response(stream_with_context(generate()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# =========================
# SNAPSHOT + ANALYSIS FROM WEBCAM
# =========================
@app.route("/webcam/snapshot", methods=["POST"])
def webcam_snapshot():
    try:
        with webcam.lock:
            frame = webcam.latest_frame
            is_running = webcam.running

        if not is_running or frame is None:
            return jsonify({"error": "Webcam is not running or no frame available"}), 400

        results = model(frame, verbose=False, conf=0.35)
        output_img = results[0].plot()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        input_path = os.path.join(OUTPUT_FOLDER, f"{timestamp}_webcam_input.jpg")
        output_path = os.path.join(OUTPUT_FOLDER, f"{timestamp}_webcam_output.jpg")
        report_path = os.path.join(OUTPUT_FOLDER, f"{timestamp}_webcam_report.json")

        cv2.imwrite(input_path, frame)
        cv2.imwrite(output_path, output_img)

        _, buffer = cv2.imencode('.jpg', output_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        report = analyze(results, frame.shape)
        report["image"] = img_base64
        report["timestamp"] = timestamp

        with open(report_path, "w") as f:
            report_save = {k: v for k, v in report.items() if k != "image"}
            json.dump(report_save, f, indent=4)

        return jsonify(report)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Snapshot error: {str(e)}"}), 500

# =========================
# IMAGE UPLOAD PREDICTION API
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]
        img_bytes = file.read()
        npimg = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Failed to decode image. Invalid format."}), 400

        results = model(img, verbose=False)
        output_img = results[0].plot()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        input_path = os.path.join(OUTPUT_FOLDER, f"{timestamp}_input.jpg")
        output_path = os.path.join(OUTPUT_FOLDER, f"{timestamp}_output.jpg")
        report_path = os.path.join(OUTPUT_FOLDER, f"{timestamp}_report.json")

        cv2.imwrite(input_path, img)
        cv2.imwrite(output_path, output_img)

        _, buffer = cv2.imencode('.jpg', output_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        report = analyze(results, img.shape)
        report["image"] = img_base64

        with open(report_path, "w") as f:
            report_save = {k: v for k, v in report.items() if k != "image"}
            json.dump(report_save, f, indent=4)

        return jsonify(report)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)