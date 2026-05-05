# =========================
# IMPORT LIBRARIES
# =========================
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import json
import threading
import time
from datetime import datetime
from ultralytics import YOLO
import socket

app = Flask(__name__)
CORS(app)

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
# WEBCAM STREAM STATE (REMOVED)
# =========================
# Webcam processing is now handled on the client-side using the device's back camera.

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
# LIVE WEBCAM PROCESSING (CLIENT-SIDE)
# =========================
@app.route("/process_frame", methods=["POST"])
def process_frame():
    try:
        data = request.json
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode base64 image
        img_data = data["image"]
        if img_data.startswith("data:image"):
            img_data = img_data.split(",")[1]
            
        img_bytes = base64.b64decode(img_data)
        npimg = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Failed to decode image."}), 400

        # Run model inference
        results = model(frame, verbose=False, conf=0.35)
        output_img = results[0].plot()

        # Compute stats
        report = analyze(results, frame.shape)
        
        # Encode output image
        _, buffer = cv2.imencode('.jpg', output_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "image": img_base64,
            "stats": report
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

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
        # Run model inference
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

    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        local_ip = "127.0.0.1"

    # Only start ngrok tunnel when running LOCALLY (not on Render/cloud)
    is_render = os.environ.get("RENDER") is not None
    ngrok_url = None

    if not is_render:
        try:
            from pyngrok import ngrok as pyngrok
            pyngrok.set_auth_token("3DI4n30ZY0OfJ5HKYcykE9uTyAd_6VF7YCj9mNQzT4NJUV6uf")
            tunnel = pyngrok.connect(port, "http")
            ngrok_url = tunnel.public_url.replace("http://", "https://")
        except Exception as e:
            print(f"  [ngrok] Could not auto-start: {e}")

    print("\n" + "="*55)
    print("  RoadGuard AI - Pothole & Crack Detection")
    print("="*55)
    if is_render:
        print("  Running on Render (HTTPS provided by Render)")
    else:
        print(f"  PC browser    ->  http://localhost:{port}")
        if ngrok_url:
            print(f"\n  MOBILE URL (open this on your phone):")
            print(f"  >>> {ngrok_url} <<<")
            print(f"  Works on any phone, any camera, any network!")
        else:
            print(f"  For mobile: run 'ngrok http {port}' in another terminal")
    print("="*55 + "\n")

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)