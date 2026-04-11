# =========================
# IMPORT LIBRARIES
# =========================
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import os
from ultralytics import YOLO

app = Flask(__name__)

# =========================
# MODEL PATH
# =========================
BEST_MODEL_PATH = "runs/detect/train/weights/best.pt"

# =========================
# TRAIN MODEL (IF NOT EXISTS)
# =========================
if not os.path.exists(BEST_MODEL_PATH):
    print(" Training Model (YOLOv8n)...")

    model = YOLO("yolov8n.pt")   #  using nano model

    model.train(
        data="data.yaml",
        epochs=35,
        imgsz=640,
        batch=8,        # nano supports higher batch
        device="cpu",   # use 0 if GPU
        patience=20,
        name="train",
        exist_ok=True
    )

    print(" Training Completed!")

else:
    print("⚡ Model already trained. Loading model...")
    model = YOLO(BEST_MODEL_PATH)

# =========================
# ANALYSIS FUNCTION
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

    # =========================
    # NO DAMAGE CONDITION
    # =========================
    if potholes == 0 and cracks == 0:
        return {
            "potholes": 0,
            "cracks": 0,
            "damage_types": "None",
            "damage_percentage": 0,
            "severity": "Low",
            "risk": "Low",
            "labor": 0,
            "days": 0,
            "material_cost": 0,
            "total_cost": 0,
            "machine": "No",
            "machine_type": "No",
            "decision": "No urgent repair"
        }

    # =========================
    # SEVERITY LOGIC
    # =========================
    if damage_percentage <= 20:
        severity = "Low"
        risk = "Low"
        days = 2
        labor_workers = 3
        machine = "No"
        machine_type = "Crack Sealing Machine"
        decision = "No urgent repair"
        material_cost = 200

    elif damage_percentage <= 50:
        severity = "Medium"
        risk = "Moderate"
        days = 4
        labor_workers = 5
        machine = "Yes"
        machine_type = "Patching Machine + Roller"
        decision = "Schedule maintenance"
        material_cost = 800

    else:
        severity = "High"
        risk = "High"
        days = 7
        labor_workers = 8
        machine = "Yes"
        machine_type = "Asphalt Paver + Roller"
        decision = "Immediate repair required"
        material_cost = 2000

    # =========================
    # COST CALCULATION
    # =========================
    labor_charge_per_day = 500
    total_labor_cost = labor_workers * labor_charge_per_day * days
    total_cost = total_labor_cost + material_cost

    damage_types = []
    if potholes > 0:
        damage_types.append("Pothole")
    if cracks > 0:
        damage_types.append("Crack")

    return {
        "potholes": potholes,
        "cracks": cracks,
        "damage_types": ", ".join(damage_types),
        "damage_percentage": round(damage_percentage, 2),
        "severity": severity,
        "risk": risk,
        "labor": labor_workers,
        "days": days,
        "material_cost": material_cost,
        "total_cost": total_cost,
        "machine": machine,
        "machine_type": machine_type,
        "decision": decision
    }

# =========================
# HOME ROUTE
# =========================
@app.route("/")
def home():
    return render_template("index.html")

# =========================
# PREDICTION API
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    img_bytes = file.read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model(img)

    output_img = results[0].plot()

    _, buffer = cv2.imencode('.jpg', output_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    report = analyze(results, img.shape)
    report["image"] = img_base64

    return jsonify(report)

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True)