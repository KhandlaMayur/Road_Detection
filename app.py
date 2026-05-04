# =========================
# IMPORT LIBRARIES
# =========================
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import os
import json
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
    print("   (Download best.pt to runs/detect/train/weights/ for better accuracy)")
    model = YOLO("yolov8n.pt")  # Fallback to pre-trained nano model

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
    # ADVANCED DAMAGE ANALYSIS
    # =========================
    avg_damage_area = total_area / (potholes + cracks) if (potholes + cracks) > 0 else 0

    # Normalize area (based on image size)
    area_ratio = total_area / img_area

    # =========================
    # DYNAMIC SEVERITY BASED ON SIZE + COUNT
    # =========================
    if area_ratio < 0.05 and (potholes + cracks) <= 2:
        severity = "Low"
        risk = "Low"

    elif area_ratio < 0.20 and (potholes + cracks) <= 5:
        severity = "Medium"
        risk = "Moderate"

    else:
        severity = "High"
        risk = "High"

    # =========================
    # WORKERS, DAYS, MACHINES BASED ON DAMAGE
    # =========================
    base_workers = 2
    base_days = 1

    # Increase workers and days based on count and size
    labor_workers = base_workers + int((potholes * 0.5) + (cracks * 0.5))
    days = base_days + int((area_ratio * 5) + (potholes * 0.5) + (cracks * 0.3))

    # Cap minimum values
    labor_workers = max(2, min(labor_workers, 10))
    days = max(1, days)

    # =========================
    # MACHINE SELECTION LOGIC
    # =========================
    if severity == "Low":
        machine = "No"
        machine_type = "Manual Crack Sealing Tools"

    elif severity == "Medium":
        machine = "Yes"
        machine_type = "Patching Machine + Roller"

    else:
        machine = "Yes"
        machine_type = "Asphalt Paver + Roller + Compactor"

    # =========================
    # MATERIAL COST BASED ON DAMAGE SIZE
    # =========================
    material_rate = 0.05  # cost per pixel area (adjustable)

    material_cost = int(total_area * material_rate)

    # Minimum cost safety
    material_cost = max(material_cost, 200)

    # =========================
    # LABOR COST
    # =========================
    labor_charge_per_day = 300
    total_labor_cost = labor_workers * labor_charge_per_day * days

    # =========================
    # FINAL COST
    # =========================
    total_cost = total_labor_cost + material_cost

    # =========================
    # FINAL DECISION
    # =========================
    if severity == "Low":
        decision = "No urgent repair"

    elif severity == "Medium":
        decision = "Schedule maintenance"

    else:
        decision = "Immediate repair required"

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
    try:
        print("[DEBUG] Prediction request received")
        
        # Check if file exists
        if "image" not in request.files:
            print("[ERROR] No image file in request")
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files["image"]
        print(f"[DEBUG] File received: {file.filename}")

        # Read and decode image
        img_bytes = file.read()
        print(f"[DEBUG] Image bytes: {len(img_bytes)}")
        
        npimg = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if img is None:
            print("[ERROR] Failed to decode image")
            return jsonify({"error": "Failed to decode image. Invalid format."}), 400
        
        print(f"[DEBUG] Image shape: {img.shape}")

        # Run model inference on CPU and disable fuse to avoid Render memory issues
        print("[DEBUG] Running model inference... (cpu, fuse=False)")
        results = model(img, device="cpu", fuse=False, verbose=False)
        print("[DEBUG] Model inference completed")

        output_img = results[0].plot()
        print("[DEBUG] Output image plotted")

        # =========================
        # SAVE OUTPUT LOGIC
        # =========================
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        input_path = os.path.join(OUTPUT_FOLDER, f"{timestamp}_input.jpg")
        output_path = os.path.join(OUTPUT_FOLDER, f"{timestamp}_output.jpg")
        report_path = os.path.join(OUTPUT_FOLDER, f"{timestamp}_report.json")

        # Save original image
        cv2.imwrite(input_path, img)
        print(f"[DEBUG] Saved input image: {input_path}")

        # Save detected image
        cv2.imwrite(output_path, output_img)
        print(f"[DEBUG] Saved output image: {output_path}")

        # Encode image for frontend
        _, buffer = cv2.imencode('.jpg', output_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        print(f"[DEBUG] Image encoded to base64: {len(img_base64)} chars")

        report = analyze(results, img.shape)
        report["image"] = img_base64
        print(f"[DEBUG] Report generated: {report}")

        # Save JSON report
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        print(f"[DEBUG] Report saved: {report_path}")

        print("[DEBUG] Returning response")
        return jsonify(report)
        
    except Exception as e:
        print(f"[ERROR] Exception in predict: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = False
    app.run(host="0.0.0.0", port=port, debug=debug)