# =========================
# IMPORT LIBRARIES
# =========================
import os
import cv2
import pandas as pd
from ultralytics import YOLO

# =========================
# PATH SETTINGS
# =========================
DATASET_PATH = "dataset"
TEST_IMAGES_PATH = "dataset/test/images"
RANDOM_IMAGE_PATH = "random_images/test3.jpg"

OUTPUT_CSV = "outputs/dataset_report.csv"
OUTPUT_IMAGE = "outputs/random_output.jpg"

BEST_MODEL_PATH = "runs/detect/train/weights/best.pt"

# =========================
# CREATE data.yaml
# =========================
yaml_content = f"""
path: {DATASET_PATH}
train: train/images
val: val/images
test: test/images

names:
  0: crack
  1: pothole
"""

with open("data.yaml", "w") as f:
    f.write(yaml_content)

# =========================
# CREATE OUTPUT FOLDER
# =========================
os.makedirs("outputs", exist_ok=True)

# =========================
# TRAIN MODEL (ONLY FIRST TIME)
# =========================
if not os.path.exists(BEST_MODEL_PATH):
    print("Training Model...")

    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yaml",
        epochs=20,
        imgsz=416,
        batch=8,
        device="cpu",
        name="train",
        exist_ok=True
    )

    print("Training Completed!")
else:
    print("Model already trained. Skipping training...")

# =========================
# LOAD MODEL
# =========================
model = YOLO(BEST_MODEL_PATH)

# =========================
# ADVANCED ANALYSIS FUNCTION
# =========================
def analyze(results, image_shape):
    boxes = results[0].boxes
    names = model.names

    potholes = 0
    cracks = 0
    damage_types = set()

    total_damage_area = 0
    img_area = image_shape[0] * image_shape[1]

    for box in boxes:
        cls = int(box.cls[0])
        label = names[cls]

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1)
        total_damage_area += area

        # Count types
        if label == "pothole":
            potholes += 1
            damage_types.add("Pothole")
        else:
            cracks += 1
            damage_types.add("Crack")

    # =========================
    # DAMAGE PERCENTAGE
    # =========================
    damage_percentage = float((total_damage_area / img_area) * 100) if img_area > 0 else 0

    # =========================
    # SEVERITY CLASSIFICATION
    # =========================
    if damage_percentage <= 20:
        severity = "Low"
        cost_range = (50, 150)
        labor = "2-3"
        machine = "No"
        machine_type = "Crack Sealing Machine"
        decision = "No urgent repair"
        risk = "Low"

    elif damage_percentage <= 50:
        severity = "Medium"
        cost_range = (150, 400)
        labor = "4-6"
        machine = "Yes"
        machine_type = "Patching Machine + Roller"
        decision = "Schedule maintenance"
        risk = "Moderate"

    elif damage_percentage <= 80:
        severity = "High"
        cost_range = (400, 1000)
        labor = "6-10"
        machine = "Yes"
        machine_type = "Asphalt Paver + Roller"
        decision = "Immediate repair required"
        risk = "High"

    else:
        severity = "Critical"
        cost_range = (1000, 2500)
        labor = "10-20"
        machine = "Yes"
        machine_type = "Milling Machine + Paver + Compactor"
        decision = "Road reconstruction required"
        risk = "Dangerous"

    # =========================
    # COST ESTIMATION
    # =========================
    avg_cost_per_m2 = sum(cost_range) / 2
    estimated_cost = total_damage_area * avg_cost_per_m2 * 0.0001  # scale factor

    # =========================
    # TIME ESTIMATION
    # =========================
    if severity == "Low":
        time_required = "1-2 days"
    elif severity == "Medium":
        time_required = "2-4 days"
    elif severity == "High":
        time_required = "4-7 days"
    else:
        time_required = "1-2 weeks"

    return {
        "potholes": potholes,
        "cracks": cracks,
        "damage_types": ", ".join(damage_types),
        "damage_percentage": round(damage_percentage, 2),
        "severity": severity,
        "risk": risk,
        "cost": round(estimated_cost, 2),
        "labor": labor,
        "machine": machine,
        "machine_type": machine_type,
        "decision": decision,
        "time": time_required
    }

# =========================
# RANDOM IMAGE DETECTION
# =========================
print("\nProcessing Random Image...")

img = cv2.imread(RANDOM_IMAGE_PATH)
results = model(RANDOM_IMAGE_PATH, conf=0.25)

for r in results:
    output_img = r.plot()

cv2.imwrite(OUTPUT_IMAGE, output_img)

report = analyze(results, img.shape)

# =========================
# FINAL STRUCTURED REPORT
# =========================
print("\n===== ROAD DAMAGE REPORT =====\n")

print(f"Damage Type: {report['damage_types']}")
print(f"Number of Potholes: {report['potholes']}")
print(f"Number of Cracks: {report['cracks']}")
print(f"Damage Percentage: {report['damage_percentage']}%")
print(f"Severity: {report['severity']}")
print(f"Risk Level: {report['risk']}")

print(f"\nEstimated Cost: ₹{report['cost']}")
print(f"Labor Required: {report['labor']} workers")
print(f"Time Required: {report['time']}")

print(f"\nMachine Required: {report['machine']}")
print(f"Machine Type: {report['machine_type']}")

print(f"\nFinal Decision: {report['decision']}")
print(f"\nOutput Image Saved at: {OUTPUT_IMAGE}")