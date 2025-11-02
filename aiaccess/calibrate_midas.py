# calibrate_midas.py
import cv2
import torch
import numpy as np
import os
import json
from ultralytics import YOLO

# Settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = os.path.join(BASE_DIR, 'models', 'calib.json')
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'my_yolo_model.pt')
SAMPLES = []  # will hold tuples (median_depth, true_distance)

# Load models (MiDaS + YOLO)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Loading MiDaS...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)
midas.eval()
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
midas_transform = transforms.small_transform
print("MiDaS loaded.")

print("Loading YOLO (for optional bounding boxes)...")
yolo = YOLO(YOLO_MODEL_PATH).to(device)
yolo.eval()
print("YOLO loaded.")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nCalibration helper.")
print("You will be asked to place an object/person at a known distance and press 'c' to capture.")
print("Press 'q' to finish and fit the mapping.\n")

distances_to_capture = [0.5, 1.0, 1.5, 2.0, 3.0]  # adjust or extend as needed
print("Suggested capture distances (meters):", distances_to_capture)
print("You can capture any distances, not limited to the suggested ones.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    display = frame.copy()
    cv2.putText(display, "Press 'c' to capture sample, 'q' to quit and fit.", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Calibration (press c to capture)", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # compute midas depth map
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = midas_transform(rgb).to(device)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        with torch.no_grad():
            depth_map = midas(input_tensor).squeeze().cpu().numpy()
        depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
        # normalize
        depth_map_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)

        # Optionally detect a person/box to crop; here we use whole frame median as fallback
        results = yolo.predict(source=rgb, device=device, verbose=False)[0]
        median_val = None
        if hasattr(results, 'boxes') and len(results.boxes) > 0:
            # pick largest box (assume main subject)
            boxes = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                area = max(0, (x2 - x1) * (y2 - y1))
                boxes.append((area, x1, y1, x2, y2))
            boxes.sort(reverse=True)
            _, x1, y1, x2, y2 = boxes[0]
            crop = depth_map_norm[y1:y2, x1:x2]
            if crop.size > 0:
                median_val = float(np.median(crop))

            # draw box for user feedback
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        if median_val is None:
            median_val = float(np.median(depth_map_norm))

        # Ask user to input true distance
        print(f"\nCaptured median_depth = {median_val:.6f}")
        user = input("Enter true distance (meters) for this capture (e.g., 1.5), or 'skip' to discard: ").strip()
        if user.lower() == 'skip':
            print("Sample discarded.")
        else:
            try:
                true_d = float(user)
                SAMPLES.append((median_val, true_d))
                print(f"Saved sample: depth={median_val:.6f} -> dist={true_d:.3f} m")
            except:
                print("Invalid input — sample discarded.")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if len(SAMPLES) < 3:
    print("Not enough samples collected (need at least 3). Exiting without fitting.")
    exit(1)

# Fit model: distance = m * (1/(depth + eps)) + c
eps = 1e-6
inv_depths = np.array([1.0 / (d + eps) for d, _ in SAMPLES])
dists = np.array([dist for _, dist in SAMPLES])
# fit linear: dists ≈ m * inv_depths + c
m, c = np.polyfit(inv_depths, dists, 1)

print("\nFitted calibration parameters:")
print(f"m = {m:.6f}, c = {c:.6f}, eps = {eps}")

os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
with open(CALIB_PATH, 'w') as f:
    json.dump({'m': float(m), 'c': float(c), 'eps': float(eps)}, f, indent=2)

print(f"Saved calibration to {CALIB_PATH}.")
print("You can now use the updated views.py which reads this calib.json.")
