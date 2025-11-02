import os
import cv2
import torch
import numpy as np
import time
from threading import Thread
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
yolo_model_path = os.path.join(BASE_DIR, "models", "my_yolo_model.pt")
yolo_model = YOLO(yolo_model_path)

# MiDaS
midas_model_type = "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDaS", midas_model_type)
midas.to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_warning = {"text": "", "timestamp": 0}
last_boxes_frame = None
frame_count = 0
DETECTION_INTERVAL = 5  # run YOLO+MiDaS every 5 frames

def gen_frames():
    global last_warning, last_boxes_frame, frame_count
    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame_count += 1

        if frame_count % DETECTION_INTERVAL == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # YOLO
            results = yolo_model.predict(frame, conf=0.5, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()

            # MiDaS
            input_batch = transform(img_rgb).to(device)
            with torch.no_grad():
                depth_pred = midas(input_batch)
                depth_pred = torch.nn.functional.interpolate(
                    depth_pred.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bilinear",
                    align_corners=False
                ).squeeze().cpu().numpy()

            depth_min, depth_max = depth_pred.min(), depth_pred.max()
            norm_depth = (depth_pred - depth_min) / (depth_max - depth_min + 1e-6)

            last_boxes_frame = []
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                cls = int(classes[i])
                conf = confidences[i]
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                distance_m = round((1 - norm_depth[cy, cx])*10, 2)
                label_text = f"{yolo_model.names[cls]} {conf*100:.1f}% {distance_m}m"
                last_boxes_frame.append((x1, y1, x2, y2, label_text, distance_m, cls))

                if distance_m < 2.0:
                    last_warning = {"text": f"Careful {yolo_model.names[cls]} nearby",
                                    "timestamp": time.time()}

        # Draw last known boxes on current frame
        if last_boxes_frame:
            for box in last_boxes_frame:
                x1, y1, x2, y2, label, dist, cls = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

# Django views
def index(request):
    return render(request, "index.html")

def video_feed(request):
    return StreamingHttpResponse(
        gen_frames(),
        content_type="multipart/x-mixed-replace; boundary=frame"
    )

def warnings_last_json(request):
    if time.time() - last_warning["timestamp"] > 5:
        return JsonResponse({"text": ""})
    return JsonResponse(last_warning)
