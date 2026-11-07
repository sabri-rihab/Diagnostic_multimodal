from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import base64
import cv2

app = FastAPI()

# MODEL_PATH = Path("/app/models/yolov8n_v1-4/weights/best.pt")
MODEL_PATH = Path("models/yolov8n_v1-4/weights/best.pt")
CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
model = YOLO(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    results = model(img)[0]
    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = round(float(box.conf[0]), 4)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{CLASSES[cls]} {conf}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        detections.append({"class": CLASSES[cls], "confidence": conf,
                           "bbox": [x1, y1, x2, y2]})

    _, buf = cv2.imencode(".jpg", img)
    img_b64 = base64.b64encode(buf).decode()
    return JSONResponse({"detections": detections, "image_b64": img_b64})
