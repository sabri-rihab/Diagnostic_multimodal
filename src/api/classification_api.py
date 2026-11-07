from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
from pathlib import Path
import io
from PIL import Image

app = FastAPI()

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
# MODEL_PATH = Path("/app/models/fine_tuning_best.keras")
MODEL_PATH = Path("models/fine_tuning_best.keras")
model = tf.keras.models.load_model(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB").resize((299, 299))
    x = preprocess_input(np.expand_dims(np.array(img), 0))
    probs = model.predict(x)[0]
    idx = int(np.argmax(probs))
    return JSONResponse({
        "class": CLASSES[idx],
        "confidence": round(float(probs[idx]), 4),
        "all_probs": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)}
    })
