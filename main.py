from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Medical Diagnosis API")

# Load model once at startup
model = tf.saved_model.load("./saved_model")
infer = model.signatures["serving_default"]

# Class labels (must match training order)
class_names = [
    "brain_glioma",
    "brain_meningioma",
    "brain_normal",
    "brain_pituitary",
    "chest_normal",
    "chest_pneumonia"
]

# ✅ ROOT ENDPOINT (FIXES 404)
@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Medical Diagnosis API is live",
        "endpoints": {
            "predict": "POST /predict (multipart/form-data, key=file)"
        }
    }

# ✅ PREDICTION ENDPOINT
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Image preprocessing (matches training)
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((256, 256))

        img_array = np.array(img).astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # Inference
        output = infer(tf.constant(img_array))
        probs = output[list(output.keys())[0]].numpy()[0]

        pred_index = int(np.argmax(probs))
        prediction = class_names[pred_index]
        confidence = float(probs[pred_index])

        probabilities = {
            class_names[i]: float(probs[i])
            for i in range(len(class_names))
        }

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
