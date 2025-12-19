from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

model = tf.saved_model.load("saved_model")
infer = model.signatures["serving_default"]

class_names = [
    "brain_glioma",
    "brain_meningioma",
    "brain_normal",
    "brain_pituitary",
    "chest_normal",
    "chest_pneumonia"
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((256, 256))

        # DON'T manually normalize or scale
        img_array = np.array(img).astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        output = infer(tf.constant(img_array))
        probs = output[list(output.keys())[0]].numpy()[0]

        pred_index = int(np.argmax(probs))
        pred_class = class_names[pred_index]
        confidence = float(probs[pred_index])

        # build probability dict
        prob_dict = {
            class_names[i]: float(probs[i])
            for i in range(len(class_names))
        }

        return {
            "prediction": pred_class,
            "confidence": confidence,
            "probabilities": prob_dict
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
