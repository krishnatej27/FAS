from fastapi import FastAPI, UploadFile, File
import torch
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI()

# Load model once
model = None

def load_model():
    global model
    if model is None:
        model = torch.load("fas_model.pth", map_location=torch.device("cpu"))
        model.eval()
    return model

@app.get("/")
def home():
    return {"message": "FAS API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Convert image
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = np.array(image)

        # Resize (IMPORTANT: match your model input)
        image = cv2.resize(image, (224, 224))

        # Normalize
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC → CHW
        image = np.expand_dims(image, axis=0)

        image_tensor = torch.tensor(image, dtype=torch.float32)

        # Load model
        model = load_model()

        # Prediction
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()

        result = "real" if prediction == 1 else "fake"

        return {
            "result": result,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        return {"error": str(e)}