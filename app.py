from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fas-mocha.vercel.app",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
model = None

def load_model():
    global model
    if model is None:
        print("Loading model...")
        model = torch.load("fas_model.pth", map_location="cpu")
        model.eval()
        print("Model loaded")
    return model

@app.get("/")
def home():
    return {"message": "FAS API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        print("Received request")
        contents = await file.read()
        print("Image read")
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = np.array(image)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image_tensor = torch.tensor(image, dtype=torch.float32)
        model = load_model()
        print("Model ready")
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()
        print("Prediction done")
        result = "real" if prediction == 1 else "fake"
        return {
            "result": result,
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}
