from fastapi import FastAPI, UploadFile, File
import torch

app = FastAPI()

# dummy model load (replace later)
model = None

@app.get("/")
def home():
    return {"message": "FAS API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # later: process image + model
    return {"result": "prediction placeholder"}