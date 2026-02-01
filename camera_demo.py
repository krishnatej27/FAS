import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

# ----------------------------
# Load model
# ----------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("fas_model.pth", map_location="cpu"))
model.eval()

# ----------------------------
# Image preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----------------------------
# Open camera
# ----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()

print("Camera started. Press Q to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess
    img = transform(rgb).unsqueeze(0)

    # Inference
    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()

    label = "REAL (Live)" if pred == 0 else "SPOOF (Fake)"
    color = (0, 255, 0) if pred == 0 else (0, 0, 255)

    # Display result
    cv2.putText(
        frame,
        label,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Face Anti-Spoofing Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
