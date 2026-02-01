import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import pandas as pd
from datasets.image_dataset import ImageDataset

# =========================================================
# Wrapper to make ImageDataset compatible with DataLoader
# =========================================================
class WrappedImageDataset(Dataset):
    def __init__(self, image_dataset, length):
        self.image_dataset = image_dataset
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.image_dataset[idx]

# =========================================================
# Load CSV (image paths + labels)
# =========================================================
df = pd.read_csv('examples/example.csv', header=None)
image_list = df[0].tolist()
labels = [int(x > 0) for x in df[1].tolist()]  # binary labels: 0=real, 1=spoof

# =========================================================
# Image preprocessing (CRITICAL)
# =========================================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),   # -> [3, 224, 224]
])

# =========================================================
# Base dataset from FAS_DataManager
# =========================================================
base_dataset = ImageDataset(
    file_list=image_list,
    label_list=labels,
    torchvision_transform=transform,
    use_original_frame=False,
    bbox_suffix='_bbox_mtccnn.txt'
)

# =========================================================
# Wrapped dataset for PyTorch
# =========================================================
dataset = WrappedImageDataset(
    image_dataset=base_dataset,
    length=len(image_list)
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)

# =========================================================
# Model: ResNet-18
# =========================================================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device("cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# =========================================================
# Training loop (1 epoch demo)
# =========================================================
model.train()

for batch in loader:
    # FAS_DataManager batch structure
    imgs = batch[1].to(device)                            # image tensor
    lbls = batch[2]['spoofing_label'].to(device)          # label tensor

    # Ensure correct shape
    if imgs.dim() == 3:
        imgs = imgs.unsqueeze(0)

    outputs = model(imgs)
    loss = criterion(outputs, lbls)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training ran successfully")

torch.save(model.state_dict(), "fas_model.pth")

