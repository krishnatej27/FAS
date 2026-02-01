import os
from datasets.image_dataset import ImageDataset

# Use provided example images
img_dir = "datasets/train_img"

# Build image list
img_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]

# Dummy labels (all 0 = live)
labels = [0] * len(img_list)

dataset = ImageDataset(
    img_list=img_list,
    label_list=labels
)

print("Dataset length:", len(dataset))

img, label = dataset[0]
print("Image type:", type(img))
print("Image shape:", img.shape)
print("Label:", label)
