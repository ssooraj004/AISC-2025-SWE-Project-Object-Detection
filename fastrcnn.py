import os
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset

# Custom Dataset with dummy boxes
class FurnitureDataset(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = ImageFolder(root=root)
        self.transform = transform

    def __getitem__(self, idx):
        path, label = self.dataset.imgs[idx]
        image = Image.open(path).convert("RGB")
        width, height = image.size

        # Dummy bounding box in the center
        box = [width // 4, height // 4, 3 * width // 4, 3 * height // 4]
        target = {
            "boxes": torch.tensor([box], dtype=torch.float32),
            "labels": torch.tensor([label], dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }

        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.dataset)

# Transform
transform = T.Compose([
    T.ToTensor()
])

# Load dataset and DataLoader
dataset = FurnitureDataset(root="furniture", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Load pretrained Faster R-CNN
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)

# Modify the classifier head for our number of classes
num_classes = len(dataset.dataset.classes) + 1  # +1 for background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9)

# Training loop
model.train()
num_epochs = 1

for epoch in range(num_epochs):
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        print(f"[Epoch {epoch}] Loss: {losses.item():.4f}")
