"""
Bug Case 3: Wrong Number of Output Classes
Changed NUM_CLASSES from 10 to 5.
The final linear layer outputs 5 values but labels go 0-9,
causing an IndexError in CrossEntropyLoss.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 3
NUM_CLASSES = 5  # BUG: Original was 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root="../data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, NUM_CLASSES),  # Only 5 outputs instead of 10
        )
    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))

model = FashionCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Training with NUM_CLASSES=5 (BUG)...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)  # IndexError: target out of range
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} done")
