"""
Bug Case 6: Removed All ReLU Activations
Without nonlinear activations, the entire CNN collapses to a linear model.
It can still train but achieves significantly lower accuracy.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 3
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root="../data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="../data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # BUG: Removed all ReLU activations
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            # nn.ReLU(),  <-- REMOVED
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.ReLU(),  <-- REMOVED
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            # nn.ReLU(),  <-- REMOVED
            nn.Dropout(0.25),
            nn.Linear(128, NUM_CLASSES),
        )
    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))

model = FashionCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Training WITHOUT ReLU activations (BUG)...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = total = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    print(f"Epoch {epoch}: Loss={running_loss/len(train_loader):.4f}, Acc={100.*correct/total:.2f}%")

model.eval()
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
print(f"Test Acc: {100.*correct/total:.2f}%")
