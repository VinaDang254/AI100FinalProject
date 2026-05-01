"""
AI 100 Final Project - Fashion-MNIST CNN Classifier
Authors: Devin Myers, Vina Dang

A convolutional neural network that classifies Fashion-MNIST images
into 10 clothing categories using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── Hyperparameters ──────────────────────────────────────────────
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fashion-MNIST class labels
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ── Data Loading ─────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def get_data_loaders():
    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


# ── CNN Model ────────────────────────────────────────────────────
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 14x14 -> 7x7
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ── Training ─────────────────────────────────────────────────────
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
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

    acc = 100.0 * correct / total
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Train Acc={acc:.2f}%")
    return avg_loss, acc


# ── Evaluation ───────────────────────────────────────────────────
def evaluate(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100.0 * correct / total
    avg_loss = running_loss / len(test_loader)
    print(f"Test: Loss={avg_loss:.4f}, Test Acc={acc:.2f}%")
    return avg_loss, acc


# ── Main ─────────────────────────────────────────────────────────
def main():
    print(f"Using device: {DEVICE}")
    print("Loading Fashion-MNIST dataset...")

    train_loader, test_loader = get_data_loaders()

    model = FashionCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nModel architecture:\n{model}\n")
    print(f"Training for {EPOCHS} epochs...\n")

    train_losses, train_accs = [], []
    for epoch in range(1, EPOCHS + 1):
        loss, acc = train(model, train_loader, criterion, optimizer, epoch)
        train_losses.append(loss)
        train_accs.append(acc)

    test_loss, test_acc = evaluate(model, test_loader, criterion)

    # Save training plot
    os.makedirs("outputs", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(range(1, EPOCHS + 1), train_losses)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.plot(range(1, EPOCHS + 1), train_accs)
    ax2.set_title("Training Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    plt.tight_layout()
    plt.savefig("outputs/training_plot.png")
    print("\nTraining plot saved to outputs/training_plot.png")

    # Save model
    torch.save(model.state_dict(), "outputs/fashion_cnn.pth")
    print(f"Model saved. Final test accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
