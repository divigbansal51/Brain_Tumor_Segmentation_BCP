import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Random MRI Dataset
# -----------------------------
class RandomMRIDataset(Dataset):
    def __init__(self, num_samples=50):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randn(1, 128, 128)
        y = torch.randint(0, 2, (1,)).float()
        return x, y


# -----------------------------
# Simple CNN (with extra conv)
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# -----------------------------
# Dice Loss
# -----------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds).view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice = (2 * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )
        return 1 - dice


# -----------------------------
# Training
# -----------------------------
def train():
    dataset = RandomMRIDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = SimpleCNN()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []

    for epoch in range(5):
        epoch_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds.view(-1), y.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Plot loss curve
    plt.plot(losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve (Random MRI Dataset)")
    plt.show()


if __name__ == "__main__":
    train()
