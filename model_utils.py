import io, sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

DB_PATH = "database/digits.db"
MODEL_PATH = "models/digit_cnn.pth"

# ----- Dataset -----
class DigitDataset(Dataset):
    def __init__(self, db_path):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT image, label FROM digits")
        self.data = cur.fetchall()
        conn.close()

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        blob, label = self.data[idx]
        img = Image.open(io.BytesIO(blob)).convert("RGB")
        return self.transform(img), label

# ----- Model -----
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

# ----- Training -----
def train_from_db(epochs=10):
    dataset = DigitDataset(DB_PATH)
    if len(dataset) < 10:
        raise ValueError("Not enough data in database to train. Upload more images first.")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    st.write(f"Training on **{device}** with {len(dataset)} samples...")

    epoch_losses = []
    epoch_accuracies = []

    progress = st.progress(0)
    chart = st.empty()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct, total = 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        epoch_losses.append(train_loss)
        epoch_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total else 0

        progress.progress((epoch + 1) / epochs)
        chart.pyplot(plot_training_progress(epoch_losses, epoch_accuracies))
        st.write(f"**Epoch {epoch+1}/{epochs}** → Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    st.success(f"✅ Model saved to `{MODEL_PATH}`")

    final_acc = epoch_accuracies[-1] * 100
    st.metric("Final Accuracy", f"{final_acc:.2f}%")

def plot_training_progress(losses, accuracies):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(losses, color='red', label='Loss')
    ax2.plot(accuracies, color='blue', label='Accuracy')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='red')
    ax2.set_ylabel('Accuracy', color='blue')
    fig.tight_layout()
    return fig

# ----- Prediction -----
def predict_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    model = DigitCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_t)
        _, predicted = torch.max(output, 1)
    return predicted.item()
