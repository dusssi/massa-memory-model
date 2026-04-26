import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from data.dataset import generate_data
from models.lstm_model import LSTMModel
from models.memory_model import MemoryModel


# =========================
# Setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs("results", exist_ok=True)


# =========================
# Data
# =========================
X, y = generate_data(seq_len=15, num_samples=2000)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=64,
    shuffle=False
)


# =========================
# Training
# =========================
def train_model(model, loader, epochs=100):
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    losses = []
    best_loss = float("inf")
    patience = 8
    counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            output = model(X_batch)
            loss = criterion(output, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered")
            break

    return losses


# =========================
# Evaluation
# =========================
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            output = model(X_batch)
            preds = torch.argmax(output, dim=1)

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    return correct / total


# =========================
# Run
# =========================
lstm_model = LSTMModel()
memory_model = MemoryModel()

print("\nTraining LSTM Model...")
lstm_losses = train_model(lstm_model, train_loader)

print("\nTraining Memory Model...")
memory_losses = train_model(memory_model, train_loader)

lstm_acc = evaluate(lstm_model, test_loader)
memory_acc = evaluate(memory_model, test_loader)

print("\n========== FINAL RESULTS ==========")
print(f"LSTM Accuracy        : {lstm_acc:.4f}")
print(f"Memory Model Accuracy: {memory_acc:.4f}")
print("===================================")


# =========================
# Graphs
# =========================
plt.figure()
plt.plot(lstm_losses, label="LSTM")
plt.plot(memory_losses, label="Memory")
plt.legend()
plt.savefig("results/loss.png")

plt.figure()
plt.bar(["LSTM", "Memory"], [lstm_acc, memory_acc])
plt.savefig("results/accuracy.png")

print("\nGraphs saved in 'results/' folder.")