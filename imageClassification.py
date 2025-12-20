import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Data directory
data_dir = ""

# --- Define transformations ---
# Basic transformation for validation and test (no augmentation)
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Augmentation only for training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Step 1: Load full dataset (without transform for now) ---
full_dataset = datasets.ImageFolder(root=data_dir)

# --- Step 2: Split into train, val, test ---
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

# --- Step 3: Assign transforms after splitting ---
train_set.dataset.transform = train_transform
val_set.dataset.transform = base_transform
test_set.dataset.transform = base_transform

# --- Step 4: Create DataLoaders ---
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")


import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(name):
    if name == "vgg16":
        model = models.vgg16(weights="IMAGENET1K_V1")
        model.classifier[6] = nn.Linear(4096, 3)
    elif name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, 3)
    elif name == "densenet121":
        model = models.densenet121(weights="IMAGENET1K_V1")
        model.classifier = nn.Linear(model.classifier.in_features, 3)
    return model.to(device)

def train_model(model, train_loader, val_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Training loss: {total_loss / len(train_loader):.4f}")
    return model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, test_loader, model_name):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print(f"\n=== {model_name.upper()} ===")
    print(classification_report(all_labels, all_preds, target_names=test_loader.dataset.dataset.classes))

models_to_compare = ["vgg16", "resnet50", "densenet121"]

for m in models_to_compare:
    model = get_model(m)
    model = train_model(model, train_loader, val_loader, epochs=5)
    evaluate_model(model, test_loader, m)

#cross validation with all metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, average_precision_score
)
import numpy as np
import time, os, sys
import pandas as pd
from tqdm import tqdm
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =============================
# Model builder
# =============================
def get_model(name):
    if name == "vgg16":
        model = models.vgg16(weights="IMAGENET1K_V1")
        model.classifier[6] = nn.Linear(4096, 3)
    elif name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, 3)
    elif name == "densenet121":
        model = models.densenet121(weights="IMAGENET1K_V1")
        model.classifier = nn.Linear(model.classifier.in_features, 3)
    return model.to(device)

# =============================
# Training loop
# =============================
def train_one_fold(model, train_loader, val_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(train_loader):.4f}")
    return model

# =============================
# Evaluation per fold
# =============================
def evaluate_fold(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')

    # One-vs-rest ROC-AUC (requires probabilities)
    try:
        roc_auc = roc_auc_score(
            np.eye(len(np.unique(y_true)))[y_true], y_probs, average='macro', multi_class='ovr'
        )
    except Exception:
        roc_auc = np.nan

    # PR curve area for Healthy class (assuming class 0)
    try:
        ap = average_precision_score(
            (y_true == 0).astype(int), y_probs[:, 0]
        )
    except Exception:
        ap = np.nan

    cm = confusion_matrix(y_true, y_pred)
    return acc, f1, prec, rec, roc_auc, ap, cm

# =============================
# Cross-validation protocol
# =============================
def cross_validate_image_model(model_name, dataset, batch_size=32, epochs=5, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_labels = np.array(dataset.targets)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
        print(f"\n===== Fold {fold+1}/{n_splits} - {model_name} =====")

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

        model = get_model(model_name)
        start_time = time.time()
        model = train_one_fold(model, train_loader, val_loader, epochs)
        inf_start = time.time()

        # Inference time (ms/sample)
        n_samples = len(val_loader.dataset)
        _ = evaluate_fold(model, val_loader)  # warm-up
        inf_end = time.time()
        inference_time = ((inf_end - inf_start) / n_samples) * 1000

        acc, f1, prec, rec, roc_auc, ap, cm = evaluate_fold(model, val_loader)
        model_size = sum(p.numel() for p in model.parameters()) * 4 / (1024**2)

        print(f"Fold {fold+1} results → Acc: {acc:.3f}, F1: {f1:.3f}, ROC-AUC: {roc_auc:.3f}")
        fold_metrics.append([acc, f1, prec, rec, roc_auc, ap, inference_time, model_size])

    fold_metrics = np.array(fold_metrics)
    mean_std = lambda arr: f"{arr.mean():.3f} ± {arr.std():.3f}"

    result = {
        "Model": model_name,
        "Accuracy": mean_std(fold_metrics[:,0]),
        "Macro F1": mean_std(fold_metrics[:,1]),
        "Precision": mean_std(fold_metrics[:,2]),
        "Recall": mean_std(fold_metrics[:,3]),
        "ROC-AUC": mean_std(fold_metrics[:,4]),
        "PR(Healthy)": mean_std(fold_metrics[:,5]),
        "Inference (ms/sample)": f"{fold_metrics[:,6].mean():.2f}",
        "Model Size (MB)": f"{fold_metrics[:,7].mean():.2f}"
    }
    return result

# =============================
# Run experiments
# =============================
from torchvision import datasets, transforms

# Replace path with your dataset root
data_dir = ""
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)

models_to_compare = ["vgg16", "resnet50", "densenet121"]
final_results = []

for name in models_to_compare:
    res = cross_validate_image_model(name, dataset, epochs=5, n_splits=10)
    final_results.append(res)

# Display summary table
results_df = pd.DataFrame(final_results)
print("\n=== Cross-Validation Summary ===")
print(results_df)
results_df.to_csv("cnn_crossval_results.csv", index=False)
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, average_precision_score
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Model builder
# -------------------------
def get_densenet_model(num_classes=3):
    model = models.densenet121(weights="IMAGENET1K_V1")
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model.to(device)

# -------------------------
# Training function
# -------------------------
def train_model(model, train_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(train_loader):.4f}")
    return model

# -------------------------
# Evaluation function
# -------------------------
def evaluate_model(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')

    roc_auc = np.nan
    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(np.eye(len(np.unique(y_true)))[y_true], y_probs, average='macro', multi_class='ovr')

    cm = confusion_matrix(y_true, y_pred)
    return acc, f1, prec, rec, roc_auc, cm

# -------------------------
# Dataset and DataLoader
# -------------------------
data_dir = ""
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Split into train/test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# -------------------------
# Train DenseNet121
# -------------------------
model = get_densenet_model(num_classes=len(dataset.classes))
model = train_model(model, train_loader, epochs=5)

# -------------------------
# Evaluate on test set
# -------------------------
acc, f1, prec, rec, roc_auc, cm = evaluate_model(model, test_loader)
print(f"\nTest Results → Acc: {acc:.3f}, F1: {f1:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, ROC-AUC: {roc_auc}")
print("Confusion Matrix:\n", cm)

# -------------------------
# Save the trained model
# -------------------------
torch.save(model.state_dict(), "best_densenet121_model.pth")
print("\n Model saved to best_densenet121_model.pth")
