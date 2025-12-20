#setup
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm

#Data preprocessing
data_dir =  ""

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)
class_names = dataset.classes

#Load DenseNet121 for feature extraction
#We remove the classification head (softmax) and keep up to the global average pooling (GAP) layer, plus optionally a small dense projection.
# Load pre-trained DenseNet121
base_model = models.densenet121(pretrained=True)

# Remove the final classifier
# DenseNet121: features → ReLU → AdaptiveAvgPool2d → Flatten → classifier
feature_extractor = nn.Sequential(
    base_model.features,
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)

# Optionally: add a small projection layer (as in your architecture)
embedding_dim = 512  # e.g. to reduce dimensionality
projector = nn.Linear(1024, embedding_dim)

# Set to eval mode
feature_extractor.eval()
projector.eval()

#Extract embeddings for all images
all_embeddings = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(loader, desc="Extracting features"):
        feats = feature_extractor(images)
        embeddings = projector(feats)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

# Convert to NumPy arrays
X_visual = np.concatenate(all_embeddings)
y_visual = np.concatenate(all_labels)

#Save for fusion with meteorological embeddings
np.savez("visual_embeddings.npz", X_visual=X_visual, y_visual=y_visual)

#load the file
data = np.load("visual_embeddings.npz")
X_visual = data["X_visual"]
y_visual = data["y_visual"]

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

#Replace path with your dataset root
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
