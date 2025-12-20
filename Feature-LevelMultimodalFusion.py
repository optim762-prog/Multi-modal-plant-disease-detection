import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load visual embeddings
vis_data = np.load("visual_embeddings.npz")
X_visual = vis_data["X_visual"]
y = vis_data["y_visual"]

# Load meteorological embeddings
met_data = np.load("meteo_embeddings.npz")
X_meteo = met_data["X_meteo"]
y_meteo = met_data["y_meteo"]

print("Unique visual labels:", np.unique(y))
print("Unique meteo labels:", np.unique(y_meteo))

# -------------------------------------------------------
# Filter visual samples to keep only those with meteorological classes (0, 1)
# -------------------------------------------------------
mask_common = np.isin(y, np.unique(y_meteo))
X_visual = X_visual[mask_common]
y = y[mask_common]

print(f"Filtered visual samples: {X_visual.shape[0]} remain")

# -------------------------------------------------------
# Align samples by class
# -------------------------------------------------------
X_meteo_matched = []
for cls in np.unique(y):
    idx_vis = np.where(y == cls)[0]
    idx_meteo = np.where(y_meteo == cls)[0]

    if len(idx_meteo) == 0:
        print(f"⚠️ No meteo data for class {cls}, skipping.")
        continue

    n = len(idx_vis)
    chosen_idx = np.random.choice(idx_meteo, size=n, replace=True)
    X_meteo_matched.append(X_meteo[chosen_idx])

X_meteo_matched = np.vstack(X_meteo_matched)

print("\nAfter mapping:")
print(f"  Visual samples: {X_visual.shape}")
print(f"  Meteo matched:  {X_meteo_matched.shape}")

# -------------------------------------------------------
# Normalize meteorological embeddings
# -------------------------------------------------------
scaler = StandardScaler()
X_meteo_scaled = scaler.fit_transform(X_meteo_matched)

# -------------------------------------------------------
# Train/test split (aligned and consistent)
# -------------------------------------------------------
Xv_train, Xv_test, Xm_train, Xm_test, y_train, y_test = train_test_split(
    X_visual, X_meteo_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("\n Final shapes:")
print(f"  Xv_train: {Xv_train.shape}, Xm_train: {Xm_train.shape}")
print(f"  y_train:  {y_train.shape}")

#Define the Multimodal Fusion Network
#This network:Projects each modality to a latent space
#Concatenates them
#Passes through dense layers with dropout and batch norm
#Outputs multi-class predictions

class MultimodalFusionNet(nn.Module):
    def __init__(self, dim_vis=512, dim_met=6, fusion_dim=256, num_classes=3):
        super(MultimodalFusionNet, self).__init__()

        # Visual branch projection
        self.vis_proj = nn.Sequential(
            nn.Linear(dim_vis, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Meteorological branch projection
        self.met_proj = nn.Sequential(
            nn.Linear(dim_met, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Fusion and classification head
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(fusion_dim, num_classes)
        )

    def forward(self, x_vis, x_met):
        vis = self.vis_proj(x_vis)
        met = self.met_proj(x_met)
        fused = torch.cat((vis, met), dim=1)
        out = self.fusion(fused)
        return out

# Prepare data tensors
Xv_train_t = torch.FloatTensor(Xv_train)
Xm_train_t = torch.FloatTensor(Xm_train)
y_train_t = torch.LongTensor(y_train)

Xv_test_t = torch.FloatTensor(Xv_test)
Xm_test_t = torch.FloatTensor(Xm_test)
y_test_t = torch.LongTensor(y_test)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalFusionNet(dim_vis=Xv_train.shape[1],
                            dim_met=Xm_train.shape[1],
                            num_classes=len(np.unique(y_train))).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

#training loop
epochs = 30
batch_size = 32
n_batches = len(Xv_train_t) // batch_size

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        vis_batch = Xv_train_t[start:end].to(device)
        met_batch = Xm_train_t[start:end].to(device)
        y_batch = y_train_t[start:end].to(device)

        optimizer.zero_grad()
        outputs = model(vis_batch, met_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/n_batches:.4f}")
	
#evaluation
model.eval()
with torch.no_grad():
    preds = model(Xv_test_t.to(device), Xm_test_t.to(device))
    predicted = torch.argmax(preds, dim=1).cpu().numpy()

acc = np.mean(predicted == y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

#With cross validation
import numpy as np

# --- Load pre-extracted embeddings ---
vis_data = np.load("visual_embeddings.npz")
X_visual = vis_data["X_visual"]
y_visual = vis_data["y_visual"]

met_data = np.load("meteo_embeddings.npz")
print("Available keys in meteo file:", met_data.files)
X_meteo = met_data["X_meteo"]
y_meteo = met_data["y_meteo"]

print(f"Visual: {X_visual.shape}, Meteo: {X_meteo.shape}")
print(f"Unique visual labels: {np.unique(y_visual)}")
print(f"Unique meteo labels: {np.unique(y_meteo)}")

#Step 2 — Match meteorological data per image by class
#We’ll randomly assign meteorological samples from the same class to each image.
from sklearn.preprocessing import StandardScaler

# Create class-wise mapping for meteorological embeddings
meteo_by_class = {c: np.where(y_meteo == c)[0] for c in np.unique(y_meteo)}

X_meteo_matched = []

for i, c in enumerate(y_visual):
    if c in meteo_by_class and len(meteo_by_class[c]) > 0:
        chosen_idx = np.random.choice(meteo_by_class[c])
        X_meteo_matched.append(X_meteo[chosen_idx])
    else:
        # if no meteorological data for that class, sample from a random available one
        random_class = np.random.choice(list(meteo_by_class.keys()))
        chosen_idx = np.random.choice(meteo_by_class[random_class])
        X_meteo_matched.append(X_meteo[chosen_idx])

X_meteo_matched = np.array(X_meteo_matched)

# --- Standardize ---
scaler = StandardScaler()
X_meteo_scaled = scaler.fit_transform(X_meteo_matched)

print("Shapes after alignment:")
print("Visual:", X_visual.shape)
print("Meteo matched:", X_meteo_scaled.shape)
print("Labels:", y_visual.shape)

# ==============================================
# Tracking all metrics across folds
# ==============================================
fold_metrics = {
    "accuracy": [],
    "macro_f1": [],
    "precision": [],
    "recall": [],
    "roc_auc_ovr": [],
    "loss": [],
    "inference_time_ms": [],
    "model_size_mb": []
}

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalNet(nn.Module):
    def __init__(self, visual_dim, meteo_dim, num_classes):
        super(MultiModalNet, self).__init__()

        # Visual branch
        self.visual_branch = nn.Sequential(
            nn.Linear(visual_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Meteorological branch
        self.meteo_branch = nn.Sequential(
            nn.Linear(meteo_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Fusion + classification head
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
			nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_visual, x_meteo):
        v = self.visual_branch(x_visual)
        m = self.meteo_branch(x_meteo)
        fused = torch.cat((v, m), dim=1)
        out = self.classifier(fused)
        return out
#evaluation
model.eval()
with torch.no_grad():
    preds = model(Xv_test_t.to(device), Xm_test_t.to(device))
    predicted = torch.argmax(preds, dim=1).cpu().numpy()

acc = np.mean(predicted == y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import numpy as np
import time
import os

# ===============================
# Assume these are already defined:
# X_visual, X_meteo_scaled, y_visual
# MultiModalNet class
# ===============================

Xv_all = X_visual        # Aligned visual embeddings
Xm_all = X_meteo_scaled  # Aligned meteorological embeddings
y_all = y_visual         

# Convert to tensors
Xv_all_t = torch.tensor(Xv_all, dtype=torch.float32)
Xm_all_t = torch.tensor(Xm_all, dtype=torch.float32)
y_all_t = torch.tensor(y_all, dtype=torch.long)

# K-Fold setup
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Training config
epochs = 30
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

# Metrics storage
metrics = {
    "accuracy": [],
    "macro_f1": [],
    "precision": [],
    "recall": [],
    "roc_auc_ovr": [],
    "confusion_matrices": [],
    "inference_time_ms": [],
    "model_size_mb": [],
}

for fold, (train_idx, val_idx) in enumerate(kf.split(Xv_all)):
    print(f"\n===== Fold {fold+1} / {n_splits} =====")

    # Split tensors
    Xv_train_t, Xv_val_t = Xv_all_t[train_idx], Xv_all_t[val_idx]
    Xm_train_t, Xm_val_t = Xm_all_t[train_idx], Xm_all_t[val_idx]
    y_train_t, y_val_t = y_all_t[train_idx], y_all_t[val_idx]

    # Initialize model for this fold
    model = MultiModalNet(Xv_all.shape[1], Xm_all.shape[1], num_classes=len(np.unique(y_all)))
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Training loop ---
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = len(Xv_train_t) // batch_size
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            vis_batch = Xv_train_t[start:end].to(device)
            met_batch = Xm_train_t[start:end].to(device)
            y_batch = y_train_t[start:end].to(device)

            optimizer.zero_grad()
            outputs = model(vis_batch, met_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # --- Evaluation ---
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs = model(Xv_val_t.to(device), Xm_val_t.to(device))
        inference_time = (time.time() - start_time) / len(Xv_val_t) * 1000  # ms/sample

        y_pred_classes = torch.argmax(outputs, dim=1).cpu().numpy()
        y_true = y_val_t.cpu().numpy()

    # Metrics computation
    acc = accuracy_score(y_true, y_pred_classes)
    macro_f1 = f1_score(y_true, y_pred_classes, average="macro")
    precision = precision_score(y_true, y_pred_classes, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred_classes, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred_classes)

    # ROC-AUC robust computation: only for present classes
    y_probs = torch.softmax(outputs, dim=1).cpu().numpy()
    present_classes = np.unique(y_true)
    roc_auc_per_class = []
    for cls in present_classes:
        y_true_bin = (y_true == cls).astype(int)
        roc_auc_per_class.append(roc_auc_score(y_true_bin, y_probs[:, cls]))
    roc_auc = np.mean(roc_auc_per_class)

    # Model size in MB
    torch.save(model.state_dict(), "temp_model.pth")
    model_size_mb = os.path.getsize("temp_model.pth") / 1e6
    os.remove("temp_model.pth")

    # Store metrics
    metrics["accuracy"].append(acc)
    metrics["macro_f1"].append(macro_f1)
    metrics["precision"].append(precision)
    metrics["recall"].append(recall)
    metrics["roc_auc_ovr"].append(roc_auc)
    metrics["confusion_matrices"].append(cm)
    metrics["inference_time_ms"].append(inference_time)
    metrics["model_size_mb"].append(model_size_mb)

    print(f"Fold {fold+1} Accuracy: {acc:.4f}, ROC-AUC: {roc_auc:.4f}")

# --- Final summary ---
print("\n===== 10-Fold Cross-Validation Results =====")
for k, v in metrics.items():
    if k != "confusion_matrices":
        print(f"{k}: {np.mean(v):.4f} ± {np.std(v):.4f}")
