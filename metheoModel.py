# ======================================================
# Meteorological Data Classification with Cross-Validation
# ======================================================

import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, roc_auc_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# -----------------------------
# 1. Load and inspect the data
# -----------------------------
df = pd.read_csv("Disease_with_Weather.csv")

print("Data preview:")
print(df.head())

# -----------------------------------
# 2. Encode categorical and target columns
# -----------------------------------
label_encoder = LabelEncoder()
df["Disease"] = label_encoder.fit_transform(df["Disease"])

# -----------------------------------
# 3. Define features and target
# -----------------------------------
X = df.drop(["Disease", "Disease in number"], axis=1)
y = df["Disease in number"]

# -----------------------------------
# 4. Normalize features
# -----------------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------
# 5. Define models
# -----------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                         solver='adam', max_iter=300, random_state=42)
}

# -----------------------------------
# 6. Cross-validation setup
# -----------------------------------
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# -----------------------------------
# 7. Storage for results
# -----------------------------------
results = []

# -----------------------------------
# 8. Loop over models
# -----------------------------------
for name, model in models.items():
    print(f"\n   Evaluating {name} with {n_splits}-fold CV...")
    accs, f1s, precs, recs, aucs = [], [], [], [], []
    conf_matrices = []
    times = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # --- Training ---
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()

        # --- Inference ---
        y_pred = model.predict(X_val)
        inf_start = time.time()
        _ = model.predict(X_val[:100])  # test 100 samples for timing
        inf_end = time.time()

        # --- Metrics ---
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro")
        prec = precision_score(y_val, y_pred, average="macro")
        rec = recall_score(y_val, y_pred, average="macro")
        cm = confusion_matrix(y_val, y_pred)
        conf_matrices.append(cm)

        # --- ROC-AUC (binary case) ---
        try:
            y_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_proba)
        except Exception:
            auc = np.nan

        # --- Store metrics ---
        accs.append(acc)
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)
        aucs.append(auc)
        times.append((inf_end - inf_start) / 100 * 1000)  # ms per sample

    # --- Compute model size (MB) ---
    tmp_path = f"{name.replace(' ', '_')}.joblib"
    joblib.dump(model, tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)

    # --- Aggregate results ---
    results.append({
        "Model": name,
        "Accuracy (mean±std)": f"{np.mean(accs):.3f} ± {np.std(accs):.3f}",
        "Macro F1 (mean±std)": f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}",
        "Precision (mean±std)": f"{np.mean(precs):.3f} ± {np.std(precs):.3f}",
        "Recall (mean±std)": f"{np.mean(recs):.3f} ± {np.std(recs):.3f}",
        "ROC-AUC (mean±std)": f"{np.nanmean(aucs):.3f} ± {np.nanstd(aucs):.3f}",
        "Inference (ms/sample)": f"{np.mean(times):.2f}",
        "Model Size (MB)": f"{size_mb:.2f}"
    })

# -----------------------------------
# 9. Display results summary
# -----------------------------------
results_df = pd.DataFrame(results)
print("\n Cross-Validation Summary:")
print(results_df.sort_values(by="Macro F1 (mean±std)", ascending=False).reset_index(drop=True))

# -----------------------------------
# 10. Optional: Train best model on full dataset
# -----------------------------------
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
best_model.fit(X_scaled, y)
joblib.dump(best_model, "best_meteo_model.joblib")
