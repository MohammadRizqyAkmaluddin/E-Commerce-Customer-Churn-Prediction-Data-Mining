import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# =============================
# Load data
# =============================
X_train = pd.read_csv("output/X_train.csv")
X_test = pd.read_csv("output/X_test.csv")
y_train = pd.read_csv("output/y_train.csv").squeeze()
y_test = pd.read_csv("output/y_test.csv").squeeze()

print("Data loaded successfully")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print("Distribusi y_train:")
print(y_train.value_counts(normalize=True))

# =============================
# Definisikan pipeline tiap model
# =============================
logreg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=5000, random_state=42))
])

svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(probability=True, class_weight='balanced', random_state=42))
])

knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier())
])

rf_model = RandomForestClassifier(random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
lgbm_model = LGBMClassifier(random_state=42)

# =============================
# Param grids (efisien dan aman)
# =============================
param_grids = {
    "Logistic Regression": {
        "model__C": [0.5, 1, 2],
        "model__solver": ["lbfgs"],
        "model__penalty": ["l2"]
    },

    "SVM": {
        "model__C": [0.1, 0.5, 1],
        "model__kernel": ["rbf", "linear"],
        "model__gamma": ["scale"]
    },

    "KNN": {
        "model__n_neighbors": [7, 9, 11],
        "model__weights": ["uniform", "distance"]
    },

    "Random Forest": {
        "n_estimators": [100],
        "max_depth": [4, 6],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [2, 4],
        "max_features": ["sqrt"]
    },

    "XGBoost": {
        "n_estimators": [100],
        "max_depth": [3, 4],
        "learning_rate": [0.05],
        "subsample": [0.8],
        "colsample_bytree": [0.8]
    },

    "LightGBM": {
        "n_estimators": [100],
        "max_depth": [4, 6],
        "num_leaves": [15, 25],
        "learning_rate": [0.05],
        "subsample": [0.7, 0.8]
    }
}


# =============================
# Fungsi untuk tuning model
# =============================
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def tune_model_once(model, name, X_train, y_train):
    if name not in param_grids:
        print(f"Tidak ada param_grid untuk {name}, pakai default model.")
        return model

    print(f"\n🔍 Tuning {name} ...")
    param_grid = param_grids[name]

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    print(f"✅ Best params for {name}: {grid_search.best_params_}")
    return grid_search.best_estimator_

# =============================
# Evaluasi dengan CV 5-Fold
# =============================
def evaluate_model_cv(model, name, X, y):
    acc_scores, prec_scores, rec_scores, f1_scores, roc_scores = [], [], [], [], []

    for train_idx, test_idx in cv_outer.split(X, y):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)

        # ROC AUC
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test_fold)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test_fold)
        else:
            y_score = y_pred

        acc_scores.append(accuracy_score(y_test_fold, y_pred))
        prec_scores.append(precision_score(y_test_fold, y_pred))
        rec_scores.append(recall_score(y_test_fold, y_pred))
        f1_scores.append(f1_score(y_test_fold, y_pred))
        roc_scores.append(roc_auc_score(y_test_fold, y_score))

    metrics = {
        "Model": name,
        "Accuracy": np.mean(acc_scores),
        "Precision": np.mean(prec_scores),
        "Recall": np.mean(rec_scores),
        "F1 Score": np.mean(f1_scores),
        "ROC AUC": np.mean(roc_scores)
    }

    print(f"\n=== {name} (CV 5-Fold) ===")
    for k, v in metrics.items():
        if k != "Model":
            print(f"{k}: {v:.4f}")

    return metrics

# =============================
# Jalankan tuning
# =============================
models = {
    "Logistic Regression": logreg_pipeline,
    "Random Forest": rf_model,
    "KNN": knn_pipeline,
    "SVM": svm_pipeline,
    "XGBoost": xgb_model,
    "LightGBM": lgbm_model
}

tuned_models = []
for name, model in models.items():
    best_model = tune_model_once(model, name, X_train, y_train)
    tuned_models.append((best_model, name))

# =============================
# Evaluasi hasil CV
# =============================
results_cv = []
for model, name in tuned_models:
    result = evaluate_model_cv(model, name, X_train, y_train)
    results_cv.append(result)

results_cv_df = pd.DataFrame(results_cv).sort_values(by="F1 Score", ascending=False).reset_index(drop=True)
print("\n📊 Hasil Akhir Setelah Hyperparameter Tuning:")
print(results_cv_df)

joblib.dump(tuned_models[0][0], "best_model.pkl")
print("\n💾 Model terbaik berhasil disimpan")

# =============================
# Check overfitting
# =============================
def check_overfitting(model, X_train, X_test, y_train, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    train_f1 = f1_score(y_train, y_pred_train)
    test_f1 = f1_score(y_test, y_pred_test)

    # Hitung ukuran data
    n_train = len(X_train)
    n_test = len(X_test)
    print(f"📊 Jumlah data: train={n_train}, test={n_test}")

    # Adaptif threshold — dataset kecil lebih toleran terhadap selisih skor
    if n_train < 1000:
        threshold = 0.10  # dataset kecil → boleh beda 10%
    elif n_train < 5000:
        threshold = 0.07
    else:
        threshold = 0.05  # dataset besar → harus stabil

    print(f"📏 Ambang toleransi overfit: {threshold:.2f}")

    # Tampilkan metrik
    print(f"Train Accuracy: {train_acc:.4f} | F1: {train_f1:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f} | F1: {test_f1:.4f}")
    print(f"Selisih Accuracy: {abs(train_acc - test_acc):.4f}")
    print(f"Selisih F1:       {abs(train_f1 - test_f1):.4f}")

    # Evaluasi overfitting
    if abs(train_acc - test_acc) > threshold or abs(train_f1 - test_f1) > threshold:
        print("⚠️  Model kemungkinan overfitting (di atas ambang toleransi)!")
    else:
        print("✅  Tidak terdeteksi overfitting signifikan.")


for model, name in tuned_models:
    print(f"\n=== Check Overfitting: {name} ===")
    model.fit(X_train, y_train)
    check_overfitting(model, X_train, X_test, y_train, y_test)
