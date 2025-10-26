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

X_train = pd.read_csv("output/X_train.csv")
X_test = pd.read_csv("output/X_test.csv")
y_train = pd.read_csv("output/y_train.csv").squeeze()  
y_test = pd.read_csv("output/y_test.csv").squeeze()

print("Data loaded successfully")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print("Distribusi y_train:")
print(y_train.value_counts(normalize=True))

rf_model = RandomForestClassifier(random_state=42)

logreg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=5000))
])

svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(kernel='rbf', probability=True, class_weight='balanced'))
])

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
lgbm_model = LGBMClassifier(random_state=42)

knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier(n_neighbors=5))
])

param_grids = {
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    },
    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1, 10],
        "model__solver": ["lbfgs", "liblinear"]
    },
    "SVM": {
        "model__C": [0.1, 1, 10],
        "model__gamma": ["scale", "auto"]
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.1, 0.2]
    },
    "LightGBM": {
        "n_estimators": [100, 200],
        "max_depth": [-1, 10, 20],
        "learning_rate": [0.01, 0.1, 0.2]
    },
    "KNN": {
        "model__n_neighbors": [3, 5, 7, 9],
        "model__weights": ["uniform", "distance"]
    }
}

cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def tune_model_once(model, name, X_train, y_train):
    if name not in param_grids:
        print(f"Tidak ada param_grid untuk {name}, pakai default model.")
        return model

    print(f"\nTuning {name} ...")
    param_grid = param_grids[name]
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1',
        cv=cv_inner,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    print(f"Best params for {name}: {grid_search.best_params_}")
    return grid_search.best_estimator_

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

models = [
    (rf_model, "Random Forest"),
    (logreg_pipeline, "Logistic Regression"),
    (svm_pipeline, "SVM"),
    (xgb_model, "XGBoost"),
    (lgbm_model, "LightGBM"),
    (knn_pipeline, "KNN")
]

tuned_models = []
for model, name in models:
    best_model = tune_model_once(model, name, X_train, y_train)
    tuned_models.append((best_model, name))

results_cv = []
for model, name in tuned_models:
    result = evaluate_model_cv(model, name, X_train, y_train)
    results_cv.append(result)

results_cv_df = pd.DataFrame(results_cv).sort_values(by="F1 Score", ascending=False).reset_index(drop=True)
print("\nHasil Akhir Setelah Hyperparameter Tuning:")
print(results_cv_df)

joblib.dump(tuned_models[0][0], "best_model.pkl")
print("\nModel terbaik berhasil disimpan")

best_model = tuned_models[0][0]
y_pred_final = best_model.predict(X_test)

plt.close('all')  # tutup semua figure lama
plt.figure(figsize=(6, 4))  # bikin figure baru biar clean
print("\n=== Final Evaluation on Test Set ===")
print(classification_report(y_test, y_pred_final))
sns.heatmap(confusion_matrix(y_test, y_pred_final), annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {tuned_models[0][1]} (Best Model)')
plt.show()