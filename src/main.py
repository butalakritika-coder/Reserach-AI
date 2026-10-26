import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
# Create output directory if it does not exist
os.makedirs("results", exist_ok=True)
# Load dataset
file_path = "data/WBES-OS_India2026_Data/India-2026-AI follow-up data.dta"
df = pd.read_stata(file_path, convert_categoricals=False)

target = "ai_adopt1"

features = [
    "ai_b1a3",
    "ai_b1a4",
    "ai_b1b",
    "fatb7a2",
    "fatb7a3",
    "fatb7a4",
    "fatb7a5",
    "fatb8a3",
    "fat_b9a5",
    "e1a",
    "ai_a2a",
    "ai_a2b",
    "ai_a2c",
    "ai_a2d"
]

# Prepare data
data = df[features + [target]].copy()
data = data.mask(data < 0)
data = data.dropna()

X = data[features]
y = data[target].astype(int)

# Same split as Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Results
print("\nGRADIENT BOOSTING RESULTS")
print("=" * 50)

print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall   :", round(recall_score(y_test, y_pred), 4))
print("F1 Score :", round(f1_score(y_test, y_pred), 4))
print("ROC-AUC  :", round(roc_auc_score(y_test, y_prob), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nRANDOM FOREST FEATURE IMPORTANCE")
print("=" * 50)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

rf_model.fit(X_train, y_train)

importance = pd.Series(
    rf_model.feature_importances_,
    index=features
).sort_values(ascending=False)

print(importance)
import matplotlib.pyplot as plt

# Human-readable names
feature_names = {
    "fatb7a4": "Specialized finance software",
    "ai_b1b": "Share of workers using computers daily",
    "fatb7a3": "Mobile/digital finance platforms",
    "ai_a2b": "Professionals & technicians",
    "e1a": "Number of paid workers",
    "ai_a2a": "Number of managers",
    "ai_a2c": "Clerical & sales workers",
    "fatb8a3": "Social media/apps for supply chain",
    "ai_a2d": "Other workers",
    "fatb7a2": "Standard finance software",
    "fat_b9a5": "CRM software",
    "fatb7a5": "ERP software for finance",
    "ai_b1a3": "Website/social media presence",
    "ai_b1a4": "Messaging apps"
}

importance_named = importance.rename(index=feature_names)

plt.figure(figsize=(10, 7))
importance_named.sort_values().plot(kind="barh")

plt.xlabel("Random Forest Feature Importance")
plt.ylabel("Factor")
plt.title("Factors Associated with AI Adoption")

plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=300)
plt.show()
# ==============================
# MODEL COMPARISON
# ==============================

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting"
    ],
    "Accuracy": [0.8327, 0.8476, 0.8699, 0.8786],
    "Precision": [0.7767, 0.7850, 0.8190, 0.8208],
    "Recall": [0.7843, 0.8235, 0.8431, 0.8529],
    "F1 Score": [0.7805, 0.8038, 0.8309, 0.8365],
    "ROC-AUC": [0.8739, 0.8775, 0.9304, 0.9156]
})

print("\nMODEL COMPARISON")
print(results.to_string(index=False))

# Save table
results.to_csv("results/model_comparison.csv", index=False)

# Plot comparison
import matplotlib.pyplot as plt

results_plot = results.set_index("Model")

results_plot.plot(
    kind="bar",
    figsize=(11, 6)
)

plt.ylabel("Score")
plt.xlabel("Model")
plt.title("Comparison of Machine Learning Models")
plt.ylim(0.70, 1.00)
plt.xticks(rotation=0)
plt.legend(title="Evaluation Metric")
plt.tight_layout()

plt.savefig("results/model_comparison.png", dpi=300)
plt.show()
# ==============================
# CONFUSION MATRICES
# ==============================

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

confusion_matrices = {
    "Logistic Regression": np.array([
        [144, 23],
        [22, 80]
    ]),

    "Decision Tree": np.array([
        [144, 23],
        [18, 84]
    ]),

    "Random Forest": np.array([
        [148, 19],
        [16, 86]
    ]),

    "Gradient Boosting": np.array([
        [148, 19],
        [15, 87]
    ])
}

for name, cm in confusion_matrices.items():

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Adopter", "Adopter"]
    )

    disp.plot()

    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()

    safe_name = name.lower().replace(" ", "_")

    plt.savefig(
        f"results/confusion_matrix_{safe_name}.png",
        dpi=300
    )

    plt.show()
    # ==============================
# 5-FOLD CROSS-VALIDATION
# ==============================

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

cv_results = []

for name, model in cv_models.items():

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ]
    )

    cv_results.append({
        "Model": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1 Score": scores["test_f1"].mean(),
        "ROC-AUC": scores["test_roc_auc"].mean()
    })

cv_table = pd.DataFrame(cv_results)

print("\n5-FOLD CROSS-VALIDATION RESULTS")
print(cv_table.to_string(index=False))

cv_table.to_csv(
    "results/cross_validation_results.csv",
    index=False
)
# ==============================
# CROSS-VALIDATION COMPARISON CHART
# ==============================

cv_plot = cv_table.set_index("Model")

cv_plot.plot(
    kind="bar",
    figsize=(11, 6)
)

plt.ylabel("Mean Cross-Validation Score")
plt.xlabel("Model")
plt.title("5-Fold Cross-Validation Model Comparison")
plt.ylim(0.70, 1.00)
plt.xticks(rotation=0)
plt.legend(title="Evaluation Metric")
plt.tight_layout()

plt.savefig(
    "results/cross_validation_comparison.png",
    dpi=300
)

plt.show()
data = data.dropna()
# Save cleaned dataset
data.to_csv("results/cleaned_data.csv", index=False)

print("\nCleaned dataset saved to results/cleaned_data.csv")
print("\n--- DATASET VARIABLES ---")
print(df.columns.tolist())

print("\n--- FIRM SIZE SUMMARY ---")
print(df["e1a"].describe())

print("\n--- FIRM SIZE VALUES ---")
print(df["e1a"].value_counts().sort_index().head(30))