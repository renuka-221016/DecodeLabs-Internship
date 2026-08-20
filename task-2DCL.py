# ============================================================
# DECODELABS - DATA SCIENCE PROJECT 2
# Supervised Learning - Fraud Detection Pipeline
# Batch: 2026
# ============================================================

# ============================================================
# 1. INSTALL REQUIRED LIBRARY
# ============================================================

#!pip -q install imbalanced-learn


# ============================================================
# 2. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

sns.set_style("whitegrid")

pd.set_option("display.max_columns", None)


print("=" * 75)
print("DECODELABS - DATA SCIENCE PROJECT 2")
print("SUPERVISED LEARNING - FRAUD DETECTION PIPELINE")
print("=" * 75)


# ============================================================
# 3. LOAD DATASET
# ============================================================

# Credit Card Fraud Detection dataset
# This dataset contains transactions where:
# 0 = Legitimate transaction
# 1 = Fraudulent transaction

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

df = pd.read_csv(url)

print("\nDataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)


# ============================================================
# 4. BASIC DATA EXPLORATION
# ============================================================

print("\n" + "=" * 75)
print("BASIC DATA EXPLORATION")
print("=" * 75)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 75)
print("MISSING VALUE ANALYSIS")
print("=" * 75)

missing_values = df.isnull().sum()

print("\nMissing values:")
print(missing_values)

print("\nTotal missing values:", df.isnull().sum().sum())


# ============================================================
# 6. CHECK DUPLICATES
# ============================================================

print("\n" + "=" * 75)
print("DUPLICATE ANALYSIS")
print("=" * 75)

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)

if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicate rows removed.")

print("Dataset shape after duplicate removal:", df.shape)


# ============================================================
# 7. IDENTIFY TARGET VARIABLE
# ============================================================

target = "Class"

print("\nTarget variable:", target)

print("\nTarget distribution:")
print(df[target].value_counts())


# ============================================================
# 8. ANALYZE CLASS IMBALANCE
# ============================================================

print("\n" + "=" * 75)
print("CLASS IMBALANCE ANALYSIS")
print("=" * 75)

class_counts = df[target].value_counts()

class_percentages = (
    df[target]
    .value_counts(normalize=True)
    * 100
)

print("\nClass counts:")
print(class_counts)

print("\nClass percentages:")
print(class_percentages.round(4))


# Plot class distribution

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x=target
)

plt.title("Fraud vs Legitimate Transactions")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")

plt.xticks(
    [0, 1],
    ["Legitimate", "Fraud"]
)

plt.show()


# ============================================================
# 9. VISUALIZE TRANSACTION AMOUNT
# ============================================================

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="Amount",
    bins=50,
    kde=True
)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.show()


# ============================================================
# 10. FRAUD VS TRANSACTION AMOUNT
# ============================================================

plt.figure(figsize=(10, 5))

sns.boxplot(
    data=df,
    x="Class",
    y="Amount"
)

plt.title("Transaction Amount by Class")
plt.xlabel("Class")
plt.ylabel("Amount")

plt.xticks(
    [0, 1],
    ["Legitimate", "Fraud"]
)

plt.show()


# ============================================================
# 11. PREPARE FEATURES AND TARGET
# ============================================================

print("\n" + "=" * 75)
print("FEATURE AND TARGET PREPARATION")
print("=" * 75)

X = df.drop(columns=[target])

y = df[target]

print("\nFeature shape:", X.shape)

print("Target shape:", y.shape)


# ============================================================
# 12. TRAIN TEST SPLIT
# ============================================================

# IMPORTANT:
# We split BEFORE applying SMOTE.
# This prevents data leakage.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining set shape:", X_train.shape)

print("Testing set shape:", X_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# ============================================================
# 13. CREATE SMOTE + LOGISTIC REGRESSION PIPELINE
# ============================================================

print("\n" + "=" * 75)
print("LOGISTIC REGRESSION WITH SMOTE")
print("=" * 75)

logistic_pipeline = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),

        (
            "smote",
            SMOTE(
                random_state=42
            )
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


# ============================================================
# 14. TRAIN LOGISTIC REGRESSION
# ============================================================

logistic_pipeline.fit(
    X_train,
    y_train
)

print("Logistic Regression training completed.")


# ============================================================
# 15. LOGISTIC REGRESSION PREDICTIONS
# ============================================================

logistic_predictions = logistic_pipeline.predict(
    X_test
)

logistic_probabilities = logistic_pipeline.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 16. LOGISTIC REGRESSION EVALUATION
# ============================================================

logistic_precision = precision_score(
    y_test,
    logistic_predictions,
    zero_division=0
)

logistic_recall = recall_score(
    y_test,
    logistic_predictions,
    zero_division=0
)

logistic_roc_auc = roc_auc_score(
    y_test,
    logistic_probabilities
)

print("\nLogistic Regression Results:")

print(
    "Precision:",
    round(logistic_precision, 4)
)

print(
    "Recall:",
    round(logistic_recall, 4)
)

print(
    "ROC-AUC:",
    round(logistic_roc_auc, 4)
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        logistic_predictions,
        target_names=[
            "Legitimate",
            "Fraud"
        ],
        zero_division=0
    )
)


# ============================================================
# 17. LOGISTIC REGRESSION CONFUSION MATRIX
# ============================================================

logistic_cm = confusion_matrix(
    y_test,
    logistic_predictions
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    logistic_cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Logistic Regression - Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.xticks(
    [0.5, 1.5],
    ["Legitimate", "Fraud"]
)

plt.yticks(
    [0.5, 1.5],
    ["Legitimate", "Fraud"]
)

plt.show()


# ============================================================
# 18. RANDOM FOREST WITH SMOTE
# ============================================================

print("\n" + "=" * 75)
print("RANDOM FOREST WITH SMOTE")
print("=" * 75)

random_forest_pipeline = Pipeline(
    steps=[
        (
            "smote",
            SMOTE(
                random_state=42
            )
        ),

        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# 19. TRAIN RANDOM FOREST
# ============================================================

random_forest_pipeline.fit(
    X_train,
    y_train
)

print("Random Forest training completed.")


# ============================================================
# 20. RANDOM FOREST PREDICTIONS
# ============================================================

rf_predictions = random_forest_pipeline.predict(
    X_test
)

rf_probabilities = random_forest_pipeline.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 21. RANDOM FOREST EVALUATION
# ============================================================

rf_precision = precision_score(
    y_test,
    rf_predictions,
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    rf_predictions,
    zero_division=0
)

rf_roc_auc = roc_auc_score(
    y_test,
    rf_probabilities
)

print("\nRandom Forest Results:")

print(
    "Precision:",
    round(rf_precision, 4)
)

print(
    "Recall:",
    round(rf_recall, 4)
)

print(
    "ROC-AUC:",
    round(rf_roc_auc, 4)
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        rf_predictions,
        target_names=[
            "Legitimate",
            "Fraud"
        ],
        zero_division=0
    )
)


# ============================================================
# 22. RANDOM FOREST CONFUSION MATRIX
# ============================================================

rf_cm = confusion_matrix(
    y_test,
    rf_predictions
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    rf_cm,
    annot=True,
    fmt="d",
    cmap="Greens"
)

plt.title("Random Forest - Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.xticks(
    [0.5, 1.5],
    ["Legitimate", "Fraud"]
)

plt.yticks(
    [0.5, 1.5],
    ["Legitimate", "Fraud"]
)

plt.show()


# ============================================================
# 23. HYPERPARAMETER TUNING - RANDOM FOREST
# ============================================================

print("\n" + "=" * 75)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 75)

rf_tuning_pipeline = Pipeline(
    steps=[
        (
            "smote",
            SMOTE(
                random_state=42
            )
        ),

        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


param_grid = {
    "classifier__n_estimators": [
        100,
        200
    ],

    "classifier__max_depth": [
        None,
        10,
        20
    ],

    "classifier__min_samples_split": [
        2,
        5
    ]
}


cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)


grid_search = GridSearchCV(
    estimator=rf_tuning_pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=1
)


grid_search.fit(
    X_train,
    y_train
)


print("\nBest Parameters:")
print(grid_search.best_params__)

print(
    "\nBest Cross-Validation ROC-AUC:",
    round(
        grid_search.best_score_,
        4
    )
)


# ============================================================
# 24. EVALUATE TUNED RANDOM FOREST
# ============================================================

best_rf = grid_search.best_estimator_

tuned_rf_predictions = best_rf.predict(
    X_test
)

tuned_rf_probabilities = best_rf.predict_proba(
    X_test
)[:, 1]


tuned_rf_precision = precision_score(
    y_test,
    tuned_rf_predictions,
    zero_division=0
)

tuned_rf_recall = recall_score(
    y_test,
    tuned_rf_predictions,
    zero_division=0
)

tuned_rf_roc_auc = roc_auc_score(
    y_test,
    tuned_rf_probabilities
)


print("\n" + "=" * 75)
print("TUNED RANDOM FOREST RESULTS")
print("=" * 75)

print(
    "Precision:",
    round(tuned_rf_precision, 4)
)

print(
    "Recall:",
    round(tuned_rf_recall, 4)
)

print(
    "ROC-AUC:",
    round(tuned_rf_roc_auc, 4)
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        tuned_rf_predictions,
        target_names=[
            "Legitimate",
            "Fraud"
        ],
        zero_division=0
    )
)


# ============================================================
# 25. TUNED RANDOM FOREST CONFUSION MATRIX
# ============================================================

tuned_rf_cm = confusion_matrix(
    y_test,
    tuned_rf_predictions
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    tuned_rf_cm,
    annot=True,
    fmt="d",
    cmap="Oranges"
)

plt.title(
    "Tuned Random Forest - Confusion Matrix"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.xticks(
    [0.5, 1.5],
    ["Legitimate", "Fraud"]
)

plt.yticks(
    [0.5, 1.5],
    ["Legitimate", "Fraud"]
)

plt.show()


# ============================================================
# 26. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

comparison = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "Tuned Random Forest"
        ],

        "Precision": [
            logistic_precision,
            rf_precision,
            tuned_rf_precision
        ],

        "Recall": [
            logistic_recall,
            rf_recall,
            tuned_rf_recall
        ],

        "ROC-AUC": [
            logistic_roc_auc,
            rf_roc_auc,
            tuned_rf_roc_auc
        ]
    }
)

print(
    comparison.round(4)
)


# ============================================================
# 27. VISUALIZE MODEL COMPARISON
# ============================================================

comparison_plot = comparison.set_index(
    "Model"
)

comparison_plot.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title(
    "Fraud Detection Model Comparison"
)

plt.ylabel("Score")

plt.xlabel("Model")

plt.xticks(
    rotation=0
)

plt.ylim(
    0,
    1
)

plt.legend(
    title="Metrics"
)

plt.tight_layout()

plt.show()


# ============================================================
# 28. ROC CURVE COMPARISON
# ============================================================

logistic_fpr, logistic_tpr, _ = roc_curve(
    y_test,
    logistic_probabilities
)

rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_probabilities
)

tuned_rf_fpr, tuned_rf_tpr, _ = roc_curve(
    y_test,
    tuned_rf_probabilities
)


plt.figure(figsize=(10, 7))

plt.plot(
    logistic_fpr,
    logistic_tpr,
    label=f"Logistic Regression (AUC = {logistic_roc_auc:.4f})"
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest (AUC = {rf_roc_auc:.4f})"
)

plt.plot(
    tuned_rf_fpr,
    tuned_rf_tpr,
    label=f"Tuned Random Forest (AUC = {tuned_rf_roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Random Classifier"
)

plt.title("ROC Curve Comparison")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend()

plt.grid()

plt.show()


# ============================================================
# 29. PRECISION-RECALL CURVE
# ============================================================

precision_values, recall_values, _ = precision_recall_curve(
    y_test,
    tuned_rf_probabilities
)

plt.figure(figsize=(9, 6))

plt.plot(
    recall_values,
    precision_values,
    color="purple"
)

plt.title(
    "Precision-Recall Curve - Tuned Random Forest"
)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.grid()

plt.show()


# ============================================================
# 30. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

# Extract Random Forest model from pipeline

rf_model = best_rf.named_steps[
    "classifier"
]

feature_importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": rf_model.feature_importances_
    }
)

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print("\n" + "=" * 75)
print("TOP 15 IMPORTANT FEATURES")
print("=" * 75)

print(
    feature_importance.head(15)
)


# Plot feature importance

plt.figure(figsize=(10, 7))

sns.barplot(
    data=feature_importance.head(15),
    x="Importance",
    y="Feature",
    color="steelblue"
)

plt.title(
    "Top 15 Feature Importances - Tuned Random Forest"
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.tight_layout()

plt.show()


# ============================================================
# 31. FINAL MODEL SELECTION
# ============================================================

# We select the tuned Random Forest based on ROC-AUC.
# Accuracy is deliberately NOT used as the main metric.

models = {
    "Logistic Regression": logistic_roc_auc,
    "Random Forest": rf_roc_auc,
    "Tuned Random Forest": tuned_rf_roc_auc
}

best_model_name = max(
    models,
    key=models.get
)

print("\n" + "=" * 75)
print("FINAL MODEL SELECTION")
print("=" * 75)

print(
    "Best model based on ROC-AUC:",
    best_model_name
)

print(
    "Best ROC-AUC:",
    round(
        models[best_model_name],
        4
    )
)


# ============================================================
# 32. SAVE FINAL MODEL
# ============================================================

joblib.dump(
    best_rf,
    "fraud_detection_random_forest_model.pkl"
)

print(
    "\nFinal model saved as:"
)

print(
    "fraud_detection_random_forest_model.pkl"
)


# ============================================================
# 33. SAVE MODEL RESULTS
# ============================================================

comparison.to_csv(
    "fraud_model_comparison.csv",
    index=False
)

feature_importance.to_csv(
    "fraud_feature_importance.csv",
    index=False
)

print(
    "Model comparison saved as: fraud_model_comparison.csv"
)

print(
    "Feature importance saved as: fraud_feature_importance.csv"
)


# ============================================================
# 34. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("PROJECT 2 COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nTechniques implemented:")

print("1. Exploratory Data Analysis")
print("2. Class imbalance analysis")
print("3. Train-test split")
print("4. SMOTE oversampling")
print("5. Logistic Regression")
print("6. Random Forest")
print("7. Hyperparameter tuning")
print("8. Precision evaluation")
print("9. Recall evaluation")
print("10. ROC-AUC evaluation")
print("11. Confusion matrix analysis")
print("12. ROC curve analysis")
print("13. Precision-Recall curve")
print("14. Feature importance analysis")

print("\nAccuracy was deliberately NOT used as the primary evaluation metric.")

print("\nGenerated files:")

print("- fraud_detection_random_forest_model.pkl")
print("- fraud_model_comparison.csv")
print("- fraud_feature_importance.csv")

print("\n" + "=" * 75)
print("FRAUD DETECTION PIPELINE COMPLETE")
print("=" * 75)
