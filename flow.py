
# ============================================================
# BANK MARKETING DATASET CLEANING + SCIKIT-LEARN CLASSIFICATION
# Dataset: Bank_Marketing_Dataset.csv
# Target: Class
# Model: Scikit-learn Neural Network (MLPClassifier)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neural_network import MLPClassifier

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv("Bank_Marketing_Dataset.csv")

print("Dataset Loaded Successfully")
print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nDataset Description:")
print(df.describe())

# ------------------------------------------------------------
# 2. DATA CLEANING
# ------------------------------------------------------------

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Remove extra spaces from categorical columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# Handle missing values
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna("unknown")

for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDuplicates After Cleaning:", df.duplicated().sum())

# ------------------------------------------------------------
# 3. RENAME COLUMNS
# ------------------------------------------------------------

column_mapping = {
    "V1": "age",
    "V2": "job",
    "V3": "marital",
    "V4": "education",
    "V5": "default",
    "V6": "balance",
    "V7": "housing",
    "V8": "loan",
    "V9": "contact",
    "V10": "day",
    "V11": "month",
    "V12": "duration",
    "V13": "campaign",
    "V14": "pdays",
    "V15": "previous",
    "V16": "poutcome",
    "Class": "target"
}

df.rename(columns=column_mapping, inplace=True)

print("\nRenamed Columns:")
print(df.columns)

# ------------------------------------------------------------
# 4. TARGET ENCODING
# ------------------------------------------------------------

# Class 1 = No subscription
# Class 2 = Yes subscription
# Convert to binary target:
# 0 = No
# 1 = Yes

df["target"] = df["target"].map({
    1: 0,
    2: 1
})

print("\nTarget Distribution:")
print(df["target"].value_counts())

# ------------------------------------------------------------
# 5. SAVE CLEANED DATASET
# ------------------------------------------------------------

df.to_csv("Bank_Marketing_Cleaned.csv", index=False)

print("\nCleaned Dataset Saved Successfully")

# ------------------------------------------------------------
# 6. FEATURE SELECTION
# ------------------------------------------------------------

# IMPORTANT:
# Duration is removed because call duration is only known
# after the call ends. Including it can cause data leakage
# in a realistic prediction made before contacting customers.

X = df.drop(columns=["target", "duration"])
y = df["target"]

print("\nInput Features:")
print(X.columns)

print("\nTarget:")
print(y.name)

# ------------------------------------------------------------
# 7. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ------------------------------------------------------------

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    include="object"
).columns.tolist()

print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)

# ------------------------------------------------------------
# 8. TRAIN TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ------------------------------------------------------------
# 9. DATA PREPROCESSING
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform testing data
X_test_processed = preprocessor.transform(X_test)

# MLPClassifier requires dense arrays. ColumnTransformer may already return a
# NumPy array, so only call toarray() when the result is sparse.
if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()
if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()

X_train_processed = np.asarray(X_train_processed, dtype="float32")
X_test_processed = np.asarray(X_test_processed, dtype="float32")

y_train = y_train.to_numpy().astype("int32")
y_test = y_test.to_numpy().astype("int32")

print("\nPreprocessed Training Shape:", X_train_processed.shape)
print("Preprocessed Testing Shape:", X_test_processed.shape)

# ------------------------------------------------------------
# 10. CLASS WEIGHT HANDLING
# ------------------------------------------------------------

# The dataset has more No-subscription records than
# Yes-subscription records. Class weights help the model
# pay attention to the minority class.

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes, weights)
}

print("\nClass Weights:")
print(class_weights)

# ------------------------------------------------------------
# 11. SCIKIT-LEARN NEURAL NETWORK MODEL
# ------------------------------------------------------------

# MLPClassifier has built-in validation-based early stopping. Scikit-learn's
# MLP currently has no class_weight option, so use per-row sample weights when
# supported by the installed version; older versions train without weighting.
model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    batch_size=32,
    learning_rate_init=0.001,
    max_iter=30,
    early_stopping=True,
    validation_fraction=0.20,
    n_iter_no_change=5,
    random_state=42,
    verbose=True
)

print("\nScikit-learn MLP model:")
print(model)

print("\nTraining neural network...")
sample_weights = np.array([class_weights[int(label)] for label in y_train])
import inspect

if "sample_weight" in inspect.signature(model.fit).parameters:
    model.fit(X_train_processed, y_train, sample_weight=sample_weights)
    print("Class-balanced sample weights applied.")
else:
    model.fit(X_train_processed, y_train)
    print("This scikit-learn version does not support MLP sample weights; trained without them.")

# ------------------------------------------------------------
# 14. MODEL EVALUATION
# ------------------------------------------------------------

test_loss = model.loss_
print("\nFinal training loss:", test_loss)

# ------------------------------------------------------------
# 15. PREDICTION
# ------------------------------------------------------------

y_pred_probability = model.predict_proba(X_test_processed)[:, 1]

# Threshold 0.50
y_pred = (y_pred_probability >= 0.50).astype(int).flatten()

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nFinal Model Accuracy: {:.2f}%".format(
    accuracy * 100
))

# ------------------------------------------------------------
# 16. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Subscription",
            "Yes Subscription"
        ],
        zero_division=0
    )
)

# ------------------------------------------------------------
# 17. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=[
        "No Subscription",
        "Yes Subscription"
    ],
    yticklabels=[
        "No Subscription",
        "Yes Subscription"
    ]
)

plt.title("Bank Marketing - Confusion Matrix")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 18. VALIDATION ACCURACY GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

if model.validation_scores_:
    plt.plot(model.validation_scores_, label="Validation Accuracy")
    plt.legend()
else:
    print("Validation accuracy history is unavailable.")

plt.title("Scikit-learn Neural Network Validation Accuracy")
plt.xlabel("Training Iterations")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 19. TRAINING LOSS GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    model.loss_curve_,
    label="Training Loss"
)

plt.title("Scikit-learn Neural Network Training Loss")
plt.xlabel("Training Iterations")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 20. SAVE SCIKIT-LEARN MODEL
# ------------------------------------------------------------

joblib.dump(model, "Bank_Marketing_MLP_Model.pkl")

print("\nScikit-learn neural network saved successfully")

# ------------------------------------------------------------
# 21. SAVE PREPROCESSOR
# ------------------------------------------------------------

joblib.dump(
    preprocessor,
    "Bank_Marketing_Preprocessor.pkl"
)

print("Preprocessor Saved Successfully")

# ------------------------------------------------------------
# 22. FINAL OUTPUT
# ------------------------------------------------------------

print("\n========================================")
print("BANK MARKETING SCIKIT-LEARN PROJECT")
print("========================================")

print("Cleaned Dataset: Bank_Marketing_Cleaned.csv")
print("Neural Network Model: Bank_Marketing_MLP_Model.pkl")
print("Preprocessor: Bank_Marketing_Preprocessor.pkl")

print("Model Accuracy: {:.2f}%".format(
    accuracy * 100
))

print("\nPROJECT COMPLETED SUCCESSFULLY!")
