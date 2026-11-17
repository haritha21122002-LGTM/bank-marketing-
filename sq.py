
# ============================================================
# BANK MARKETING DATASET CLEANING + MYSQL + ML CLASSIFICATION
# Dataset: Bank_Marketing_Dataset.csv
# Database: bank_marketing_db
# Table: bank_marketing_cleaned
# Model: Scikit-learn MLPClassifier
# ============================================================

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mysql.connector
import inspect

from mysql.connector import Error

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

# ============================================================
# 1. MYSQL DATABASE CONFIGURATION
# ============================================================

MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD")

DATABASE_NAME = "bank_marketing_db"
TABLE_NAME = "bank_marketing_cleaned"

# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("Bank_Marketing_Dataset.csv")

print("\nDataset Loaded Successfully")
print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nDataset Description:")
print(df.describe())

# ============================================================
# 3. DATA CLEANING
# ============================================================

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Remove extra spaces from categorical columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# Handle missing categorical values
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna("unknown")

# Handle missing numerical values
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicates:", df.duplicated().sum())

# ============================================================
# 4. RENAME COLUMNS
# ============================================================

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

# ============================================================
# 5. TARGET ENCODING
# ============================================================

# Class 1 = No subscription
# Class 2 = Yes subscription
# 0 = No
# 1 = Yes

df["target"] = pd.to_numeric(df["target"], errors="coerce")

df["target"] = df["target"].map({
    1: 0,
    2: 1
})

# Remove rows with invalid target values
df.dropna(subset=["target"], inplace=True)

df["target"] = df["target"].astype("int32")

print("\nTarget Distribution:")
print(df["target"].value_counts())

# ============================================================
# 6. SAVE CLEANED DATASET
# ============================================================

df.to_csv("Bank_Marketing_Cleaned.csv", index=False)

print("\nCleaned CSV Saved Successfully")

# ============================================================
# 7. MYSQL DATABASE CONNECTION
# ============================================================

try:

    # Connect to MySQL server
    connection = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD
    )

    cursor = connection.cursor()

    # Create database
    cursor.execute(
        f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}"
    )

    print("\nMySQL Database Created Successfully")

    cursor.close()
    connection.close()

    # Connect to the created database
    connection = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=DATABASE_NAME
    )

    cursor = connection.cursor()

    # ========================================================
    # 8. CREATE MYSQL TABLE
    # ========================================================

    column_definitions = []

    for col in df.columns:

        if col == "target":
            sql_type = "INT"

        elif pd.api.types.is_integer_dtype(df[col]):
            sql_type = "BIGINT"

        elif pd.api.types.is_float_dtype(df[col]):
            sql_type = "DOUBLE"

        else:
            sql_type = "VARCHAR(255)"

        column_definitions.append(
            f"`{col}` {sql_type}"
        )

    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS `{TABLE_NAME}` (
        id INT AUTO_INCREMENT PRIMARY KEY,
        {', '.join(column_definitions)}
    )
    """

    cursor.execute(create_table_query)

    print("\nMySQL Table Created Successfully")

    # ========================================================
    # 9. INSERT CLEANED DATA INTO MYSQL
    # ========================================================

    # Clear old data to avoid duplicate insertions
    cursor.execute(
        f"TRUNCATE TABLE `{TABLE_NAME}`"
    )

    columns = ", ".join(
        f"`{col}`" for col in df.columns
    )

    placeholders = ", ".join(
        ["%s"] * len(df.columns)
    )

    insert_query = f"""
    INSERT INTO `{TABLE_NAME}` ({columns})
    VALUES ({placeholders})
    """

    # Convert dataframe values into MySQL-compatible values
    data = []

    for row in df.itertuples(index=False, name=None):

        converted_row = []

        for value in row:

            if pd.isna(value):
                converted_row.append(None)

            elif isinstance(value, np.generic):
                converted_row.append(value.item())

            else:
                converted_row.append(value)

        data.append(tuple(converted_row))

    cursor.executemany(insert_query, data)

    connection.commit()

    print("\nCleaned Data Inserted into MySQL Successfully")
    print("Total Records Inserted:", cursor.rowcount)

    # Verify inserted records
    cursor.execute(
        f"SELECT COUNT(*) FROM `{TABLE_NAME}`"
    )

    total_records = cursor.fetchone()[0]

    print("Total Records in MySQL:", total_records)

    # Display sample records
    cursor.execute(
        f"SELECT * FROM `{TABLE_NAME}` LIMIT 5"
    )

    print("\nMySQL Sample Records:")

    for record in cursor.fetchall():
        print(record)

    cursor.close()
    connection.close()

except Error as e:

    print("\nMySQL Error:", e)

    if "connection" in locals() and connection.is_connected():
        connection.rollback()
        connection.close()

    raise

# ============================================================
# 10. FEATURE SELECTION
# ============================================================

# Duration is removed to avoid data leakage
# because call duration is known only after the call ends.

X = df.drop(columns=["target", "duration"])
y = df["target"]

print("\nInput Features:")
print(X.columns)

print("\nTarget:")
print(y.name)

# ============================================================
# 11. NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

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

# ============================================================
# 12. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ============================================================
# 13. DATA PREPROCESSING
# ============================================================

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

# Fit on training data only
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data
X_test_processed = preprocessor.transform(X_test)

# Convert sparse matrix into dense array
if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()

if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()

X_train_processed = np.asarray(
    X_train_processed,
    dtype="float32"
)

X_test_processed = np.asarray(
    X_test_processed,
    dtype="float32"
)

y_train = y_train.to_numpy().astype("int32")
y_test = y_test.to_numpy().astype("int32")

print("\nPreprocessed Training Shape:",
      X_train_processed.shape)

print("Preprocessed Testing Shape:",
      X_test_processed.shape)

# ============================================================
# 14. CLASS WEIGHT HANDLING
# ============================================================

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

sample_weights = np.array([
    class_weights[int(label)]
    for label in y_train
])

# ============================================================
# 15. MLP CLASSIFICATION MODEL
# ============================================================

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

print("\nTraining Neural Network...")

if "sample_weight" in inspect.signature(
    model.fit
).parameters:

    model.fit(
        X_train_processed,
        y_train,
        sample_weight=sample_weights
    )

    print("\nClass-balanced sample weights applied.")

else:

    model.fit(
        X_train_processed,
        y_train
    )

    print("\nModel trained without sample weights.")

# ============================================================
# 16. MODEL EVALUATION
# ============================================================

print("\nFinal Training Loss:", model.loss_)

# ============================================================
# 17. PREDICTION
# ============================================================

y_pred_probability = model.predict_proba(
    X_test_processed
)[:, 1]

y_pred = (
    y_pred_probability >= 0.50
).astype(int)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n========================================")
print("FINAL MODEL ACCURACY")
print("========================================")

print("Accuracy: {:.2f}%".format(
    accuracy * 100
))

# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

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

# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

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

# ============================================================
# 20. VALIDATION ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

if model.validation_scores_:

    plt.plot(
        model.validation_scores_,
        label="Validation Accuracy"
    )

    plt.legend()

plt.title("Neural Network Validation Accuracy")
plt.xlabel("Training Iterations")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 21. TRAINING LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    model.loss_curve_,
    label="Training Loss"
)

plt.title("Neural Network Training Loss")
plt.xlabel("Training Iterations")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 22. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "Bank_Marketing_MLP_Model.pkl"
)

print("\nML Model Saved Successfully")

# ============================================================
# 23. SAVE PREPROCESSOR
# ============================================================

joblib.dump(
    preprocessor,
    "Bank_Marketing_Preprocessor.pkl"
)

print("\nPreprocessor Saved Successfully")

# ============================================================
# 24. FINAL OUTPUT
# ============================================================

print("\n========================================")
print("BANK MARKETING ML + MYSQL PROJECT")
print("========================================")

print("Database:", DATABASE_NAME)
print("Table:", TABLE_NAME)

print("Cleaned Dataset: Bank_Marketing_Cleaned.csv")
print("ML Model: Bank_Marketing_MLP_Model.pkl")
print("Preprocessor: Bank_Marketing_Preprocessor.pkl")

print("Model Accuracy: {:.2f}%".format(
    accuracy * 100
))

print("\nPROJECT COMPLETED SUCCESSFULLY!")
