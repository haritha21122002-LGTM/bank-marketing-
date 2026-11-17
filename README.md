# Bank Marketing Analysis

This project cleans a bank marketing dataset and trains a scikit-learn neural network to predict whether a customer subscribes to a term deposit. It includes a standalone machine learning workflow and a workflow that also stores cleaned data in MySQL.

## Project files

- `flow.py` cleans the data, trains and evaluates the model, and saves the cleaned CSV, trained model, and fitted preprocessor.
- `sq.py` performs the same analysis and also creates a MySQL database and table and writes the cleaned records to MySQL.
- `Bank_Marketing_Dataset.csv` is the input dataset expected by both scripts. Data files and generated `.pkl` artifacts are excluded from this repository; provide the input CSV locally before running either script.

## Requirements

Python 3.9 or later is recommended. Install the required packages with:

python -m pip install pandas numpy matplotlib seaborn scikit-learn joblib mysql-connector-python

`mysql-connector-python` is only needed for `sq.py`.

## Prepare the dataset

Place `Bank_Marketing_Dataset.csv` in the project directory (the same directory as the scripts). The scripts expect the source columns used in the dataset, including `V1` through `V17` and `Class`.

## Run the machine learning workflow

python flow\.py

The script cleans and encodes the data, splits it into training and test sets, preprocesses numeric and categorical features, trains an `MLPClassifier`, and prints evaluation results. It excludes call `duration` from the model features to avoid using information that is only known after a call ends.

The run writes these files in the project directory:

- `Bank_Marketing_Cleaned.csv`
- `Bank_Marketing_MLP_Model.pkl`
- `Bank_Marketing_Preprocessor.pkl`

It also displays validation accuracy and training loss plots.

## Run with MySQL

Make sure a local MySQL server is running and the configured MySQL user can create databases and tables. Set the password in your environment; do not put it in the source code.

PowerShell:

$env = "your-mysql-password" python sq.py

Command Prompt:

set MYSQL_PASSWORD=your-mysql-password python sq.py

The script connects to `localhost` as user `root`, creates the `bank_marketing_db` database and `bank_marketing_cleaned` table if needed, and inserts the cleaned records. The host, user, database, and table names are configured near the top of `sq.py`.

## Notes

- Both scripts use `Bank_Marketing_Dataset.csv` as a local input file; it is not included in this repository.
- CSV outputs, serialized model files, and Python cache files are ignored by Git.
- Only load `.pkl` files from sources you trust. Pickle files can execute code when loaded.

**Aspiring Haritha K | Data Analyst & Data Scientist**
