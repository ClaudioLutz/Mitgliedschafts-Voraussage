# Model Development Implementation

This document provides a detailed description of the model development and implementation for the lead generation project. The process is implemented in the `training_lead_generation_model.py` script, which automates the end-to-end workflow from data loading to generating a ranked list of leads.

## 1. Data Loading and Preparation

The model relies on two types of data snapshots, both loaded from a Microsoft SQL Server database.

### `load_modeling_data`
This function is responsible for creating the training dataset. It loads historical monthly snapshots of company data.

- **Time-Aware Labeling**: It only loads snapshots where the outcome (i.e., whether a company becomes a member) can be fully determined. This is controlled by the `HORIZON_MONTHS` parameter, ensuring that the snapshot date is at least that many months in the past.
- **Target Variable**: The target variable, `Target`, is created based on whether a company joins within `HORIZON_MONTHS` after the snapshot date.
- **Risk Set Construction**: The function carefully constructs the "risk set" to include only non-members at the time of the snapshot. This prevents data leakage by excluding companies that were already members or were founded after the snapshot.

### `load_current_snapshot`
This function loads the most recent snapshot of company data, which represents the current pool of potential leads to be scored by the model.

- **No Target Label**: This dataset does not have a `Target` label, as the outcomes are not yet known.
- **Inclusion Criteria**: It includes all non-member companies as of the current date.

## 2. Feature Engineering

After loading the data, a series of feature engineering and preprocessing steps are applied.

### Temporal Feature Engineering
The `temporal_feature_engineer` function creates time-dependent features:

- **`Company_Age_Years`**: Calculated as the difference between the snapshot year and the company's founding year.
- **`Has_Employees`**: A binary flag indicating if the company has a non-zero number of employees.
- **`Has_Revenue`**: A binary flag indicating if the company has non-zero revenue.

### Automatic Column Grouping
The `auto_column_groups` function categorizes the columns for preprocessing:

- **Numeric Columns**: Columns with numeric data types.
- **Low-Cardinality Categorical Columns**: Non-numeric columns with a small number of unique values (<= 20 by default), suitable for one-hot encoding.
- **High-Cardinality Categorical Columns**: Non-numeric columns with many unique values, handled with target encoding.

### Preprocessing Pipeline
A `ColumnTransformer` is used to apply different preprocessing steps to each column group:

- **Numeric**: Missing values are imputed with the median.
- **Low-Cardinality Categorical**: Features are one-hot encoded.
- **High-Cardinality Categorical**: Features are target encoded to prevent creating an excessive number of new columns. The implementation uses the `TargetEncoder` from scikit-learn (if available), which has built-in cross-fitting to prevent target leakage.

## 3. Model Training and Hyperparameter Tuning

The model is trained using a time-aware approach to ensure that the evaluation reflects real-world performance.

### Algorithm
The core of the model is a `HistGradientBoostingClassifier` from scikit-learn, which is a fast and efficient gradient boosting implementation.

### Time-Aware Splitting
The dataset is split into training, validation, and test sets based on unique `snapshot_date` values. The chronologically last snapshot is reserved as a hold-out test set.

### Imbalance Handling
Class imbalance is addressed using one of two strategies:
- If using scikit-learn version 1.5 or newer, `class_weight='balanced'` is passed to the classifier.
- Otherwise, `SMOTE` (Synthetic Minority Over-sampling Technique) from `imbalanced-learn` is used to oversample the minority class.

### Hyperparameter Tuning
Hyperparameter tuning is performed using `RandomizedSearchCV` with `TimeSeriesSplit` for cross-validation. This ensures that the validation folds are always chronologically after the training folds.

- **Checkpointing**: The results of the hyperparameter search are checkpointed to disk. This allows for faster subsequent runs by reusing the best-found parameters without rerunning the search. The script implements a multi-tier approach:
    1. Use pre-defined best-known parameters (fastest).
    2. Load a previously saved checkpoint.
    3. Run a new search if no checkpoint is available.

## 4. Model Calibration and Evaluation

After training, the model is calibrated to produce more reliable probability estimates.

### Calibration
A `CalibratedClassifierCV` with `method='isotonic'` is used to calibrate the model's output probabilities. The calibration is also performed using a `TimeSeriesSplit` to respect the temporal order of the data. The calibrated model is saved for later use.

### Evaluation
The model is evaluated on the hold-out test set using two primary metrics:
- **Average Precision (PR-AUC)**: A robust metric for imbalanced datasets.
- **Precision@K**: Measures the precision within the top K predicted leads, where K is a configurable parameter (`LEAD_CAPACITY_K`).

A gains table is also generated to show the model's performance across different score deciles.

## 5. Scoring and Output

The final step is to use the trained and calibrated model to score the current list of prospects.

- **Scoring**: The `predict_proba` method is used to get the probability of conversion for each company in the current snapshot.
- **Ranking**: The companies are ranked based on their conversion probability.
- **Output**: The final ranked list of leads is saved to a CSV file in the `outputs/` directory, with a timestamp in the filename (e.g., `ranked_leads_YYYYMMDD_HHMM.csv`). This file includes the company's ID, name, and the predicted conversion probability.
