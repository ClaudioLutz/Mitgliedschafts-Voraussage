# Feature Engineering Implementation

This document details the feature engineering process in the lead generation model.

## Data Source

The data for the model is sourced from a Microsoft SQL Server database. The connection is established using `pyodbc` and `SQLAlchemy`. Two functions are responsible for loading the data:

### `load_modeling_data`
This function retrieves historical monthly snapshots of company data. It is designed to create a dataset suitable for training a time-aware model. Key aspects of this function are:
- It only loads snapshots where the label (whether a company became a member) can be fully determined. This means the snapshot date is at least `HORIZON_MONTHS` in the past.
- It defines the target variable `Target`. A company is labeled as a positive case (1) if it becomes a member within `HORIZON_MONTHS` after the snapshot date. Otherwise, it's labeled as 0.
- The "risk set" is carefully constructed to include only non-members at the time of the snapshot. Companies that were already members or were founded after the snapshot are excluded to prevent data leakage.

### `load_current_snapshot`
This function loads the most recent snapshot of company data. This dataset represents the current pool of potential leads that need to be scored by the trained model.
- It does not have a `Target` label, as the outcomes are not yet known.
- It includes all non-member companies as of the current date.

## Temporal Feature Engineering

A dedicated function, `temporal_feature_engineer`, is used to create new features based on the snapshot date and other company attributes. This step is performed after loading the data and before any preprocessing. The goal is to create features that capture the state of the company at a specific point in time.

The following features are created:

-   **`Company_Age_Years`**: This feature represents the age of the company in years at the time of the snapshot. It is calculated by subtracting the company's founding year (`Gruendung_Jahr`) from the year of the snapshot (`snapshot_date`). If the founding year is not available, it is imputed with the snapshot year, resulting in an age of 0. The age is clipped at a minimum of 0 to handle any inconsistencies in the data.

-   **`Has_Employees`**: This is a binary flag that indicates whether a company has any employees. It is derived from the `MitarbeiterBestand` column. If the number of employees is greater than 0, this flag is set to 1; otherwise, it is 0. Missing values are treated as 0.

-   **`Has_Revenue`**: Similar to `Has_Employees`, this is a binary flag that indicates whether a company has any revenue. It is derived from the `Umsatz` column. If the revenue is greater than 0, this flag is set to 1; otherwise, it is 0. Missing values are treated as 0.

## Automatic Column Grouping

After feature engineering, the `auto_column_groups` function is used to automatically group the columns of the dataframe into three categories, which will be used for downstream preprocessing. This function helps in applying different preprocessing techniques to different types of columns.

The columns are grouped as follows:

-   **Numeric Columns**: These are columns that have a numeric data type (e.g., integer or float). A list of columns can also be explicitly provided to be treated as numeric, overriding the automatic detection.

-   **Low-Cardinality Categorical Columns**: These are non-numeric columns where the number of unique values is less than or equal to a specified threshold (`high_card_threshold`, which defaults to 20). These columns are suitable for one-hot encoding.

-   **High-Cardinality Categorical Columns**: These are non-numeric columns where the number of unique values is greater than the specified threshold. These columns are handled with target encoding to avoid creating a very large number of features.

The function also ensures that leakage columns such as `Target`, `Eintritt`, `Austritt`, and `snapshot_date` are excluded from these groups.

## Preprocessing Steps

Once the columns are grouped, a `ColumnTransformer` is used to apply different preprocessing steps to each group. This ensures that each type of feature is treated appropriately. The preprocessing steps are defined in separate pipelines for each column group.

### Numeric Preprocessing

The pipeline for numeric columns consists of a single step:
-   **Imputation**: `SimpleImputer` with `strategy='median'` is used to fill in any missing values. Median imputation is chosen because it is robust to outliers.

### Low-Cardinality Categorical Preprocessing

The pipeline for low-cardinality categorical columns consists of a single step:
-   **One-Hot Encoding**: `OneHotEncoder` is used to convert the categorical features into a numerical format. This creates a new binary column for each category in the original feature. `handle_unknown='ignore'` is used to prevent errors if new categories are encountered during prediction.

### High-Cardinality Categorical Preprocessing

The pipeline for high-cardinality categorical columns consists of a single step:
-   **Target Encoding**: `TargetEncoder` is used to encode these features. This method replaces each category with the mean of the target variable for that category. This is a powerful technique for handling high-cardinality features without creating a large number of new columns. The `TargetEncoder` from scikit-learn (if available) is used, which has built-in cross-fitting to prevent target leakage.
