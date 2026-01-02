# User Interfaces & Reporting

This document describes the current state of user interfaces and reporting for the *Mitgliedschafts-Voraussage* (Membership Prediction) project. The project is a machine learning pipeline that operates via command-line scripts.

## User Interaction

There is currently no graphical user interface (GUI) for this project. All interactions are performed by running Python scripts from the command line.

- **Main Script:** `training_lead_generation_model.py` is the primary script for training the model and generating predictions.
- **Utility Script:** `time_estimation.py` estimates the runtime of a full training cycle using the same preprocessing pipeline and temporal feature step as training.

## Data Sources

The model retrieves its data from a production database:

- **Server:** `PRODSVCREPORT70`
- **Database:** `CAG_Analyse`
- **Schema:** `mitgliederstatistik`
- **Primary Table:** `MitgliederSegmentierung`

## Reporting Outputs

The project generates several outputs that serve as its reporting mechanism. These files are saved in the `outputs/` and `artifacts/` directories.

### Primary Business Reporting

- **Ranked Lead Lists (CSV):** The most important output is a CSV file containing a ranked list of potential new members (leads). This file is intended for use by the sales and marketing teams.
  - **File Naming:** `outputs/ranked_leads_YYYYMMDD_HHMM.csv` (e.g., `ranked_leads_20230928_1530.csv`)
  - **Content:** Includes `CrefoID`, `Name_Firma`, a prediction score (`p_convert`), and a rank.

### Model Performance Reporting

- **Gains Table (CSV):** For evaluating the model's performance, a gains table is generated from the test set. This report helps data scientists understand the model's effectiveness at identifying positive leads.
  - **File Naming:** `outputs/gains_table_test.csv`
  - **Content:** Provides statistics like conversion rates and lift by model score decile.

- **Console Logs:** During execution, the scripts produce detailed logs to the console. This provides real-time feedback for the technical user running the model.

### Model Artifacts

- **Serialized Models:** The trained machine learning models and pipelines are saved as `joblib` files. These are not human-readable reports but are essential for the system's operation, allowing for model reuse and checkpointing.
  - **Location:** `./artifacts/`
  - **Files:** `best_pipeline.joblib`, `calibrated_model.joblib`, etc.

## Future Development (Aspirational)

The following features were considered in the project design but are **not yet implemented**:

- **Interactive Dashboards:** A `Streamlit` dashboard for visualizing predictions and metrics.
- **Custom Alerts:** An automated system to send email or Slack alerts for high-probability leads.
- **Self-Service Analytics:** A user interface for non-technical stakeholders to run their own analyses.
- **Database Integration:** The main script includes commented-out code to write ranked leads directly to a SQL database table (`lead_generation_rankings`), which is currently disabled.
