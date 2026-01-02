
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import os
import shutil

# Set backend to xgb_cpu for testing
os.environ["MODEL_BACKEND"] = "xgb_cpu"

# Import the module under test
import training_lead_generation_model

class TestXGBPipeline(unittest.TestCase):
    def setUp(self):
        # Create a temporary output directory
        self.test_dir = "test_outputs"
        os.makedirs(self.test_dir, exist_ok=True)
        # Patch the output directory in the module
        training_lead_generation_model.OUTDIR = self.test_dir
        training_lead_generation_model.ARTIFACTS_DIR = self.test_dir

        # Override config to use best params (faster)
        training_lead_generation_model.USE_BEST_KNOWN_PARAMS = True
        training_lead_generation_model.FORCE_NEW_SEARCH = False

    def tearDown(self):
        # Cleanup
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def create_dummy_data(self, n_samples=2000):
        # Generate enough data for 3 snapshots (train, val, test)
        dates = pd.to_datetime(['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01'])

        df_list = []
        for date in dates:
            n_snap = n_samples // 4
            df = pd.DataFrame({
                'CrefoID': np.arange(n_snap) + (dates.tolist().index(date) * 10000),
                'Name_Firma': [f'Firma_{i}' for i in range(n_snap)],
                'Gruendung_Jahr': np.random.randint(1990, 2021, n_snap),
                'PLZ': np.random.choice(['8000', '3000', '1000'], n_snap),
                'Kanton': np.random.choice(['ZH', 'BE', 'VD'], n_snap),
                'Rechtsform': np.random.choice(['AG', 'GmbH'], n_snap),
                'BrancheText_06': ['B6']*n_snap,
                'BrancheCode_06': np.random.choice(['A', 'B'], n_snap),
                'MitarbeiterBestand': np.random.randint(1, 100, n_snap),
                'MitarbeiterBestandKategorie': ['MBK']*n_snap,
                'Umsatz': np.random.randint(1000, 100000, n_snap),
                'UmsatzKategorie': ['UK']*n_snap,
                'UmsatzKategorieOrder': np.random.randint(1, 5, n_snap),
                'Risikoklasse': np.random.randint(1, 4, n_snap),
                'Ort': ['Ort']*n_snap,
                'RechtsCode': np.random.choice(['01', '02'], n_snap),
                'GroessenKategorie': np.random.choice(['KLEIN', 'MITTEL'], n_snap),
                'V_Bestand_Kategorie': np.random.choice(['A', 'B'], n_snap),
                'MitarbeiterBestandKategorieOrder': np.random.randint(1, 5, n_snap),
                'BrancheCode_02': ['B2']*n_snap,
                'BrancheCode_04': ['B4']*n_snap,
                'BrancheText_02': ['BT2']*n_snap,
                'BrancheText_04': ['BT4']*n_snap,
                'Eintritt': [None]*n_snap, # mostly null
                'Austritt': [None]*n_snap,
                'DT_LoeschungAusfall': [None]*n_snap,
                'snapshot_date': date,
                'Target': np.random.randint(0, 2, n_snap)
            })
            df_list.append(df)

        return pd.concat(df_list, ignore_index=True)

    @patch('training_lead_generation_model.make_engine')
    @patch('training_lead_generation_model.load_modeling_data')
    @patch('training_lead_generation_model.load_current_snapshot')
    @patch('training_lead_generation_model.compute_ts_gap_samples')
    def test_pipeline_execution(self, mock_gap, mock_load_current, mock_load_model, mock_make_engine):
        print("\nTesting End-to-End Pipeline with XGB_CPU...")

        # Setup mocks
        mock_make_engine.return_value = MagicMock()
        mock_gap.return_value = 10 # Force small gap for testing

        df_model = self.create_dummy_data(n_samples=2000)
        mock_load_model.return_value = df_model

        # Current snapshot is just one date (latest)
        df_current = self.create_dummy_data(n_samples=100)
        df_current['snapshot_date'] = pd.to_datetime('2023-01-01')
        mock_load_current.return_value = df_current

        # Run main
        try:
            training_lead_generation_model.main()
            print("SUCCESS: Pipeline ran to completion.")
        except Exception as e:
            self.fail(f"Pipeline failed with error: {e}")

        # Verify outputs exist
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "calibrated_model.joblib")))
        # Check for csvs
        files = os.listdir(self.test_dir)
        csv_files = [f for f in files if f.endswith('.csv')]
        self.assertTrue(len(csv_files) > 0, "No CSV outputs found.")
        print(f"Generated artifacts: {files}")

if __name__ == "__main__":
    unittest.main()
