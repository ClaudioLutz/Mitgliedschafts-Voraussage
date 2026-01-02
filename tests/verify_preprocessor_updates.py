
import pandas as pd
import numpy as np
import scipy.sparse as sp
from column_transformer_lead_gen import create_lead_gen_preprocessor, to_float32, ToFloat32Transformer

def test_preprocessor_updates():
    # Create dummy data (larger size for TargetEncoder CV)
    n_samples = 20
    df = pd.DataFrame({
        'MitarbeiterBestand': np.random.choice([10, 20, np.nan, 40], n_samples),
        'Umsatz': np.random.choice([1000, 2000, 3000, np.nan], n_samples),
        'Risikoklasse': np.random.choice([1, 2, 3], n_samples),
        'Gruendung_Jahr': np.random.choice([2000, 1990, 2010, 2020], n_samples),
        'snapshot_date': pd.to_datetime(['2023-01-01']*n_samples),
        'MitarbeiterBestandKategorieOrder': np.random.randint(1, 5, n_samples),
        'UmsatzKategorieOrder': np.random.randint(1, 5, n_samples),
        'Kanton': np.random.choice(['ZH', 'BE', 'VD', 'AG'], n_samples),
        'Rechtsform': np.random.choice(['AG', 'GmbH'], n_samples),
        'GroessenKategorie': np.random.choice(['KLEIN', 'MITTEL', 'GROSS'], n_samples),
        'V_Bestand_Kategorie': np.random.choice(['A', 'B'], n_samples),
        'RechtsCode': np.random.choice(['01', '02'], n_samples),
        'PLZ': np.random.choice(['8000', '3000', '1000', '5000'], n_samples),
        'BrancheCode_06': np.random.choice(['A', 'B'], n_samples),
        'Target': np.random.randint(0, 2, n_samples)
    })

    # Manually add Company_Age_Years as it is expected (usually done by temporal_feature_engineer)
    df['Company_Age_Years'] = 2023 - df['Gruendung_Jahr']

    print("Testing OneHot Sparse Mode...")
    # 1. Test OneHot Sparse Mode
    preprocessor = create_lead_gen_preprocessor(onehot_sparse=True)
    X_transformed = preprocessor.fit_transform(df, df['Target'])

    # Check if output is sparse
    is_sparse = sp.issparse(X_transformed)
    print(f"Is output sparse? {is_sparse}")

    if not is_sparse:
        print("WARNING: Output is NOT sparse, but onehot_sparse=True was requested.")
        # Check ColumnTransformer configuration
        print(f"Sparse threshold: {preprocessor.named_steps['preprocessing'].sparse_threshold}")
    else:
        print("SUCCESS: Output is sparse.")

    # 2. Test Float32 Conversion
    print("\nTesting Float32 Transformer...")
    to_float = ToFloat32Transformer()
    X_float = to_float.transform(X_transformed)

    print(f"Output dtype: {X_float.dtype}")
    if X_float.dtype == np.float32:
        print("SUCCESS: Output is float32.")
    else:
        print(f"FAILURE: Output is {X_float.dtype}, expected float32.")

    # Check if sparse property is preserved
    if is_sparse and sp.issparse(X_float):
        print("SUCCESS: Sparsity preserved after float32 conversion.")
    elif is_sparse and not sp.issparse(X_float):
        print("FAILURE: Sparsity LOST after float32 conversion.")

    # 3. Test Dense Mode (Backward Compatibility)
    print("\nTesting Dense Mode...")
    preprocessor_dense = create_lead_gen_preprocessor(onehot_sparse=False)
    X_dense = preprocessor_dense.fit_transform(df, df['Target'])
    is_sparse_dense = sp.issparse(X_dense)
    print(f"Is output sparse? {is_sparse_dense}")
    if not is_sparse_dense:
        print("SUCCESS: Output is dense (default behavior).")
    else:
        print("FAILURE: Output is sparse, expected dense.")


if __name__ == "__main__":
    test_preprocessor_updates()
