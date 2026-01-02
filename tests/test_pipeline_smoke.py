import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from column_transformer_lead_gen import create_lead_gen_preprocessor, DROP_COLS
from training_lead_generation_model import temporal_feature_engineer

def make_synthetic_frame(n=500, n_snapshots=6, seed=42):
    rng = np.random.default_rng(seed)
    snapshot_dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_snapshots, freq="MS")

    df = pd.DataFrame({
        "CrefoID": rng.integers(1_000_000, 9_999_999, size=n),
        "Name_Firma": [f"Demo_{i:05d}" for i in range(n)],
        "Gruendung_Jahr": rng.integers(1950, pd.Timestamp.today().year + 1, size=n),
        "PLZ": rng.integers(1000, 9999, size=n),
        "Kanton": rng.choice(["ZH","BE","VD","GE","AG"], size=n),
        "Rechtsform": rng.choice(["Einzelunternehmen","GmbH","Aktiengesellschaft"], size=n),
        "RechtsCode": rng.choice(["AG","GMBH","EINF"], size=n),
        "BrancheCode_06": rng.choice(["620100","471100","692000"], size=n),
        "BrancheText_06": rng.choice(["IT","Handel","Treuhand"], size=n),

        "MitarbeiterBestand": rng.lognormal(2.0, 1.0, size=n).round().astype(int),
        "MitarbeiterBestandKategorieOrder": rng.integers(1, 7, size=n),
        "MitarbeiterBestandKategorie": rng.choice(["0","1-9","10-49","50-249"], size=n),

        "Umsatz": rng.lognormal(14.0, 1.0, size=n),
        "UmsatzKategorieOrder": rng.integers(1, 7, size=n),
        "UmsatzKategorie": rng.choice(["<1M","1-5M","5-20M",">20M"], size=n),

        "Risikoklasse": rng.integers(1, 5, size=n),

        "Ort": rng.choice(["Zürich","Bern","Lausanne"], size=n),
        "GroessenKategorie": rng.choice(["MICRO","KLEIN","MITTEL"], size=n),
        "V_Bestand_Kategorie": rng.choice(["NEU","BESTAND","ALT"], size=n),

        "Eintritt": pd.NaT,
        "Austritt": pd.NaT,
        "DT_LoeschungAusfall": pd.NaT,

        "snapshot_date": pd.to_datetime(rng.choice(snapshot_dates, size=n)),
    })

    # simple target
    df["Target"] = rng.binomial(1, 0.08, size=n)
    df = df.sort_values(["snapshot_date", "CrefoID"]).reset_index(drop=True)
    return df

def test_pipeline_fit_predict_proba_smoke():
    df = make_synthetic_frame()
    df_eng = temporal_feature_engineer(df)

    feature_cols = [c for c in df_eng.columns if c not in DROP_COLS]
    X = df_eng[feature_cols]
    y = df_eng["Target"].astype(int).values

    pre = create_lead_gen_preprocessor()

    pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=500))
    ])

    pipe.fit(X, y)
    proba = pipe.predict_proba(X)

    assert proba.shape == (len(X), 2)
    assert np.isfinite(proba).all()
    assert (proba >= 0).all() and (proba <= 1).all()
