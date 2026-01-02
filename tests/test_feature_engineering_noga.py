import pandas as pd

from column_transformer_lead_gen import FeatureEngineeringTransformer


def test_feature_engineering_adds_noga_features():
    df = pd.DataFrame({
        "BrancheCode_06": ["620100", "A", None],
        "Kanton": ["ZH", "BE", None],
        "PLZ": ["8000", "3000", "1000"],
    })

    transformer = FeatureEngineeringTransformer()
    out = transformer.fit_transform(df)

    assert "NOGA_section" in out.columns
    assert "NOGA_division" in out.columns
    assert "NOGA_group" in out.columns
    assert "Kanton_NOGA_section" in out.columns

    assert out.loc[0, "NOGA_section"] == "6"
    assert out.loc[0, "NOGA_division"] == "62"
    assert out.loc[0, "NOGA_group"] == "620"
    assert out.loc[1, "NOGA_section"] == "A"
    assert out.loc[2, "NOGA_section"] == "missing"
    assert out.loc[0, "Kanton_NOGA_section"] == "ZH_6"
