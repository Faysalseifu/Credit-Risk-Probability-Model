import pandas as pd

from src.data_processing import preprocess


def test_preprocess_splits_features_and_target():
    df = pd.DataFrame({"f1": [1, 2], "target": [0, 1]})
    X, y = preprocess(df)

    assert list(X.columns) == ["f1"]
    assert y.tolist() == [0, 1]
