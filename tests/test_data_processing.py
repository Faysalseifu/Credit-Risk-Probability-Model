import numpy as np
import pandas as pd

from src.data_processing import (
    AggregateByGroup,
    DateTimeFeatureExtractor,
    WeightOfEvidenceEncoder,
    build_preprocessing_pipeline,
    split_features_target,
)


def test_split_features_target_uses_named_column():
    df = pd.DataFrame({"f1": [1, 2], "FraudResult": [0, 1]})
    X, y = split_features_target(df)

    assert list(X.columns) == ["f1"]
    assert y.tolist() == [0, 1]


def test_datetime_feature_extractor_creates_parts():
    df = pd.DataFrame({"TransactionStartTime": ["2020-01-01T05:10:00Z"]})
    transformer = DateTimeFeatureExtractor()
    result = transformer.fit_transform(df)

    assert {"transaction_hour", "transaction_day", "transaction_month", "transaction_year"}.issubset(result.columns)
    assert result.loc[0, "transaction_hour"] == 5


def test_aggregate_by_group_adds_customer_stats():
    df = pd.DataFrame(
        {
            "CustomerId": ["c1", "c1", "c2"],
            "Amount": [10.0, 30.0, 5.0],
        }
    )
    agg = AggregateByGroup()
    result = agg.fit_transform(df)

    assert np.isclose(result.loc[0, "total_amount_sum"], 40.0)
    assert result.loc[2, "transaction_count"] == 1


def test_pipeline_runs_end_to_end():
    df = pd.DataFrame(
        {
            "TransactionStartTime": ["2020-01-01T05:10:00Z", "2020-01-02T07:30:00Z"],
            "CustomerId": ["c1", "c2"],
            "Amount": [10.0, 20.0],
            "ProductCategory": ["airtime", "utility_bill"],
            "FraudResult": [0, 1],
        }
    )
    X, y = split_features_target(df)
    pipeline = build_preprocessing_pipeline()

    transformed = pipeline.fit_transform(X, y)

    assert transformed.shape[0] == 2
    assert transformed.shape[1] > 0
