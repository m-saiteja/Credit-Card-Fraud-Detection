import numpy as np
import pandas as pd
import pytest

from src.config import ALL_COLS, FEATURE_COLS, PCA_COLS, TARGET_COL
from src.data import split_data
from src.drift import detect_feature_drift
from src.evaluate import compute_metrics, find_best_threshold
from src.features import build_preprocessor, encode_cyclical_time


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    Create a small synthetic dataset (40 legit, 10 fraud) for fast tests.
    """
    rng = np.random.default_rng(42)
    n_rows = 50
    data = {col: rng.normal(size=n_rows) for col in PCA_COLS}
    data["Time"] = np.linspace(0, 172_000, n_rows)
    data["Amount"] = rng.uniform(1.0, 500.0, size=n_rows)
    data[TARGET_COL] = [0] * 40 + [1] * 10
    return pd.DataFrame(data, columns=ALL_COLS)


def test_split_data_stratified(sample_df):
    train_df, val_df, test_df = split_data(sample_df, strategy="stratified")

    # All rows accounted for, no data lost
    assert len(train_df) + len(val_df) + len(test_df) == len(sample_df)
    # Every split must contain both classes
    assert train_df[TARGET_COL].nunique() == 2
    assert val_df[TARGET_COL].nunique() == 2
    assert test_df[TARGET_COL].nunique() == 2


def test_split_data_time_order(sample_df):
    # Shuffle first to prove split_data sorts chronologically
    shuffled = sample_df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    train_df, val_df, test_df = split_data(shuffled, strategy="time")

    assert train_df["Time"].max() <= val_df["Time"].min()
    assert val_df["Time"].max() <= test_df["Time"].min()


def test_preprocessor_output_shapes(sample_df):
    X = sample_df[FEATURE_COLS]

    prep_with_time = build_preprocessor(amount_scaler="robust", use_cyclical_time=True)
    out_with_time = prep_with_time.fit_transform(X)
    assert out_with_time.shape == (len(X), 31)  # 28 PCA + 1 Amount + 2 (sin, cos)

    prep_no_time = build_preprocessor(amount_scaler="log_standard", use_cyclical_time=False)
    out_no_time = prep_no_time.fit_transform(X)
    assert out_no_time.shape == (len(X), 29)  # 28 PCA + 1 Amount


def test_encode_cyclical_time():
    # 0s = midnight -> (sin=0, cos=1); 43200s = noon -> (sin=0, cos=-1)
    encoded = encode_cyclical_time(np.array([[0.0], [43_200.0]]))
    expected = np.array([[0.0, 1.0], [0.0, -1.0]])
    assert np.allclose(encoded, expected, atol=1e-6)


def test_evaluate_threshold_and_metrics():
    y_true = np.array([0, 0, 1, 0, 1])
    y_proba = np.array([0.05, 0.20, 0.40, 0.35, 0.90])

    threshold, best_score = find_best_threshold(y_true, y_proba)
    assert isinstance(threshold, (float, np.floating))
    assert 0.0 <= best_score <= 1.0

    metrics = compute_metrics(y_true, y_proba, threshold)
    assert metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"] == len(y_true)
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_detect_feature_drift_flags_shifted_column(sample_df):
    prod_df = sample_df.copy()
    prod_df["Amount"] = prod_df["Amount"] + 10_000.0  # force massive shift

    report = detect_feature_drift(sample_df, prod_df, ["V14", "Amount"])
    assert report["features"]["Amount"]["drifted"] is True
    assert report["features"]["V14"]["drifted"] is False
