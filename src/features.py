import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    RobustScaler, 
    StandardScaler, 
    FunctionTransformer
)

from src.config import (
    DATA_PROCESSED_PATH,
    PCA_COLS,
    TIME_COL,
    AMOUNT_COL,
    FEATURE_COLS,
    setup_logger
)

logger = setup_logger(__name__)
SECONDS_IN_DAY = 86400

def cyclical_time_name(transformer, input_features):
    return ["time_sin", "time_cos"]


def encode_cyclical_time(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """
    Convert the time from seconds to [sin, cos] coordinates of
    24th cycle representing the 24 hours of the day.
    """
    seconds = np.asarray(X, dtype=float).ravel()
    angle = 2.0 * np.pi * (seconds % SECONDS_IN_DAY) / SECONDS_IN_DAY
    return np.column_stack([np.sin(angle), np.cos(angle)])


def build_preprocessor(
    amount_scaler: str = 'robust',
    use_cyclical_time: bool = True
) -> ColumnTransformer:
    """
    Build an UNFITTED ColumnTransformer for credit card fraud feature engineering.
    """
    if amount_scaler == 'robust':
        amount_transformer = RobustScaler()
    elif amount_scaler == 'log_standard':
        amount_transformer = Pipeline([
            ('log', FunctionTransformer(np.log1p, validate=True, feature_names_out='one-to-one')),
            ('scaler', StandardScaler())
        ])
    else:
        raise ValueError(f'Unsupported amount_scaler {amount_scaler}')
    
    if use_cyclical_time:
        time_transformer = FunctionTransformer(
            encode_cyclical_time,
            feature_names_out=cyclical_time_name
        )
    else:
        time_transformer = "drop"

    preprocessor = ColumnTransformer(
        transformers=[
            ('pca', 'passthrough', PCA_COLS),
            ('amount', amount_transformer, [AMOUNT_COL]),
            ('time', time_transformer, [TIME_COL]),
        ],
        remainder='drop'
    )
    return preprocessor


if __name__ == "__main__":
    train_df = pd.read_csv(DATA_PROCESSED_PATH / "train.csv")
    val_df = pd.read_csv(DATA_PROCESSED_PATH / "val.csv")
    
    X_train = train_df[FEATURE_COLS]
    X_val = val_df[FEATURE_COLS]

    prep = build_preprocessor(amount_scaler='robust', use_cyclical_time=True)
    X_train_proc = prep.fit_transform(X_train)
    X_val_proc = prep.transform(X_val)

    print(X_train_proc.shape, X_val_proc.shape)
    print(X_train_proc)


    prep_ = build_preprocessor(amount_scaler='log_standard', use_cyclical_time=False)
    X_train_proc = prep_.fit_transform(X_train)
    X_val_proc = prep_.transform(X_val)

    print(X_train_proc.shape, X_val_proc.shape)
    print(X_train_proc)