import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import (
    train_test_split,
)
from typing import Tuple

from src.config import (
    DATA_RAW_PATH,
    DATA_PROCESSED_PATH,
    SPLIT_STRATEGY,
    VALID_SPLIT_STRATEGIES,
    TIME_COL,
    FEATURE_COLS,
    TARGET_COL,
    ALL_COLS,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
    setup_logger
)

logger = setup_logger(__name__)

def load_raw_data(data_path: Path | str = DATA_RAW_PATH) -> pd.DataFrame:
    """
    Load the raw dataset from the specified path. 
    Args:
        data_path (str): Path to the raw data file.   
    Returns:
        pd.DataFrame: DataFrame containing the raw data.
    """
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Raw data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"Duplicated rows count: {df.duplicated().sum()}")
        for col in ALL_COLS:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the dataset")
        df = df[ALL_COLS].drop_duplicates().reset_index(drop=True)
        return df
    except Exception as e:
        logger.error(f"Error loading raw data: {e}")
        raise


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    strategy: str = SPLIT_STRATEGY,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training, validation, and testing sets.
    """
    try:
        if strategy not in VALID_SPLIT_STRATEGIES:
            raise ValueError(f"Invalid split strategy. Must be one of {VALID_SPLIT_STRATEGIES}")
        if strategy == 'stratified':
            X = df[FEATURE_COLS]
            y = df[TARGET_COL]
            train_val_X, test_X, train_val_y, test_y = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )
            train_X, val_X, train_y, val_y = train_test_split(
                train_val_X, train_val_y, test_size=val_size/(1-test_size), stratify=train_val_y, random_state=random_state
            )
            train_df = pd.concat([train_X, train_y], axis=1)
            val_df = pd.concat([val_X, val_y], axis=1)
            test_df = pd.concat([test_X, test_y], axis=1)
            logger.info(f"Stratified split performed: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
            logger.info(f"train_fraud_count: {train_df[TARGET_COL].sum()}, val_fraud_count: {val_df[TARGET_COL].sum()}, test_fraud_count: {test_df[TARGET_COL].sum()}")
            return train_df, val_df, test_df
        elif strategy == 'time':
            df = df.sort_values(by=TIME_COL).reset_index(drop=True)
            n_samples = len(df)
            val_split_idx = int(n_samples * (1-test_size-val_size))
            test_split_idx = int(n_samples * (1-test_size))
            train_df = df.iloc[:val_split_idx]
            val_df = df.iloc[val_split_idx:test_split_idx]
            test_df = df.iloc[test_split_idx:]
            logger.info(f"Time-based split performed: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
            logger.info(f"train_fraud_count: {train_df[TARGET_COL].sum()}, val_fraud_count: {val_df[TARGET_COL].sum()}, test_fraud_count: {test_df[TARGET_COL].sum()}")
            return train_df, val_df, test_df
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise
    

def save_splits(train_df, val_df, test_df, output_dir=DATA_PROCESSED_PATH):
    """
    Save the train, validation, and test DataFrames to CSV files.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        logger.info(f"Train, validation, and test sets saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error saving splits: {e}")
        raise


if __name__ == "__main__":
    raw_df = load_raw_data()
    train_df, val_df, test_df = split_data(raw_df)
    print(train_df.shape, val_df.shape, test_df.shape)
    save_splits(train_df, val_df, test_df)