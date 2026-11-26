import json
import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

import mlflow

from src.config import (
    DRIFT_EXPERIMENT_NAME,
    MONITOR_FEATURES,
    MLFLOW_TRACKING_URI,
    DATA_PROCESSED_PATH,
    setup_logger
)

logger = setup_logger(__name__)

def load_logs(log_path, last_n=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load prediction logs from prediction.jsonl file.
    """
    if not Path(log_path).exists():
        raise FileNotFoundError(f"File not found: {log_path}")

    with open(log_path,'r') as f:
        records = [json.loads(line) for line in f]

    if len(records) == 0:
        raise ValueError("No records found in log file")

    if last_n:
        records = records[-last_n:]
    
    features_df = pd.DataFrame([r['features'] for r in records])
    meta_df = pd.DataFrame([
        {
            'timestamp':r['timestamp'],
            'fraud_probability': r['fraud_probability'],
            'is_fraud':r['is_fraud'],
            'model_version':r['model_version'],
            'latency_ms': r['latency_ms']
        } for r in records])
    
    return features_df, meta_df


def detect_feature_drift(ref_df, prod_df, features, ks_threshold=0.15, alpha=0.05) -> dict:
    results = {}
    drifted_features = []

    for feature in features:
        stat, p_val = ks_2samp(ref_df[feature].dropna(), prod_df[feature].dropna())
        is_drifted = (stat >= ks_threshold) and (p_val <  alpha)

        if is_drifted:
            drifted_features.append(feature)
            logger.debug(f"Drift detected in {feature}: KS stat={stat:.4f}, p-value={p_val:.4f}")

        results[feature] = {
            "ks_stat": float(stat),
            "p_value": float(p_val),
            "drifted": bool(is_drifted),
            "reference_mean": float(ref_df[feature].mean()),
            "production_mean": float(prod_df[feature].mean()),
        }
    
    return {
        "n_reference": len(ref_df),
        "n_production": len(prod_df),
        "drifted_features": drifted_features,
        "drifted_count": len(drifted_features),
        "drift_share": len(drifted_features) / len(features),
        "features": results
    }
    

def plot_drift_summary(report, ks_threshold=0.15):
    features = report["features"]
    ks_stats = [f["ks_stat"] for f in features.values()]
    colors = ["red" if features[key]["drifted"] else "green" for key in features]
    feature_names = list(features.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feature_names, ks_stats, color=colors)
    ax.axvline(ks_threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"KS Threshold ({ks_threshold})")
    ax.set_xlabel("KS Statistic (0 = identical, 1 = completely shifted)")
    ax.set_title(f"Feature Drift Check (n_prod={report['n_production']})")
    ax.legend()
    fig.tight_layout()
    return fig


def log_drift_to_mlflow(report, meta_df, ks_threshold=0.15):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(DRIFT_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f'drift-check_{meta_df["timestamp"].max()}') as run:
        mlflow.log_metrics({
            "n_production": report["n_production"],
            "drifted_count": report["drifted_count"],
            "drift_share": report["drift_share"],
            "prod_flagged_rate": float(meta_df["is_fraud"].mean()),
            "prod_avg_prob": float(meta_df["fraud_probability"].mean())
        })

        for feature, vals in report['features'].items():
            mlflow.log_metrics({
                f"drift.{feature}.ks_stat": vals["ks_stat"],
                f"drift.{feature}.p_value": vals["p_value"],
            })
    
        mlflow.log_dict(report, "drift_report.json")
        fig = plot_drift_summary(report, ks_threshold=ks_threshold)
        mlflow.log_figure(fig, "feature_drift.png")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-path",
        default="logs/predictions.jsonl"
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=1000,
        help="Number of recent predictions to use for drift detection"
    )
    parser.add_argument(
        "--ks-threshold",
        default=0.15,
        type=float
    )
    args = parser.parse_args()

    ref_df = pd.read_csv(DATA_PROCESSED_PATH / "train.csv")
    prod_logs, prod_meta_df = load_logs(args.log_path, last_n=args.last_n)

    if len(prod_logs) < 30:
        logger.warning("Low sample size; drift stats may be noisy")
    
    drift_result = detect_feature_drift(
        ref_df,
        prod_logs,
        MONITOR_FEATURES,
        ks_threshold=args.ks_threshold
    )

    if drift_result['drifted_count'] > 0:
        logger.warning(f"Drift detected in {drift_result['drifted_count']}/{len(MONITOR_FEATURES)} features")
    else:
        logger.info("No drift detected")
    print("Drift report:", json.dumps(drift_result, indent=2))

    log_drift_to_mlflow(
        drift_result,
        prod_meta_df,
        ks_threshold=args.ks_threshold
    )

    logger.info("Drift detection completed")
    

if __name__ == "__main__":
    main()