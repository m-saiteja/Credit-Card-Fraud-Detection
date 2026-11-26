import os
import json
from datetime import datetime, timezone

from src.config import (
    setup_logger,
    PROJECT_ROOT
)

logger = setup_logger(__name__)

PREDICTION_LOG_PATH = PROJECT_ROOT / "logs" / "predictions.jsonl"


def log_prediction(
    features_dict,
    fraud_probability,
    is_fraud,
    threshold,
    model_version,
    latency_ms
):
    os.makedirs(PREDICTION_LOG_PATH.parent, exist_ok=True)

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features_dict,
        "fraud_probability": fraud_probability,
        "is_fraud": is_fraud,
        "threshold": threshold,
        "model_version": model_version,
        "latency_ms": latency_ms,
    }

    try:
        with open(PREDICTION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        logger.debug("Prediction logged successfully")
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")


def get_live_stats():
    if not PREDICTION_LOG_PATH.exists():
        return {"total_predictions": 0}
    else:
        try:
            with open(PREDICTION_LOG_PATH, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            total_predictions = len(lines)

            if total_predictions == 0:
                return {"total_predictions": 0}

            stats = {
                "total_predictions": total_predictions,
                "flagged_fraud_count": 0,
                "flagged_rate": 0,
                "avg_latency_ms": 0,
                "avg_probability": 0,                
            }

            total_latency = 0
            total_prob = 0
            fraud_count = 0
            recent = []

            ## keeping only the last 100 predictions to keep the stats recent
            for line in lines[-100:]:
                try:
                    entry = json.loads(line.strip())

                    total_latency += entry.get("latency_ms", 0)
                    total_prob += entry.get("fraud_probability", 0)

                    if entry.get("is_fraud", False):
                        fraud_count += 1

                    # Append essential info for recent predictions
                    recent.append({
                        "timestamp": entry.get("timestamp"),
                        "prob": entry.get("fraud_probability"),
                        "is_fraud": entry.get("is_fraud")
                    })

                except Exception as e:
                    logger.debug(f"Skipping invalid log entry: {e}")
                    continue

            len_recent_preds = len(recent)
            if len_recent_preds == 0:
                return {"total_predictions": total_predictions}
            
            stats["flagged_fraud_count"] = fraud_count
            stats["flagged_rate"] = fraud_count / len_recent_preds
            stats["avg_latency_ms"] = total_latency / len_recent_preds
            stats["avg_probability"] = total_prob / len_recent_preds
            stats["recent_predictions"] = recent[-10:]

            return stats

        except Exception as e:
            logger.error(f"Failed to get live stats: {e}")
            return {"total_predictions": 0}
