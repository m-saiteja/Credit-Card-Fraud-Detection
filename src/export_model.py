import json
import shutil
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.config import (
    CHAMPION_ALIAS,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    REGISTERED_MODEL_NAME,
    setup_logger,
)

logger = setup_logger(__name__)


def export_champion(export_dir: Path | None = None) -> Path:
    """
    Download the @champion model and its threshold/version metadata for Docker serving.
    """
    if export_dir is None:
        export_dir = MODELS_DIR / "champion"

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)

    # Clear previous export so stale files never linger across upgrades
    if export_dir.exists():
        shutil.rmtree(export_dir)

    model_dir = export_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Download model directly into models/champion/model/
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}",
        dst_path=str(model_dir),
    )

    threshold_dict = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/threshold.json")
    metadata = {
        "model_name": REGISTERED_MODEL_NAME,
        "alias": CHAMPION_ALIAS,
        "version": str(mv.version),
        "run_id": mv.run_id,
        "threshold": float(threshold_dict["threshold"]),
    }

    (export_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    logger.info(
        f"Exported {REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS} (v{mv.version}) to {export_dir}"
    )
    return export_dir


if __name__ == "__main__":
    export_champion()
