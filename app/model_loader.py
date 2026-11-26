import json
import mlflow


from src.config import (
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    CHAMPION_ALIAS,
    PROJECT_ROOT,
    setup_logger
)

logger = setup_logger(__name__)

def load_champion_model():
    try:
        CHAMPION_EXPORT_DIR = PROJECT_ROOT / "models" / "champion"
        if (CHAMPION_EXPORT_DIR / "model").exists() and (CHAMPION_EXPORT_DIR / "metadata.json").exists():
            model = mlflow.sklearn.load_model(str(CHAMPION_EXPORT_DIR / "model"))
            logger.info(f'Model loaded successfully: {model}')
            metadata_dict = json.loads((CHAMPION_EXPORT_DIR / "metadata.json").read_text(encoding="utf-8"))
            logger.info(f'Threshold loaded successfully: {metadata_dict['threshold']}')
            return (model, float(metadata_dict['threshold']), str(metadata_dict['version']))

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        model_version = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            CHAMPION_ALIAS
        )
        model = mlflow.sklearn.load_model(f'models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}')
        logger.info(f'Model loaded successfully: {model}')
        threshold_dict = mlflow.artifacts.load_dict(
            f'runs:/{model_version.run_id}/threshold.json'
        )
        logger.info(f'Threshold loaded successfully: {threshold_dict['threshold']}')
        return (model, float(threshold_dict['threshold']), str(model_version.version))
    except Exception as e:
        logger.error(f'Error loading model: {e}')
        raise RuntimeError("Model NOT found: {e}")
