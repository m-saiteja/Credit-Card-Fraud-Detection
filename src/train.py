import argparse
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit

import mlflow
from mlflow.models import infer_signature

from src.data import load_raw_data, split_data
from src.features import build_preprocessor
from src.evaluate import (
    find_best_threshold, 
    compute_metrics,
    plot_pr_curve,
    plot_confusion_matrix
)
from src.config import (
    setup_logger,
    RANDOM_STATE,
    FEATURE_COLS,
    TARGET_COL,
    REGISTERED_MODEL_NAME,
    CHAMPION_ALIAS,
    SPLIT_STRATEGY,
    VALID_SPLIT_STRATEGIES,
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    PARAM_GRIDS
)


logger = setup_logger(__name__)


def map_model_name(model_name: str, scale_pos_weight: float = 1.0, random_state=RANDOM_STATE):
    models = {
        "logreg": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "rf": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced_subsample",
            random_state=random_state,
            max_depth=12,
            n_jobs=-1,
        ),
        "hgb": HistGradientBoostingClassifier(
            class_weight="balanced",
            max_iter=200,
            random_state=random_state,
        ),
        "xgb": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=random_state,
            n_jobs=-1
        ),
        "lgbm": LGBMClassifier(
            class_weight="balanced",
            verbosity=-1,
            n_jobs=-1,
            random_state=random_state,
        )
    }
    if model_name in models:
        return models[model_name]
    raise ValueError(f"Unknown model: {model_name}")


def train_and_log_run(
    model_name,
    train_df,
    val_df,
    test_df,
    split_strategy,
    amount_scaler,
    use_cyclical_time,
    beta=1.0,
    tune: bool = False
):
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]

    X_val = val_df[FEATURE_COLS]
    y_val = val_df[TARGET_COL]

    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    classifier = map_model_name(model_name)
    preprocessor = build_preprocessor(amount_scaler, use_cyclical_time)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    with mlflow.start_run(run_name=f'{model_name}_{amount_scaler}_{split_strategy}') as run:
        mlflow.log_params({
            "model_name": model_name,
            "split_strategy": split_strategy,
            "amount_scaler": amount_scaler,
            "use_cyclical_time": use_cyclical_time,
            "beta": beta,
            **{f"clf_{k}": v for k, v in classifier.get_params().items()},
        })

        if tune and model_name in PARAM_GRIDS:
            if split_strategy == 'stratified':
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            else:
                cv = TimeSeriesSplit(n_splits=5)
            search = GridSearchCV(
                pipeline,
                PARAM_GRIDS[model_name],
                scoring="average_precision",
                cv=cv,
                n_jobs=-1
            )
            search.fit(X_train, y_train)
            fitted_pipeline = search.best_estimator_
            mlflow.log_params({f'best_{k}': v for k, v in search.best_params_.items()})
            mlflow.log_metric("cv_best_pr_auc", float(search.best_score_))
        else:
            pipeline.fit(X_train, y_train)
            fitted_pipeline = pipeline


        # tune the params using validation set
        logger.info('Tuning threshold on validation set...')
        val_proba = fitted_pipeline.predict_proba(X_val)[:, 1]
        best_threshold, best_score = find_best_threshold(y_val, val_proba, beta)
        val_metrics = compute_metrics(y_val, val_proba, best_threshold)
        mlflow.log_metrics({
            "best_threshold": best_threshold,
            "best_score": best_score,
            **{f'val_{k}': v for k, v in val_metrics.items()}
        })

        # evaluate on the test set only once
        logger.info('Evaluating on the test set once...')
        test_proba = fitted_pipeline.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, test_proba, best_threshold)
        mlflow.log_metrics({
            f'test_{k}': v for k, v in test_metrics.items()
        })

        # log plots - validation set
        val_pr_fig = plot_pr_curve(y_val, val_proba, title=model_name)
        val_cm_fig = plot_confusion_matrix(y_val, val_proba, threshold=best_threshold)

        mlflow.log_figure(val_pr_fig, "pr_curve_val.png")
        mlflow.log_figure(val_cm_fig, "confusion_matrix_val.png")

        #log plots - test set
        test_pr_fig = plot_pr_curve(y_test, test_proba, title=model_name)
        test_cm_fig = plot_confusion_matrix(y_test, test_proba, threshold=best_threshold)

        mlflow.log_figure(test_pr_fig, "pr_curve_test.png")
        mlflow.log_figure(test_cm_fig, "confusion_matrix_test.png")

        plt.close('all')

        mlflow.log_dict({'threshold': best_threshold}, 'threshold.json')

        log_kwargs = (
            {"serialization_format": "cloudpickle"}
            if model_name in ("xgb", "lgbm")
            else {
                "skops_trusted_types": [
                    "sklearn.tree._tree.Tree",
                    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
                    "src.features.encode_cyclical_time",
                    "src.features.cyclical_time_name",
                ]
            }
        )

        sample = X_train.head(5)
        signature = infer_signature(sample, fitted_pipeline.predict_proba(sample))
        model_info = mlflow.sklearn.log_model(
            sk_model=fitted_pipeline,
            name="model",
            signature=signature,
            input_example=sample,
            registered_model_name=REGISTERED_MODEL_NAME,
            **log_kwargs,
        )
        return {
            "run_id": run.info.run_id,
            "model_name": model_name,
            "val_pr_auc": val_metrics['pr_auc'],
            "test_pr_auc": test_metrics['pr_auc'],
            "best_threshold": best_threshold,
            "version": model_info.registered_model_version
        }


def promote_champion_model(registered_model_name, version):
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name=registered_model_name,
        alias=CHAMPION_ALIAS,
        version=version
    )
    logger.info(f"Promoted {registered_model_name} v{version} -> @{CHAMPION_ALIAS}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-strategy", 
        choices=VALID_SPLIT_STRATEGIES, 
        default=SPLIT_STRATEGY
    )
    parser.add_argument(
        "--amount-scaler",
        choices = ['robust', 'log_standard'],
        default='robust'
    )
    parser.add_argument(
        "--use-cyclical-time",
        action="store_true"
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['rf', 'hgb', 'logreg', 'xgb', 'lgbm']
    )
    parser.add_argument(
        "--tune",
        action="store_true"
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="promote best model to champion"
    )
    args = parser.parse_args()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_raw_data()
    train_df, val_df, test_df = split_data(df, strategy=args.split_strategy, random_state=RANDOM_STATE)

    model_results = []
    for model in args.models:
        result = train_and_log_run(
            model_name=model,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            split_strategy=args.split_strategy,
            amount_scaler=args.amount_scaler,
            use_cyclical_time=args.use_cyclical_time,
            tune=args.tune
        )
        model_results.append(result)

    winner = max(model_results, key=lambda x: x['val_pr_auc'])
    logger.info(f"Winner: {winner['model_name']} v{winner['version']}")
    if args.promote:
        promote_champion_model(REGISTERED_MODEL_NAME, winner['version'])
    else:
        logger.info('Skipping @champion promotion. Use --promote to update @champion')


if __name__ == "__main__":
    main()


    