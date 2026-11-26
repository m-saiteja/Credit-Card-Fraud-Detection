# Credit Card Fraud Detection

An end-to-end machine learning classification system to detect fraudulent credit card transactions, built with **scikit-learn**, **XGBoost**, **LightGBM**, **MLflow** (experiment tracking, model registry, and drift monitoring), **FastAPI**, **pytest**, and **Docker**.

---

## Key Highlights

* **Leakage-Free Data Pipeline (`src/data.py`):** Deduplicates the raw dataset (`284,807` → `283,726` unique rows, removing `1,081` duplicates) *before* splitting into `train` (65%), `val` (15%), and `test` (20%). Supports both `"stratified"` and chronological `"time"` splits.
* **Preprocessing via `ColumnTransformer` (`src/features.py`):** Passes PCA features `V1`–`V28` through untouched, scales skewed `Amount` (`RobustScaler` or `log1p + StandardScaler`), and drops raw elapsed `Time` or converts it into cyclical 24-hour `(sin, cos)` coordinates inside the `Pipeline`.
* **Multi-Model Training & Hyperparameter Tuning (`src/train.py`):** Trains **Random Forest**, **LightGBM**, **XGBoost**, **HistGradientBoosting**, and **Logistic Regression** inside leak-free `Pipeline`s, with optional `GridSearchCV` (`--tune`) using `StratifiedKFold` or `TimeSeriesSplit` scored on `average_precision`.
* **Imbalance-Aware Evaluation (`src/evaluate.py`):** With only `0.173%` fraud (`492` raw / `473` unique frauds), models are evaluated using **PR-AUC**, **Precision**, **Recall**, and **$F_\beta$** rather than accuracy, with probability thresholds tuned strictly on the validation set.
* **MLflow Lifecycle (`src/train.py`, `src/export_model.py`):** Logs parameters, validation/test metrics, PR curves, confusion matrices, and serialized `Pipeline` artifacts (`skops` for `sklearn` models, `cloudpickle` for `XGBoost`/`LightGBM`) to `mlflow.db`, gates promotion behind `--promote` (`@champion`), and exports artifacts for portable container serving.
* **FastAPI Serving & Telemetry (`app/`):** Serves `/health`, `/predict` (with Pydantic input validation), and `/stats`, while logging predictions (`logs/predictions.jsonl`) with latency and model version.
* **Batch Drift Monitoring (`src/drift.py`):** Runs two-sample Kolmogorov–Smirnov (`ks_2samp`) checks on key fraud-signal features (`V17`, `V14`, `V12`, `V10`, `V11`, `V4`, `Amount`) against training data and logs drift reports and charts back to MLflow (`credit-card-fraud-monitoring`).

---

## Project Structure

```text
Credit-Card-Fraud-Detection/
├── app/
│   ├── main.py             # FastAPI app (/health, /predict, /stats) with lifespan loading
│   ├── model_loader.py     # Loads exported models/champion/ or MLflow @champion
│   ├── monitor.py          # Appends prediction logs (JSONL) and computes live stats
│   └── schemas.py          # Pydantic request/response schemas (Time, Amount, V1..V28)
├── data/
│   ├── raw/                # creditcard.csv
│   └── processed/          # train.csv, val.csv, test.csv
├── models/
│   └── champion/           # Exported @champion model & metadata.json for Docker
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory data analysis
├── src/
│   ├── config.py           # Paths, schema constants, split, MLflow & PARAM_GRIDS settings
│   ├── data.py             # Ingestion, validation, deduplication, stratified/time splits
│   ├── features.py         # ColumnTransformer preprocessing factory
│   ├── evaluate.py         # Threshold tuning (F-beta) and PR/ROC/confusion metrics & plots
│   ├── train.py            # Training pipeline + MLflow tracking & @champion promotion
│   ├── export_model.py     # Exports @champion artifacts for self-contained Docker builds
│   └── drift.py            # Batch KS feature drift monitor logged to MLflow
├── tests/
│   ├── test_pipeline.py    # Unit tests for splitting, preprocessing, evaluation, and drift
│   └── test_api.py         # FastAPI schema & endpoint tests
├── Dockerfile
├── .dockerignore
├── pyproject.toml
└── uv.lock
```

---

## Model Results

### Tuned Models (`time` split, `log_standard` scaler, `use_cyclical_time=True`, `GridSearchCV`)

| Model | CV PR-AUC | Val PR-AUC | Test PR-AUC | Test Precision | Test Recall | Test F1 | Tuned Threshold | Best Params |
|---|---|---|---|---|---|---|---|---|
| **XGBoost** (`@champion` v16) | **0.8002** | **0.7814** | 0.7831 | 0.9123 | 0.7027 | 0.7939 | `0.7018` | `max_depth=3, lr=0.05, n_estimators=300` |
| **LightGBM** | 0.7960 | 0.7809 | 0.7981 | 0.9310 | **0.7297** | **0.8182** | `0.9947` | `num_leaves=63, lr=0.1, n_estimators=300` |
| Random Forest | 0.7972 | 0.7512 | **0.8044** | 0.8710 | **0.7297** | 0.7941 | `0.4533` | `max_depth=10, n_estimators=200` |
| Logistic Regression | 0.7183 | 0.7495 | 0.7332 | 0.9245 | 0.6622 | 0.7717 | `1.0000` | `C=0.1` |
| HistGradientBoosting | 0.7372 | 0.7383 | 0.7647 | **0.9400** | 0.6351 | 0.7581 | `0.9747` | `max_depth=4, lr=0.1` |

### Baseline Results (`stratified` split, `robust` scaler, untuned)

| Model | Val PR-AUC | Test PR-AUC | Test Precision | Test Recall | Test F1 | Tuned Threshold |
|---|---|---|---|---|---|---|
| **Random Forest** | **0.8694** | 0.7833 | 0.8795 | **0.7684** | 0.8202 | `0.3295` |
| **LightGBM** | 0.8685 | **0.8161** | **0.9714** | 0.7158 | **0.8242** | `0.9929` |
| XGBoost | 0.8161 | 0.7251 | **0.9714** | 0.7158 | **0.8242** | `0.3127` |
| Logistic Regression | 0.8157 | 0.6887 | 0.8846 | 0.7263 | 0.7977 | `1.0000` |
| HistGradientBoosting | 0.8024 | 0.7150 | 0.8235 | 0.7368 | 0.7778 | `0.9196` |

---

## Quickstart

### 1. Setup Environment
Requires Python `>=3.12` and [`uv`](https://docs.astral.sh/uv/). On macOS, XGBoost and LightGBM also require OpenMP (`libomp`). Place `creditcard.csv` inside `data/raw/`.

```bash
brew install libomp   # macOS only (needed for xgboost & lightgbm)
uv sync
```

### 2. Prepare Data Splits
Deduplicates `data/raw/creditcard.csv` and writes `train.csv`, `val.csv`, and `test.csv` into `data/processed/`:

```bash
uv run python -m src.data
```

### 3. Train Models & Inspect in MLflow
```bash
# Train all 5 models (rf, hgb, logreg, xgb, lgbm) without overwriting @champion
uv run python -m src.train

# Run GridSearchCV hyperparameter tuning and promote the best model to @champion
uv run python -m src.train --tune --promote

# Optional flags:
# uv run python -m src.train --models xgb lgbm --split-strategy time --amount-scaler log_standard --use-cyclical-time

# Launch MLflow UI (http://127.0.0.1:5000)
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**Training CLI Options (`src/train.py`):**

| Flag | Options / Default | Description |
|---|---|---|
| `--models` | `rf hgb logreg xgb lgbm` | Space-separated list of models to train |
| `--split-strategy` | `stratified` \| `time` (default: `stratified`) | Train/val/test split strategy |
| `--amount-scaler` | `robust` \| `log_standard` (default: `robust`) | Scaler used on `Amount` in `ColumnTransformer` |
| `--use-cyclical-time` | flag (default: `False`) | Encode `Time` as 24h `(sin, cos)` instead of dropping |
| `--tune` | flag (default: `False`) | Run 5-fold `GridSearchCV` using `PARAM_GRIDS` in `src/config.py` |
| `--promote` | flag (default: `False`) | Promote the highest `val_pr_auc` model to `@champion` |

### 4. Export `@champion` & Run the FastAPI Service Locally
```bash
uv run python -m src.export_model
uv run uvicorn app.main:app --reload --port 8000
```
Open **`http://127.0.0.1:8000/docs`** for interactive Swagger documentation (`/health`, `/predict`, `/stats`).

### 5. Run with Docker
```bash
docker build -t fraud-detector-api .
docker run -p 8000:8000 -v $(pwd)/logs:/app/logs fraud-detector-api
```

### 6. Run Batch Drift Monitoring
After sending predictions through `/predict` (`logs/predictions.jsonl`):

```bash
uv run python -m src.drift
```
Drift metrics, JSON reports, and `feature_drift.png` are logged to the `credit-card-fraud-monitoring` experiment in MLflow.

### 7. Run Tests
```bash
uv run pytest -v
```
