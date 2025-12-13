import mlflow
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, precision_score, recall_score
from fairlearn.metrics import demographic_parity_difference

# -----------------------------
# Config
# -----------------------------
MLFLOW_TRACKING_URI = "http://34.173.223.45:8100"
EXPERIMENT_NAME = "fraud-detection-1"

# Models
LOCATION_MODEL = "fraud_location_augmented"
CLEAN_MODEL = "fraud_clean_v0"

MODEL_STAGE = "None"  # use "None" when not using stages

# Columns
TARGET_COL = "Class"
SENSITIVE_COL = "location"

# Data paths
V0_LOCATION_PATH = "data/v0/with_location/train.csv"
V1_CLEAN_PATH = "data/v1/clean/train.csv"

# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# -----------------------------
# Load datasets
# -----------------------------
v0_loc = pd.read_csv(V0_LOCATION_PATH)
v1_clean = pd.read_csv(V1_CLEAN_PATH)

X_v0_loc = v0_loc.drop(columns=[TARGET_COL])
y_v0_loc = v0_loc[TARGET_COL]

X_v1_clean = v1_clean.drop(columns=[TARGET_COL])
y_v1_clean = v1_clean[TARGET_COL]

# -----------------------------
# Start analysis run
# -----------------------------
with mlflow.start_run(run_name="post_training_analysis"):

    # =====================================================
    # 1️⃣ Fairness + Explainability (location model)
    # =====================================================
    loc_model_uri = f"models:/{LOCATION_MODEL}/{MODEL_STAGE}"
    loc_model = mlflow.pyfunc.load_model(loc_model_uri)

    v0_loc_preds = loc_model.predict(X_v0_loc)

    # ---- Fairness
    dp_diff = demographic_parity_difference(
        y_pred=v0_loc_preds,
        sensitive_features=X_v0_loc[SENSITIVE_COL],
    )

    mlflow.log_metric("demographic_parity_difference", dp_diff)

    # ---- SHAP
    try:
        underlying_model = loc_model._model_impl
    except AttributeError:
        underlying_model = loc_model

    explainer = shap.Explainer(underlying_model, X_v0_loc)
    shap_values = explainer(X_v0_loc)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_v0_loc, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    plt.close()

    mlflow.log_artifact("shap_summary.png")

    # =====================================================
    # 2️⃣ Concept Drift (clean v0 model → v1 data)
    # =====================================================
    clean_model_uri = f"models:/{CLEAN_MODEL}/{MODEL_STAGE}"
    clean_model = mlflow.pyfunc.load_model(clean_model_uri)

    v1_preds = clean_model.predict(X_v1_clean)

    mlflow.log_metric("f1_v1_drift", f1_score(y_v1_clean, v1_preds))
    mlflow.log_metric("precision_v1_drift", precision_score(y_v1_clean, v1_preds))
    mlflow.log_metric("recall_v1_drift", recall_score(y_v1_clean, v1_preds))

    # =====================================================
    # Metadata
    # =====================================================
    mlflow.set_tag("analysis_type", "explainability_fairness_drift")
    mlflow.set_tag("fairness_model", LOCATION_MODEL)
    mlflow.set_tag("drift_model", CLEAN_MODEL)
    mlflow.set_tag("sensitive_attribute", SENSITIVE_COL)
