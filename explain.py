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

MODEL_NAME = "fraud_location_augmented"   # registered model name
MODEL_STAGE = "Production"                # or "None"/"Staging"

TARGET_COL = "Class"
SENSITIVE_COL = "location"

V0_PATH = "data/clean_v0.csv"
V1_PATH = "data/clean_v1.csv"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# -----------------------------
# Load datasets
# -----------------------------
v0 = pd.read_csv(V0_PATH)
v1 = pd.read_csv(V1_PATH)

X_v0 = v0.drop(columns=[TARGET_COL])
y_v0 = v0[TARGET_COL]

X_v1 = v1.drop(columns=[TARGET_COL])
y_v1 = v1[TARGET_COL]

# -----------------------------
# Load model from MLflow
# -----------------------------
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
model = mlflow.pyfunc.load_model(model_uri)

print(f"Loaded model from {model_uri}")

# -----------------------------
# Start analysis run
# -----------------------------
with mlflow.start_run(run_name="post_training_analysis"):

    # -------------------------
    # Predictions
    # -------------------------
    v0_preds = model.predict(X_v0)
    v1_preds = model.predict(X_v1)

    # -------------------------
    # Performance metrics
    # -------------------------
    mlflow.log_metric("f1_v0", f1_score(y_v0, v0_preds))
    mlflow.log_metric("precision_v0", precision_score(y_v0, v0_preds))
    mlflow.log_metric("recall_v0", recall_score(y_v0, v0_preds))

    mlflow.log_metric("f1_v1", f1_score(y_v1, v1_preds))
    mlflow.log_metric("precision_v1", precision_score(y_v1, v1_preds))
    mlflow.log_metric("recall_v1", recall_score(y_v1, v1_preds))

    # -------------------------
    # Fairness audit
    # -------------------------
    dp_diff = demographic_parity_difference(
        y_pred=v0_preds,
        sensitive_features=X_v0[SENSITIVE_COL],
    )

    mlflow.log_metric("demographic_parity_difference", dp_diff)

    # -------------------------
    # SHAP explainability
    # -------------------------
    # IMPORTANT:
    # For pyfunc models, extract the underlying model if needed
    try:
        underlying_model = model._model_impl
    except AttributeError:
        underlying_model = model

    explainer = shap.Explainer(underlying_model, X_v0)
    shap_values = explainer(X_v0)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_v0, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    plt.close()

    mlflow.log_artifact("shap_summary.png")

    # -------------------------
    # Metadata
    # -------------------------
    mlflow.set_tag("analysis_type", "explainability_fairness_drift")
    mlflow.set_tag("model_source", MODEL_NAME)
    mlflow.set_tag("sensitive_attribute", SENSITIVE_COL)
