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

LOCATION_MODEL = "fraud_location_augmented"
CLEAN_MODEL = "fraud_clean_v0"
MODEL_STAGE = "None"

TARGET_COL = "Class"
SENSITIVE_COL = "location"

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

X_v0 = v0_loc.drop(columns=[TARGET_COL])
y_v0 = v0_loc[TARGET_COL]

X_v1 = v1_clean.drop(columns=[TARGET_COL])
y_v1 = v1_clean[TARGET_COL]

# 🔑 SHAP-safe dataset (drop categorical column)
X_v0_shap = X_v0.drop(columns=[SENSITIVE_COL])

# -----------------------------
# Start analysis run
# -----------------------------
with mlflow.start_run(run_name="post_training_analysis"):

    # =====================================================
    # 1️⃣ Fairness + Explainability (location model)
    # =====================================================
    loc_model = mlflow.pyfunc.load_model(
        f"models:/{LOCATION_MODEL}/{MODEL_STAGE}"
    )

    # ---- Predictions (FULL features, incl. location)
    v0_preds = loc_model.predict(X_v0)

    # ---- Fairness (v0 only)
    dp_diff = demographic_parity_difference(
        y_true=y_v0,
        y_pred=v0_preds,
        sensitive_features=X_v0[SENSITIVE_COL],
    )

    mlflow.log_metric("demographic_parity_difference", dp_diff)

    # ---- SHAP (NUMERIC FEATURES ONLY)
    background = X_v0_shap.sample(
        n=min(100, len(X_v0_shap)),
        random_state=42
    )

    # Wrapper so SHAP sees correct input
    def predict_wrapper(X):
        X_full = X.copy()
        X_full[SENSITIVE_COL] = X_v0[SENSITIVE_COL].iloc[:len(X)].values
        return loc_model.predict(X_full)

    explainer = shap.Explainer(
        predict_wrapper,
        background
    )

    shap_values = explainer(X_v0_shap)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_v0_shap, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    plt.close()

    mlflow.log_artifact("shap_summary.png")

    # =====================================================
    # 2️⃣ Concept Drift (clean v0 model → v1 data)
    # =====================================================
    clean_model = mlflow.pyfunc.load_model(
        f"models:/{CLEAN_MODEL}/{MODEL_STAGE}"
    )

    v1_preds = clean_model.predict(X_v1)

    mlflow.log_metric("f1_v1_drift", f1_score(y_v1, v1_preds))
    mlflow.log_metric("precision_v1_drift", precision_score(y_v1, v1_preds))
    mlflow.log_metric("recall_v1_drift", recall_score(y_v1, v1_preds))

    # =====================================================
    # Metadata
    # =====================================================
    mlflow.set_tag("analysis_type", "explainability_fairness_drift")
    mlflow.set_tag("fairness_model", LOCATION_MODEL)
    mlflow.set_tag("drift_model", CLEAN_MODEL)
    mlflow.set_tag("sensitive_attribute", SENSITIVE_COL)
