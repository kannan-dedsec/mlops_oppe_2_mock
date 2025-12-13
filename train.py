import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# -----------------------------
# CONFIGURATION
# -----------------------------
MLFLOW_TRACKING_URI = "http://34.55.250.137:8100"
EXPERIMENT_NAME = "fraud-detection"
TARGET_COL = "Class"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# -----------------------------
# Utilities
# -----------------------------
def load_data(train_path, val_path):
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]

    X_val = val.drop(columns=[TARGET_COL])
    y_val = val[TARGET_COL]

    return X_train, y_train, X_val, y_val


def build_pipeline(X):
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    model = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def train_and_log(model_name, train_path, val_path, metadata):
    X_train, y_train, X_val, y_val = load_data(train_path, val_path)

    pipeline = build_pipeline(X_train)

    with mlflow.start_run(run_name=model_name):
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_val)
        f1 = f1_score(y_val, preds)

        mlflow.log_metric("f1_score", f1)
        mlflow.log_params(metadata)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name=model_name
        )

        print(f"✅ Logged model: {model_name} | F1={f1:.4f}")


# -----------------------------
# Main
# -----------------------------
def main():
    # Clean model
    train_and_log(
        model_name="fraud_clean_v0",
        train_path="data/v0/clean/train.csv",
        val_path="data/v0/clean/val.csv",
        metadata={"dataset": "clean", "poisoning": 0}
    )

    # Poisoned models
    poison_configs = {
        "fraud_poisoned_2": ("data/v0/poisoned_2_percent", 2),
        "fraud_poisoned_8": ("data/v0/poisoned_8_percent", 8),
        "fraud_poisoned_20": ("data/v0/poisoned_20_percent", 20),
    }

    for model_name, (path, level) in poison_configs.items():
        train_and_log(
            model_name=model_name,
            train_path=f"{path}.csv",
            val_path=f"data/v0/clean/val.csv",
            metadata={"dataset": "poisoned", "poisoning": level}
        )

    # Location-augmented model
    try:
        train_and_log(
            model_name="fraud_location_augmented",
            train_path="data/v0/with_location/train.csv",
            val_path="data/v0/with_location/val.csv",
            metadata={"dataset": "location_augmented"}
        )
    except FileNotFoundError:
        print("⚠️ Location dataset not found, skipping.")


if __name__ == "__main__":
    main()
