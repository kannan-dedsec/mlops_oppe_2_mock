import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

MLFLOW_EXPERIMENT_NAME = "fraud-detection"
TARGET_COL = "Class"

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def load_data(train_path, val_path):
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]

    X_val = val.drop(columns=[TARGET_COL])
    y_val = val[TARGET_COL]

    return X_train, y_train, X_val, y_val


def train_and_log(model_name, train_path, val_path, extra_params=None):
    X_train, y_train, X_val, y_val = load_data(train_path, val_path)

    with mlflow.start_run(run_name=model_name):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        f1 = f1_score(y_val, preds)

        mlflow.log_metric("f1_score", f1)

        if extra_params:
            mlflow.log_params(extra_params)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name
        )

        print(f"Logged model: {model_name} | F1: {f1:.4f}")


def main():
    # Clean model
    train_and_log(
        model_name="fraud_clean_v0",
        train_path="data/v0/clean/train.csv",
        val_path="data/v0/clean/val.csv",
        extra_params={"dataset": "clean"}
    )

    # Poisoned models
    poison_configs = {
        "fraud_poisoned_2": "data/v0/poisoned_2_percent",
        "fraud_poisoned_8": "data/v0/poisoned_8_percent",
        "fraud_poisoned_20": "data/v0/poisoned_20_percent",
    }

    for name, path in poison_configs.items():
        train_and_log(
            model_name=name,
            train_path=f"{path}.csv",
            val_path=f"data/v0/clean/val.csv",
            extra_params={"dataset": name}
        )

    # Location-augmented model (optional)
    try:
        train_and_log(
            model_name="fraud_location_augmented",
            train_path="data/v0/with_location/train.csv",
            val_path="data/v0/with_location/val.csv",
            extra_params={"dataset": "location_augmented"}
        )
    except FileNotFoundError:
        print("Location dataset not found, skipping.")


if __name__ == "__main__":
    main()
