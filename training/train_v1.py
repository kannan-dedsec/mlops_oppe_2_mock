import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("data/v0/clean/train.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="v1_clean_model"):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds)

    mlflow.log_metric("f1", f1)
    mlflow.sklearn.log_model(model, "model")

    joblib.dump(model, "models/v1/model.pkl")
