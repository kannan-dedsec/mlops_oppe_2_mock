from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import time

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

# -----------------------------
# CONFIGURATION
# -----------------------------
MLFLOW_TRACKING_URI = "http://34.55.250.137:8100"
CLEAN_MODEL_NAME = "fraud_clean_v0"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# -----------------------------
# OpenTelemetry
# -----------------------------
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer("fraud-api")

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="Fraud Detection API")

MODEL = None

# -----------------------------
# Load model on startup
# -----------------------------
@app.on_event("startup")
def load_model():
    global MODEL
    model_uri = f"models:/{CLEAN_MODEL_NAME}/latest"
    MODEL = mlflow.sklearn.load_model(model_uri)
    print(f"Loaded clean model: {CLEAN_MODEL_NAME}")

# -----------------------------
# Schemas
# -----------------------------
class PredictRequest(BaseModel):
    features: dict

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": CLEAN_MODEL_NAME
    }

@app.post("/predict")
def predict(req: PredictRequest):
    with tracer.start_as_current_span("predict_inference") as span:
        start = time.time()

        X = [list(req.features.values())]
        prediction = MODEL.predict(X)[0]

        latency_ms = (time.time() - start) * 1000
        span.set_attribute("latency_ms", latency_ms)

    return {
        "prediction": int(prediction),
        "latency_ms": latency_ms
    }

