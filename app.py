from fastapi import FastAPI, Request, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import time
import json

import mlflow
import mlflow.sklearn

# -----------------------------
# OpenTelemetry imports
# -----------------------------
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# -----------------------------
# Configuration
# -----------------------------
MLFLOW_TRACKING_URI = "http://YOUR_MLFLOW_SERVER:5000"
MODEL_NAME = "fraud_clean_v0"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# -----------------------------
# Tracer setup (Cloud Trace)
# -----------------------------
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# -----------------------------
# Structured logging
# -----------------------------
logger = logging.getLogger("ml-model-service")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter(json.dumps({
    "severity": "%(levelname)s",
    "message": "%(message)s",
    "timestamp": "%(asctime)s"
}))
handler.setFormatter(formatter)
logger.addHandler(handler)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Fraud Detection Service")

MODEL = None
app_state = {
    "is_ready": False,
    "is_alive": True
}

# -----------------------------
# Input schema
# -----------------------------
class PredictRequest(BaseModel):
    features: dict

# -----------------------------
# Startup: load model
# -----------------------------
@app.on_event("startup")
async def startup_event():
    global MODEL
    try:
        start = time.time()
        model_uri = f"models:/{MODEL_NAME}/latest"
        MODEL = mlflow.sklearn.load_model(model_uri)

        app_state["is_ready"] = True

        logger.info(json.dumps({
            "event": "model_loaded",
            "model_name": MODEL_NAME,
            "load_time_ms": round((time.time() - start) * 1000, 2)
        }))
    except Exception as e:
        logger.exception(json.dumps({
            "event": "model_load_failed",
            "error": str(e)
        }))
        app_state["is_alive"] = False
        raise

# -----------------------------
# Probes
# -----------------------------
@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

# -----------------------------
# Middleware: latency header
# -----------------------------
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response

# -----------------------------
# Global exception handler
# -----------------------------
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")

    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id},
    )

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
async def predict(request_body: PredictRequest, request: Request):
    if not app_state["is_ready"]:
        raise HTTPException(status_code=503, detail="Model not ready")

    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")

        try:
            features = request_body.features

            # sklearn Pipeline handles preprocessing internally
            X = [list(features.values())]
            prediction = int(MODEL.predict(X)[0])

            latency = round((time.time() - start_time) * 1000, 2)

            span.set_attribute("latency_ms", latency)
            span.set_attribute("model_name", MODEL_NAME)

            logger.info(json.dumps({
                "event": "prediction",
                "trace_id": trace_id,
                "input_features": features,
                "prediction": prediction,
                "latency_ms": latency,
                "status": "success"
            }))

            return {
                "prediction": prediction,
                "latency_ms": latency,
                "trace_id": trace_id
            }

        except Exception as e:
            logger.exception(json.dumps({
                "event": "prediction_error",
                "trace_id": trace_id,
                "error": str(e)
            }))
            raise HTTPException(status_code=500, detail="Prediction failed")
