import os
import time
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Fallback (mevcut dosyan)
from src.fallback import HeuristicModel

app = FastAPI(title="Course Completion Prediction API")

# ----------------------------
# Monitoring (Prometheus)
# ----------------------------
REQ_COUNT = Counter("request_count_total", "Total API requests")
REQ_LATENCY = Histogram("request_latency_seconds", "Request latency in seconds")
PRED_MODE = Counter("prediction_mode_total", "Predictions by mode", ["mode"])

# ----------------------------
# Model loading (safe)
# ----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")

model = None
model_loaded = False

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception:
    model = None
    model_loaded = False

fallback_model = HeuristicModel()

# ----------------------------
# Column alignment (to prevent KeyError)
# These match your trained pipeline's expected raw columns
# ----------------------------
EXPECTED_COLS = [
    "Student_ID",
    "Name",
    "Gender",
    "Age",
    "Education_Level",
    "Employment_Status",
    "City",
    "Device_Type",
    "Internet_Connection_Quality",
    "Course_ID",
    "Course_Name",
    "Category",
    "Course_Level",
    "Course_Duration_Days",
    "Instructor_Rating",
    "Login_Frequency",
    "Average_Session_Duration_Min",
    "Video_Completion_Rate",
    "Discussion_Participation",
    "Time_Spent_Hours",
    "Days_Since_Last_Login",
    "Notifications_Checked",
    "Peer_Interaction_Score",
    "Assignments_Submitted",
    "Assignments_Missed",
    "Quiz_Attempts",
    "Quiz_Score_Avg",
    "Project_Grade",
    "Progress_Percentage",
    "Rewatch_Count",
    "Enrollment_Date",
    "Payment_Mode",
    "Fee_Paid",
    "Discount_Used",
    "Payment_Amount",
    "App_Usage_Percentage",
    "Reminder_Emails_Clicked",
    "Support_Tickets_Raised",
    "Satisfaction_Rating",
]

# Numeric cols in the pipeline -> ensure numeric casting
NUMERIC_COLS = {
    "Age",
    "Course_Duration_Days",
    "Instructor_Rating",
    "Login_Frequency",
    "Average_Session_Duration_Min",
    "Video_Completion_Rate",
    "Discussion_Participation",
    "Time_Spent_Hours",
    "Days_Since_Last_Login",
    "Notifications_Checked",
    "Peer_Interaction_Score",
    "Assignments_Submitted",
    "Assignments_Missed",
    "Quiz_Attempts",
    "Quiz_Score_Avg",
    "Project_Grade",
    "Progress_Percentage",
    "Rewatch_Count",
    "Payment_Amount",
    "App_Usage_Percentage",
    "Reminder_Emails_Clicked",
    "Support_Tickets_Raised",
    "Satisfaction_Rating",
}


def _align_payload_to_df(payload: dict) -> pd.DataFrame:
    """
    Prevent KeyError by:
    - Keeping only EXPECTED_COLS
    - Filling missing cols with None
    - Casting numeric cols when possible (else None)
    """
    row = {}
    for col in EXPECTED_COLS:
        val = payload.get(col, None)

        if col in NUMERIC_COLS:
            if val is None or val == "":
                row[col] = None
            else:
                try:
                    row[col] = float(val)
                except Exception:
                    row[col] = None
        else:
            # categorical/text/date-like: keep string if provided
            if val is None:
                row[col] = None
            else:
                row[col] = str(val)

    return pd.DataFrame([row], columns=EXPECTED_COLS)


def _basic_guard(payload: dict):
    """
    Minimal safety checks to avoid huge payload / weird types.
    (Keeps demo stable without being overly strict.)
    """
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object (dict).")

    # Too many keys => suspicious / can slow demo
    if len(payload.keys()) > 200:
        raise ValueError("payload has too many fields.")

    # prevent extremely large strings
    for k, v in payload.items():
        if isinstance(v, str) and len(v) > 5000:
            raise ValueError(f"field '{k}' is too large.")

@app.middleware("http")
async def count_all_requests(request: Request, call_next):
    REQ_COUNT.inc()
    response = await call_next(request)
    return response


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH,
    }


@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict(request: Request):
    start = time.time()
    REQ_COUNT.inc()

    try:
        payload = await request.json()
        _basic_guard(payload)

        # Align columns to training schema
        df = _align_payload_to_df(payload)

        # If model not available -> fallback
        if not model_loaded or model is None:
            fb = fallback_model.predict(payload)
            PRED_MODE.labels(mode="fallback_no_model").inc()
            return {
                "prediction": int(fb.get("prediction", 0)),
                "probability": float(fb.get("probability", 0.0)),
                "meta": {"mode": "fallback", "reason": "model_not_loaded"},
            }

        # Try real prediction
        pred = model.predict(df)[0]
        PRED_MODE.labels(mode="model").inc()
        return {
            "prediction": int(pred),
            "meta": {"mode": "model"},
        }

    except Exception as e:
        # Any error => fallback (demo resilience)
        try:
            fb = fallback_model.predict(payload if isinstance(payload, dict) else {})
            PRED_MODE.labels(mode="fallback_error").inc()
            return {
                "prediction": int(fb.get("prediction", 0)),
                "probability": float(fb.get("probability", 0.0)),
                "meta": {"mode": "fallback", "reason": "exception", "error": str(e)},
            }
        except Exception:
            PRED_MODE.labels(mode="fallback_failed").inc()
            # Worst case: controlled response, demo doesn't crash
            return {
                "prediction": 0,
                "meta": {"mode": "fallback", "reason": "fallback_failed", "error": str(e)},
            }
    finally:
        REQ_LATENCY.observe(time.time() - start)
