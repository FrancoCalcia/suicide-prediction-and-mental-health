# src/app.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
from joblib import load
from pathlib import Path

BUNDLE = load(Path("models/rf_pipeline.joblib"))
PIPE = BUNDLE["pipeline"]
FEATS = BUNDLE["features"]

class Record(BaseModel):
    country: str
    year: int = Field(ge=1900, le=2100)
    sex: str  # "male" | "female"
    age: str  # ej: "35-54 years"

class PredictRequest(BaseModel):
    records: List[Record]

class PredictResponse(BaseModel):
    predictions: List[float]

app = FastAPI(title="Suicide Rate Predictor", version="1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    df = pd.DataFrame([r.model_dump() for r in req.records])[FEATS]
    preds = PIPE.predict(df)
    return {"predictions": [float(p) for p in preds]}

# Local
# uvicorn src.app:app --reload --port 8000

#Prueba
#curl -X POST "http://127.0.0.1:8000/predict" \
#  -H "Content-Type: application/json" \
#  -d '{"records":[{"country":"Argentina","year":2016,"sex":"male","age":"35-54 years"},
#                  {"country":"Uruguay","year":2010,"sex":"female","age":"55-74 years"}]}'
