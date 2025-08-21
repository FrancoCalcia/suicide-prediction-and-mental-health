# src/train_pipeline.py
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

DATA = Path("data/processed/suicides_clean.csv")
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 1) Cargar
df = pd.read_csv(DATA)

# 2) Features/target
FEATURES = ["country", "sex", "age", "year"]
TARGET = "suicides_100k_pop"
X = df[FEATURES]
y = df[TARGET]

# 3) Preprocesamiento
cat_cols = ["country", "sex", "age"]
num_cols = ["year"]

pre = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols),
    ],
    remainder="drop",
)

# 4) Modelo (mejores hiperparámetros encontrados)
rf = RandomForestRegressor(
    n_estimators=400,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features=0.7,
    max_depth=None,
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
)

pipe = Pipeline([("pre", pre), ("model", rf)])

# 5) Entrenar en TODO el dataset limpio
pipe.fit(X, y)

# 6) Guardar pipeline completo (pre + modelo)
dump({"pipeline": pipe, "features": FEATURES, "target": TARGET}, MODEL_DIR / "rf_pipeline.joblib")
print("✅ Guardado models/rf_pipeline.joblib")
