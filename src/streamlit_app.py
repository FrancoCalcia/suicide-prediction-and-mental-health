import streamlit as st
import pandas as pd
from joblib import load
from pathlib import Path

st.set_page_config(page_title="Suicide Rate Predictor", page_icon="🧠", layout="centered")

BUNDLE_PATH = Path("models/rf_pipeline.joblib")
if not BUNDLE_PATH.exists():
    st.error("No encontré models/rf_pipeline.joblib. Entrená y guardá el pipeline con src/train_pipeline.py")
    st.stop()

bundle = load(BUNDLE_PATH)
pipe = bundle["pipeline"]
FEATS = bundle["features"]
TARGET = bundle["target"]

# Opciones limpias para selects
AGE_OPTS = ["5-14 years","15-24 years","25-34 years","35-54 years","55-74 years","75+ years"]
SEX_OPTS = ["male","female"]

st.title("🧠 Suicide Rate Predictor")
st.caption("Modelo RandomForest tunado (tasa por 100k habitantes).")

with st.form("single_pred"):
    st.subheader("Predicción individual")
    col1, col2 = st.columns(2)
    country = col1.text_input("Country", "Argentina")
    year = col2.number_input("Year", min_value=1985, max_value=2016, value=2016, step=1)
    sex = col1.selectbox("Sex", SEX_OPTS, index=0)
    age = col2.selectbox("Age", AGE_OPTS, index=3)
    submitted = st.form_submit_button("Predict")

    if submitted:
        X = pd.DataFrame([{"country": country, "year": year, "sex": sex, "age": age}])[FEATS]
        pred = pipe.predict(X)[0]
        st.success(f"Predicción: **{pred:.2f}** suicidios por 100k")

st.divider()

st.subheader("Batch (CSV)")
st.caption("Subí un CSV con columnas: country, year, sex, age")
file = st.file_uploader("CSV", type=["csv"])
if file is not None:
    df_in = pd.read_csv(file)
    # Validación mínima de columnas requeridas
    missing = [c for c in FEATS if c not in df_in.columns]
    if missing:
        st.error(f"Faltan columnas requeridas: {missing}")
    else:
        preds = pipe.predict(df_in[FEATS])
        out = df_in.copy()
        out["prediction_suicides_100k"] = preds
        st.success("¡Listo! Mostrando primeras filas:")
        st.dataframe(out.head(20), use_container_width=True)
        st.download_button("Descargar resultados (CSV)",
                           data=out.to_csv(index=False).encode("utf-8"),
                           file_name="predicciones.csv",
                           mime="text/csv")

#Local
# streamlit run src/streamlit_app.py
