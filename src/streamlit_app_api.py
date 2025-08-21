import os
import streamlit as st
import pandas as pd
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Suicide Rate Predictor", page_icon="🧠", layout="centered")
st.title("🧠 Suicide Rate Predictor (via FastAPI)")
st.caption(f"Endpoint: {API_URL}")

AGE_OPTS = ["5-14 years","15-24 years","25-34 years","35-54 years","55-74 years","75+ years"]
SEX_OPTS = ["male","female"]

with st.form("single"):
    col1, col2 = st.columns(2)
    country = col1.text_input("Country", "Argentina")
    year    = col2.number_input("Year", min_value=1979, max_value=2016, value=2016, step=1)
    sex     = col1.selectbox("Sex", SEX_OPTS, index=0)
    age     = col2.selectbox("Age", AGE_OPTS, index=3)
    submit  = st.form_submit_button("Predict")

    if submit:
        payload = {"records":[{"country": country, "year": int(year), "sex": sex, "age": age}]}
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            r.raise_for_status()
            pred = r.json()["predictions"][0]
            st.success(f"Predicción: **{pred:.2f}** suicidios por 100k")
        except Exception as e:
            st.error(f"Error llamando a la API: {e}")

st.divider()
st.subheader("Batch (CSV → API)")
file = st.file_uploader("CSV con country,year,sex,age", type=["csv"])
if file is not None:
    df = pd.read_csv(file)
    required = {"country","year","sex","age"}
    if not required.issubset(df.columns):
        faltan = list(required - set(df.columns))
        st.error(f"Faltan columnas requeridas: {faltan}")
    else:
        payload = {"records": df[list(required)].to_dict(orient="records")}
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=120)
            r.raise_for_status()
            preds = r.json()["predictions"]
            out = df.copy(); out["prediction_suicides_100k"] = preds
            st.dataframe(out.head(20), use_container_width=True)
            st.download_button("Descargar resultados (CSV)",
                               data=out.to_csv(index=False).encode("utf-8"),
                               file_name="predicciones.csv",
                               mime="text/csv")
        except Exception as e:
            st.error(f"Error llamando a la API: {e}")
