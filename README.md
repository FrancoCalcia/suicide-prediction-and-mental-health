
# 🧠 Suicide Prediction and Mental Health

Proyecto de Ciencia de Datos y MLOps para analizar y predecir la **tasa de suicidios por 100k habitantes** a partir de datos históricos de la OMS (1979–2016).

Incluye:

* 📊 **EDA (Exploratory Data Analysis)** con visualizaciones (mapas, rankings, evolución temporal).
* 🤖 **Modelado predictivo** con Random Forest, XGBoost y Gradient Boosting.
* ⚙️ **Optimización de hiperparámetros** y evaluación con curvas de aprendizaje y métricas.
* 🔍 **Interpretabilidad** con Feature Importances y SHAP.
* 🌐 **API en FastAPI** para servir el modelo.
* 🎛 **UI en Streamlit** para probar predicciones de forma interactiva.
* 🐳 **Docker Compose** para levantar todo el stack (API + UI).

---

## 📂 Estructura del proyecto

```
suicide-prediction-and-mental-health/
│
├── data/                  # Datos raw y procesados
│   ├── raw/
│   └── processed/
│
├── notebooks/             # Notebooks de análisis y modelado
│   └── 02_modelado.ipynb
│
├── models/                # Modelos entrenados (.joblib)
│
├── src/                   # Código fuente
│   ├── make_dataset.py    # Limpieza y guardado del dataset
│   ├── train_pipeline.py  # Entrenamiento y pipeline
│   ├── predict_cli.py     # Predicción desde terminal
│   ├── app.py             # API con FastAPI
│   ├── streamlit_app.py   # UI standalone
│   └── streamlit_app_api.py # UI que consume la API
│
├── requirements.txt       # Dependencias base
├── requirements-api.txt   # Dependencias para API/UI
├── Dockerfile             # Imagen de API
├── Dockerfile.streamlit   # Imagen de UI
├── docker-compose.yml     # Orquestación
└── README.md              # Este archivo
```

---

## 📊 Exploratory Data Analysis (EDA)

* Dataset con **43776 filas** y 6 columnas: `country`, `year`, `sex`, `age`, `suicides_no`, `population`.
* Visualizaciones principales:

  * Mapa mundial de tasas promedio (1979–2016).
  * Top/Bottom 10 países.
  * Evolución temporal de países seleccionados.
* Hallazgos:

  * Europa del Este lidera con las tasas más altas (Hungría, Lituania, Rusia).
  * América Latina muestra valores moderados (Uruguay \~18, Argentina \~10, Brasil \~6).
  * Algunos países con tasas bajas probablemente reflejan **falta de datos**.

---

## 🤖 Modelado

Se probaron 3 algoritmos:

| Modelo               | MAE  | RMSE | R²    |
| -------------------- | ---- | ---- | ----- |
| Random Forest        | 2.70 | 6.56 | 0.887 |
| HistGradientBoosting | 4.73 | 8.39 | 0.815 |
| XGBoost              | 4.29 | 7.52 | 0.852 |

👉 **Random Forest fue el mejor modelo.**

### 🔧 Tuning

* `RandomSearchCV` mejoró el equilibrio train/test, reduciendo overfitting.
* Curvas de aprendizaje mostraron alta varianza en datos pequeños, pero buena generalización en datos completos.

---

## 🔍 Interpretabilidad

* Variables más importantes:

  * `sex` (los hombres muestran mayor riesgo).
  * `year` (tendencia temporal).
  * Rango etario (más incidencia en adultos jóvenes y mayores).
  * Países del Este europeo.


---

## 🌐 API con FastAPI

Endpoints:

* `GET /` → mensaje de bienvenida.
* `POST /predict` → recibe JSON con `country, year, sex, age` y devuelve predicción.

Ejemplo request:

```json
{
  "records": [
    {"country": "Argentina", "year": 2016, "sex": "male", "age": "25-34 years"}
  ]
}
```

Ejemplo response:

```json
{
  "predictions": [12.45]
}
```

Correr localmente:

```bash
uvicorn src.app:app --reload --port 8000
```

Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🎛 UI con Streamlit

Standalone:

```bash
streamlit run src/streamlit_app.py
```

Via API:

```bash
streamlit run src/streamlit_app_api.py
```

Disponible en [http://localhost:8501](http://localhost:8501)

---

## 🐳 Docker Compose

Levantar todo el stack:

```bash
docker compose up --build
```

Servicios:

* API → [http://localhost:8000/docs](http://localhost:8000/docs)
* UI  → [http://localhost:8501](http://localhost:8501)

