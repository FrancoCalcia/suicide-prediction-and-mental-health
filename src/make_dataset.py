# src/make_dataset.py
import pandas as pd
from pathlib import Path
from config import RAW_DIR, PROC_DIR

VALID_START, VALID_END = 1985, 2016

def load_raw() -> pd.DataFrame:
    candidates = [RAW_DIR / "suicide.csv"]
    for p in candidates:
        print("Probando:", p)
        if p.is_file():             
            print("✅ Encontrado:", p)
            return pd.read_csv(p)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Normalizar nombres
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_").replace("/", "_") for c in df.columns]

    # 2) Tipos y normalizaciones
    # year
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # suicides_no & population como enteros con nulos
    for c in ("suicides_no", "population"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # strings/categorías
    if "country" in df.columns:
        df["country"] = df["country"].astype("string").str.strip()
    if "sex" in df.columns:
        df["sex"] = df["sex"].astype("category")
    if "age" in df.columns:
        age_order = ["5-14 years","15-24 years","25-34 years","35-54 years","55-74 years","75+ years"]
        df["age"] = df["age"].astype("string").str.strip()
        df["age"] = pd.Categorical(df["age"], categories=age_order, ordered=True)

    # 3) Crear/renombrar tasa estandarizada
    if "suicides_100k_pop" not in df.columns:
        if "suicides_100k_pop" in df.columns:
            pass  # ya está con buen nombre
        elif "suicides_100k_pop" not in df.columns and "suicides_100k_pop" not in df.columns:
            # nada
            pass
    # rename si viene como suicides/100k_pop
    if "suicides_100k_pop" not in df.columns and "suicides_100k_pop" in df.columns:
        df = df.rename(columns={"suicides/100k_pop": "suicides_100k_pop"})
    if "suicides_100k_pop" not in df.columns:
        if {"suicides_no","population"}.issubset(df.columns):
            df["suicides_100k_pop"] = (df["suicides_no"] / df["population"] * 100000).astype("float")

    # 4) Filtrar rango de años confiable
    if "year" in df.columns:
        df = df[(df["year"] >= VALID_START) & (df["year"] <= VALID_END)]

    return df

def main():
    df_raw = load_raw()
    df = clean_data(df_raw)

    PROC_DIR.mkdir(parents=True, exist_ok=True)

    # Separar filas inválidas para cálculo de tasa
    if {"suicides_no","population"}.issubset(df.columns):
        mask_valid = (~df["suicides_no"].isna()) & (~df["population"].isna()) & (df["population"] > 0)
        df_valid = df[mask_valid].copy()
        df_nulls = df[~mask_valid].copy()
    else:
        df_valid, df_nulls = df.copy(), pd.DataFrame()

    # Guardados
    df.to_csv(PROC_DIR / "suicides_clean_full.csv", index=False)
    df_valid.to_csv(PROC_DIR / "suicides_clean.csv", index=False)
    if not df_nulls.empty:
        df_nulls.to_csv(PROC_DIR / "rows_with_nulls.csv", index=False)

    # Log rápido
    print("✅ Guardado:")
    print("  -", PROC_DIR / "suicides_clean.parquet", f"({len(df_valid):,} filas válidas)")
    print("  -", PROC_DIR / "suicides_clean_full.parquet", f"({len(df):,} filas totales)")
    if not df_nulls.empty:
        print("  -", PROC_DIR / "rows_with_nulls.csv", f"({len(df_nulls):,} filas con nulos)")

if __name__ == "__main__":
    main()
