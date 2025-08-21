# src/predict_cli.py
import argparse, json
import pandas as pd
from joblib import load
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--sex", choices=["male","female"], required=True)
    parser.add_argument("--age", required=True)  # ej: "35-54 years"
    args = parser.parse_args()

    bundle = load(Path("models/rf_pipeline.joblib"))
    pipe = bundle["pipeline"]; feats = bundle["features"]

    X = pd.DataFrame([{
        "country": args.country,
        "year": args.year,
        "sex": args.sex,
        "age": args.age
    }])[feats]

    pred = pipe.predict(X)[0]
    print(json.dumps({"prediction_suicides_100k": float(pred)}, indent=2))

if __name__ == "__main__":
    main()

# Usage example:
# python src/predict_cli.py --country Argentina --year 2011 --sex male --age "35-54 years"
