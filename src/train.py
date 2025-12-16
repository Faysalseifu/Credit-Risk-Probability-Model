"""Model training script."""
import argparse
from pathlib import Path

from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.data_processing import TARGET_COLUMN, build_preprocessing_pipeline, load_raw, split_features_target


def train_model(input_path: str, output_path: str) -> Path:
    """Train classifier with full preprocessing pipeline and persist it."""
    df = load_raw(input_path)
    X, y = split_features_target(df, target=TARGET_COLUMN)

    preprocessing = build_preprocessing_pipeline()
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    pipeline = Pipeline([("preprocess", preprocessing), ("model", model)])
    pipeline.fit(X, y)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dump(pipeline, output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train credit risk model")
    parser.add_argument("--input", default="data/raw/data.csv", help="Path to raw training data")
    parser.add_argument("--output", default="artifacts/model.joblib", help="Where to save the trained model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = train_model(args.input, args.output)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
