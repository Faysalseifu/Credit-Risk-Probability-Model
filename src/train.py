"""Model training script."""
import argparse
from pathlib import Path

from joblib import dump
from sklearn.ensemble import RandomForestClassifier

from src.data_processing import load_raw, preprocess


def train_model(input_path: str, output_path: str) -> Path:
    """Train a simple classifier and persist it to disk."""
    df = load_raw(input_path)
    X, y = preprocess(df)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dump(model, output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train credit risk model")
    parser.add_argument("--input", default="data/processed/train.csv", help="Path to processed training data")
    parser.add_argument("--output", default="artifacts/model.joblib", help="Where to save the trained model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = train_model(args.input, args.output)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
