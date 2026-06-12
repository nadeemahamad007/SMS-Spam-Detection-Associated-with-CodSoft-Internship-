from __future__ import annotations

import argparse
from pathlib import Path

import joblib


DEFAULT_MODEL_PATH = Path("models/spam_classifier.joblib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict whether an SMS message is spam or ham.")
    parser.add_argument("message", type=str, help="SMS text to classify.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the trained model. Defaults to {DEFAULT_MODEL_PATH}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model)
    prediction = int(model.predict([args.message])[0])
    label = "spam" if prediction == 1 else "ham"
    print(label)


if __name__ == "__main__":
    main()
