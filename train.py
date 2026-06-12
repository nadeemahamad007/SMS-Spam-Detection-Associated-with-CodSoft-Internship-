from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DEFAULT_DATA_PATH = Path("data/raw/spam.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")


def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def load_dataset(data_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(data_path, encoding="latin-1")
    dataframe = dataframe.iloc[:, :2].copy()
    dataframe.columns = ["label", "message"]
    dataframe["message"] = dataframe["message"].fillna("").map(clean_text)
    dataframe["label"] = dataframe["label"].map({"ham": 0, "spam": 1})
    dataframe = dataframe.dropna(subset=["label"]).reset_index(drop=True)
    return dataframe


def build_models() -> dict[str, Pipeline]:
    base_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    return {
        "MultinomialNB": Pipeline(
            [
                ("vectorizer", base_vectorizer),
                ("classifier", MultinomialNB()),
            ]
        ),
        "LogisticRegression": Pipeline(
            [
                ("vectorizer", base_vectorizer),
                ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "LinearSVC": Pipeline(
            [
                ("vectorizer", base_vectorizer),
                ("classifier", LinearSVC(class_weight="balanced")),
            ]
        ),
    }


def evaluate_models(X_train: pd.Series, y_train: pd.Series) -> tuple[str, dict[str, float]]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores: dict[str, float] = {}

    for name, pipeline in build_models().items():
        f1_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
        scores[name] = float(f1_scores.mean())

    best_model_name = max(scores, key=scores.get)
    return best_model_name, scores


def save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["ham", "spam"],
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_title("SMS Spam Detection Confusion Matrix")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def train(data_path: Path) -> dict[str, object]:
    dataframe = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        dataframe["message"],
        dataframe["label"],
        test_size=0.2,
        random_state=42,
        stratify=dataframe["label"],
    )

    best_model_name, cv_scores = evaluate_models(X_train, y_train)
    best_model = build_models()[best_model_name]
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "spam_classifier.joblib"
    joblib.dump(best_model, model_path)

    report = classification_report(y_test, predictions, target_names=["ham", "spam"], output_dict=True)
    metrics = {
        "dataset_rows": int(len(dataframe)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "best_model": best_model_name,
        "cv_f1_scores": cv_scores,
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_precision": float(precision_score(y_test, predictions)),
        "test_recall": float(recall_score(y_test, predictions)),
        "test_f1_score": float(f1_score(y_test, predictions)),
        "classification_report": report,
        "model_path": str(model_path.as_posix()),
    }

    metrics_path = REPORT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix(y_test, predictions, REPORT_DIR / "confusion_matrix.png")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the SMS spam detection model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to the dataset CSV file. Defaults to {DEFAULT_DATA_PATH}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train(args.data)
    summary = {
        "best_model": metrics["best_model"],
        "test_accuracy": round(metrics["test_accuracy"], 4),
        "test_precision": round(metrics["test_precision"], 4),
        "test_recall": round(metrics["test_recall"], 4),
        "test_f1_score": round(metrics["test_f1_score"], 4),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
