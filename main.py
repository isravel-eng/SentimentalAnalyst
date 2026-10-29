

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "sentiment_sample.csv"
SENTIMENT140_DATA = ROOT / "data" / "training.1600000.processed.noemoticon.csv"
DEFAULT_MODEL = ROOT / "models" / "sentiment_model.pkl"
DEFAULT_REPORT = ROOT / "reports" / "evaluation.txt"
DEFAULT_IMAGE_REPORT = ROOT / "reports" / "evaluation_report.png"
SENTIMENT140_URL = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
SENTIMENT140_CSV = "training.1600000.processed.noemoticon.csv"
LABELS = ("negative", "neutral", "positive")


def load_data(path: Path, text_column: str, label_column: str) -> pd.DataFrame:
    """Load and validate the two columns needed for training."""
    data = pd.read_csv(path)
    missing = {text_column, label_column} - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing column(s) {', '.join(sorted(missing))}. "
            f"Available columns: {', '.join(data.columns)}"
        )

    data = data[[text_column, label_column]].dropna().copy()
    data[text_column] = data[text_column].astype(str).str.strip()
    data[label_column] = data[label_column].astype(str).str.strip().str.lower()
    data = data[data[text_column] != ""]
    unexpected = sorted(set(data[label_column]) - set(LABELS))
    if unexpected:
        raise ValueError(
            f"Unknown sentiment label(s): {', '.join(unexpected)}. "
            f"Use only: {', '.join(LABELS)}. Map your dataset's labels first."
        )
    if data[label_column].nunique() < 2:
        raise ValueError("Training data needs at least two different sentiment labels.")
    if data[label_column].value_counts().min() < 2:
        raise ValueError("Each sentiment needs at least two rows for a train/test split.")
    return data


def load_sentiment140(path: Path, max_samples: int) -> pd.DataFrame:
    """Download Sentiment140 from Stanford if needed and convert its labels."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        archive = path.parent / "sentiment140.zip"
        print("Downloading Sentiment140 from Stanford (about 80 MB)...")
        try:
            urlretrieve(SENTIMENT140_URL, archive)
            with ZipFile(archive) as zipped:
                member = next(name for name in zipped.namelist() if name.endswith(SENTIMENT140_CSV))
                zipped.extract(member, path.parent)
                extracted = path.parent / member
                if extracted != path:
                    extracted.replace(path)
        except Exception as error:
            raise RuntimeError(
                "Could not download Sentiment140. Check your internet connection, or download "
                f"{SENTIMENT140_CSV} from the Sentiment140 project page and place it at {path}."
            ) from error
        finally:
            archive.unlink(missing_ok=True)

    columns = ["sentiment_id", "id", "date", "query", "user", "text"]
    data = pd.read_csv(path, encoding="latin-1", header=None, names=columns)
    data = data[["text", "sentiment_id"]].rename(columns={"sentiment_id": "sentiment"})
    data["sentiment"] = data["sentiment"].map({0: "negative", 4: "positive"})
    data = data.dropna(subset=["text", "sentiment"])
    if max_samples > 0 and len(data) > max_samples:
        per_class = max(1, max_samples // data["sentiment"].nunique())
        data = data.groupby("sentiment", group_keys=False).sample(
            n=per_class, random_state=42
        )
    return data.reset_index(drop=True)


def load_imdb(max_samples: int) -> pd.DataFrame:
    """Load the built-in Keras IMDB reviews and decode their word sequences."""
    try:
        from tensorflow.keras.datasets import imdb
    except ImportError as error:
        raise RuntimeError(
            "The Keras IMDB option needs TensorFlow. Install it with: "
            "python -m pip install -r requirements-imdb.txt"
        ) from error

    (reviews, labels), _ = imdb.load_data(num_words=20_000)
    word_index = imdb.get_word_index()
    index_to_word = {index + 3: word for word, index in word_index.items()}
    index_to_word.update({0: "", 1: "", 2: "unknown", 3: ""})
    texts = [" ".join(index_to_word.get(token, "unknown") for token in review) for review in reviews]
    data = pd.DataFrame(
        {"text": texts, "sentiment": ["positive" if label == 1 else "negative" for label in labels]}
    )
    if max_samples > 0 and len(data) > max_samples:
        per_class = max(1, max_samples // 2)
        data = data.groupby("sentiment", group_keys=False).sample(
            n=per_class, random_state=42
        )
    return data.reset_index(drop=True)


def make_model() -> Pipeline:
    """Turn text into TF-IDF features, then classify them with logistic regression."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    max_features=20_000,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1_000, class_weight="balanced")),
        ]
    )


def save_image_report(
    path: Path,
    dataset_name: str,
    accuracy: float,
    report_data: dict,
    matrix,
    labels: list[str],
) -> None:
    """Save a readable PNG with per-class metrics and the confusion matrix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = ("precision", "recall", "f1-score")
    class_metrics = {
        label: [report_data[label][metric] for metric in metric_names]
        for label in labels
    }

    figure, (metrics_axis, matrix_axis) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ("#4C78A8", "#F58518", "#54A24B")
    bar_width = 0.24
    positions = range(len(labels))
    for index, metric in enumerate(metric_names):
        values = [class_metrics[label][index] for label in labels]
        offsets = [position + (index - 1) * bar_width for position in positions]
        metrics_axis.bar(offsets, values, bar_width, label=metric.replace("-", " "), color=colors[index])
    metrics_axis.set_xticks(list(positions), labels)
    metrics_axis.set_ylim(0, 1.05)
    metrics_axis.set_ylabel("Score")
    metrics_axis.set_title("Per-class evaluation scores")
    metrics_axis.legend(loc="lower right")
    metrics_axis.grid(axis="y", alpha=0.2)

    image = matrix_axis.imshow(matrix, cmap="Blues")
    matrix_axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    matrix_axis.set_yticks(range(len(labels)), labels)
    matrix_axis.set_xlabel("Predicted sentiment")
    matrix_axis.set_ylabel("Actual sentiment")
    matrix_axis.set_title("Confusion matrix")
    midpoint = matrix.max() / 2 if matrix.size and matrix.max() else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            matrix_axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > midpoint else "black",
            )
    figure.colorbar(image, ax=matrix_axis, fraction=0.046, pad=0.04)
    figure.suptitle(f"Sentiment analysis report — {dataset_name} | accuracy: {accuracy:.1%}")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def train(args: argparse.Namespace) -> None:
    if args.dataset == "sentiment140":
        data = load_sentiment140(args.data or SENTIMENT140_DATA, args.max_samples)
        text_column, label_column = "text", "sentiment"
    elif args.dataset == "imdb":
        data = load_imdb(args.max_samples)
        text_column, label_column = "text", "sentiment"
    else:
        data = load_data(args.data or DEFAULT_DATA, args.text_column, args.label_column)
        text_column, label_column = args.text_column, args.label_column

    x_train, x_test, y_train, y_test = train_test_split(
        data[text_column],
        data[label_column],
        test_size=args.test_size,
        random_state=args.seed,
        stratify=data[label_column],
    )

    model = make_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    labels = sorted(data[label_column].unique())
    report_data = classification_report(
        y_test, predictions, labels=labels, zero_division=0, output_dict=True
    )
    report = classification_report(y_test, predictions, labels=labels, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)
    report_text = (
        f"Dataset: {args.dataset} ({args.data or 'built-in/default source'})\n"
        f"Rows used: {len(data)}\n"
        f"Training rows: {len(x_train)}\n"
        f"Test rows: {len(x_test)}\n"
        f"Accuracy: {accuracy:.3f}\n\n"
        "Classification report (precision, recall, F1-score):\n"
        f"{report}\n"
        f"Confusion matrix (actual rows, predicted columns; labels {', '.join(labels)}):\n"
        f"{matrix}\n"
    )
    args.report.write_text(report_text, encoding="utf-8")
    save_image_report(args.image_report, args.dataset, accuracy, report_data, matrix, labels)
    print(report_text)
    print(f"Saved model to: {args.model}")
    print(f"Saved evaluation report to: {args.report}")
    print(f"Saved image report to: {args.image_report}")


def predict(args: argparse.Namespace) -> None:
    if not args.model.exists():
        raise FileNotFoundError(
            f"Model not found at {args.model}. Run the train command first."
        )
    model = joblib.load(args.model)
    result = model.predict([args.text])[0]
    probabilities = model.predict_proba([args.text])[0]
    class_names = model.named_steps["classifier"].classes_
    confidence = probabilities[list(class_names).index(result)]
    print(f"Text: {args.text}")
    print(f"Predicted sentiment: {result}")
    print(f"Model confidence: {confidence:.1%}")
    print("Class probabilities:")
    print(json.dumps(dict(zip(class_names, probabilities.round(3).tolist())), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a sentiment model or predict the sentiment of a sentence."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Train and evaluate the model.")
    train_parser.add_argument(
        "--dataset",
        choices=("sentiment140", "imdb", "sample"),
        default="sentiment140",
        help="Dataset source (default: sentiment140).",
    )
    train_parser.add_argument(
        "--data", type=Path, default=None, help="Optional CSV path, used with sentiment140 or sample."
    )
    train_parser.add_argument("--text-column", default="text", help="CSV column containing text.")
    train_parser.add_argument(
        "--label-column", default="sentiment", help="CSV column containing labels."
    )
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Test split, from 0 to 1.")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable splits.")
    train_parser.add_argument(
        "--max-samples",
        type=int,
        default=100_000,
        help="Maximum rows to use (0 means all available rows).",
    )
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Where to save model.")
    train_parser.add_argument(
        "--report", type=Path, default=DEFAULT_REPORT, help="Where to save evaluation results."
    )
    train_parser.add_argument(
        "--image-report",
        type=Path,
        default=DEFAULT_IMAGE_REPORT,
        help="Where to save the evaluation PNG.",
    )
    train_parser.set_defaults(function=train)

    predict_parser = commands.add_parser("predict", help="Predict one text's sentiment.")
    predict_parser.add_argument("text", help="The sentence or review to classify.")
    predict_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Saved model file.")
    predict_parser.set_defaults(function=predict)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.function(arguments)

