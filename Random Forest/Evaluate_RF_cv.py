from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold


# Store this script inside the "Random Forest" folder.
RANDOM_FOREST_FOLDER = Path(__file__).resolve().parent
PROJECT_FILES_FOLDER = RANDOM_FOREST_FOLDER.parent
DEFAULT_DATABASE_ROOT = (
    PROJECT_FILES_FOLDER / "Database" / "Player Recordings"
)
DEFAULT_RESULTS_DIRECTORY = RANDOM_FOREST_FOLDER / "Evaluation Results"

PLAYER_FOLDERS = {
    "Dani Alves BFC 22": "Dani Alves",
    "Iniesta BFC 8": "Iniesta",
    "Pique BFC 3": "Pique",
}

IMAGE_SIZE = (64, 64)
N_SPLITS = 3
RANDOM_STATE = 42


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run three-fold cross-validation for Random Forest."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_ROOT,
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=DEFAULT_RESULTS_DIRECTORY,
    )
    return parser.parse_args()


def find_geis(gei_folder: Path) -> list[Path]:
    return sorted(
        path
        for path in gei_folder.rglob("*.png")
        if path.is_file() and path.stem.lower().endswith("_gei")
    )


def discover_samples(database_root: Path) -> tuple[list[Path], list[str]]:
    paths: list[Path] = []
    labels: list[str] = []

    for folder_name, player_label in PLAYER_FOLDERS.items():
        gei_folder = database_root / folder_name / "GEI"
        if not gei_folder.is_dir():
            raise FileNotFoundError(
                f"GEI folder not found for {player_label}: {gei_folder}"
            )

        player_geis = find_geis(gei_folder)
        if not player_geis:
            raise FileNotFoundError(
                f"No files ending in _GEI.png found in: {gei_folder}"
            )

        print(f"{player_label}: {len(player_geis)} GEIs")
        paths.extend(player_geis)
        labels.extend([player_label] * len(player_geis))

    return paths, labels


def load_features(paths: list[Path]) -> np.ndarray:
    features: list[np.ndarray] = []

    for gei_path in paths:
        image = cv2.imread(str(gei_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read GEI: {gei_path}")
        image = cv2.resize(image, IMAGE_SIZE)
        features.append(image.astype(np.float32).flatten() / 255.0)

    return np.asarray(features, dtype=np.float32)


def create_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        criterion="gini",
        max_features="sqrt",
        class_weight="balanced",
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def save_predictions(
    destination: Path,
    paths: list[Path],
    actual: np.ndarray,
    predicted: np.ndarray,
    folds: np.ndarray,
) -> None:
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["fold", "clip", "actual", "predicted", "correct"])
        for path, true_label, prediction, fold in zip(
            paths,
            actual,
            predicted,
            folds,
        ):
            writer.writerow(
                [
                    int(fold),
                    path.name,
                    true_label,
                    prediction,
                    true_label == prediction,
                ]
            )


def save_confusion_matrix(
    destination: Path,
    matrix: np.ndarray,
    class_names: list[str],
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Greens")
    figure.colorbar(image, ax=axis)
    axis.set(
        title="Random Forest: Three-fold Cross-validation",
        xlabel="Predicted player",
        ylabel="Actual player",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = arguments()
    database_root = args.database.expanduser().resolve()
    results_directory = args.results_directory.expanduser().resolve()

    if not database_root.is_dir():
        raise FileNotFoundError(f"Database not found: {database_root}")

    paths, labels = discover_samples(database_root)
    features = load_features(paths)
    actual = np.asarray(labels)
    class_names = list(PLAYER_FOLDERS.values())

    print(f"Total: {len(paths)} GEIs")
    print(f"Feature dimensions: {features.shape[1]}")
    print(f"Folds: {N_SPLITS}")
    print(f"Random state: {RANDOM_STATE}")
    print("\nRANDOM FOREST THREE-FOLD CROSS-VALIDATION")
    print("=" * 78)

    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    predictions = np.empty(len(actual), dtype=object)
    fold_numbers = np.zeros(len(actual), dtype=np.int64)
    training_times: list[float] = []
    prediction_times: list[float] = []

    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(features, actual),
        start=1,
    ):
        model = create_model()

        start = perf_counter()
        model.fit(features[train_indices], actual[train_indices])
        training_seconds = perf_counter() - start

        start = perf_counter()
        fold_predictions = model.predict(features[test_indices])
        prediction_seconds = perf_counter() - start

        predictions[test_indices] = fold_predictions
        fold_numbers[test_indices] = fold
        training_times.append(training_seconds)
        prediction_times.append(prediction_seconds / len(test_indices))

        fold_accuracy = accuracy_score(actual[test_indices], fold_predictions)
        print(
            f"Fold {fold}: train={len(train_indices)}, test={len(test_indices)}, "
            f"correct={int(np.sum(actual[test_indices] == fold_predictions))}/"
            f"{len(test_indices)}, accuracy={fold_accuracy:.2%}, "
            f"training={training_seconds:.6f}s, "
            f"prediction/clip={prediction_times[-1]:.6f}s"
        )

    accuracy = accuracy_score(actual, predictions)
    balanced_accuracy = balanced_accuracy_score(actual, predictions)
    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            actual,
            predictions,
            labels=class_names,
            average="macro",
            zero_division=0,
        )
    )
    matrix = confusion_matrix(actual, predictions, labels=class_names)
    correct = int(np.sum(actual == predictions))

    print("=" * 78)
    print("COMBINED OUT-OF-FOLD RESULTS")
    print(f"Correct: {correct}/{len(actual)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Balanced accuracy: {balanced_accuracy:.2%}")
    print(f"Macro precision: {macro_precision:.2%}")
    print(f"Macro recall: {macro_recall:.2%}")
    print(f"Macro F1: {macro_f1:.2%}")
    print(f"Mean training time: {np.mean(training_times):.6f} seconds")
    print(
        "Mean prediction time per clip: "
        f"{np.mean(prediction_times):.6f} seconds"
    )
    print("\nPER-PLAYER RESULTS")
    print(
        classification_report(
            actual,
            predictions,
            labels=class_names,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )
    print("CONFUSION MATRIX")
    print(f"Rows/columns: {class_names}")
    print(matrix)

    results_directory.mkdir(parents=True, exist_ok=True)
    predictions_path = results_directory / "random_forest_cv_predictions.csv"
    matrix_path = (
        results_directory / "random_forest_cv_confusion_matrix.png"
    )
    save_predictions(
        predictions_path,
        paths,
        actual,
        predictions,
        fold_numbers,
    )
    save_confusion_matrix(matrix_path, matrix, class_names)

    print(f"\nPredictions saved to: {predictions_path}")
    print(f"Confusion matrix saved to: {matrix_path}")
    print(
        "Important: these are out-of-fold predictions. Each GEI was tested "
        "by a model that was not trained using that GEI."
    )


if __name__ == "__main__":
    main()
