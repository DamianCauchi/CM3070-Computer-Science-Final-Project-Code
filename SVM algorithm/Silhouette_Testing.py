from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import cv2
import joblib
import numpy as np


# Store this script inside the "SVM algorithm" folder.
SVM_FOLDER = Path(__file__).resolve().parent
PROJECT_FILES_FOLDER = SVM_FOLDER.parent
DEFAULT_MODEL_PATH = SVM_FOLDER / "Models" / "Three_player_svm.pkl"
DEFAULT_GEI_DIRECTORY = (
    PROJECT_FILES_FOLDER
    / "Database"
    / "Unknown Recordings"
    / "Unknown GEIs"
)

IMAGE_SIZE = (64, 64)
LOW_CONFIDENCE_THRESHOLD = 0.65


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict every unseen GEI using the saved SVM model."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--gei-directory", type=Path, default=DEFAULT_GEI_DIRECTORY)
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=LOW_CONFIDENCE_THRESHOLD,
    )
    return parser.parse_args()


def load_gei_feature(gei_path: Path) -> np.ndarray:
    """Convert one GEI into the same 4,096-feature format as training."""
    image = cv2.imread(str(gei_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read unseen GEI: {gei_path}")

    image = cv2.resize(image, IMAGE_SIZE)
    feature_vector = image.astype(np.float32).flatten() / 255.0
    return feature_vector.reshape(1, -1)


def find_geis(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*.png")
        if path.is_file() and path.stem.lower().endswith("_gei")
    )


def main() -> None:
    args = arguments()
    model_path = args.model.expanduser().resolve()
    gei_directory = args.gei_directory.expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"SVM model not found: {model_path}")

    if not gei_directory.is_dir():
        raise FileNotFoundError(
            f"Unknown GEI directory not found: {gei_directory}\n"
            "Run create_unknown_geis.py from the Project Files folder first."
        )

    if not 0.0 <= args.low_confidence_threshold <= 1.0:
        raise ValueError("--low-confidence-threshold must be between 0 and 1")

    model = joblib.load(model_path)

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "The saved SVM does not support probabilities. "
            "Retrain it using SVC(probability=True)."
        )

    if not hasattr(model, "classes_"):
        raise TypeError("The saved SVM does not contain class labels.")

    gei_paths = find_geis(gei_directory)
    if not gei_paths:
        raise FileNotFoundError(
            f"No files ending in _GEI.png were found in: {gei_directory}"
        )

    print(f"Model: {model_path}")
    print(f"Unknown GEI directory: {gei_directory}")
    print(f"Players known by model: {', '.join(map(str, model.classes_))}")
    print(f"GEIs found: {len(gei_paths)}")
    print("\nSVM UNSEEN-RECORDING PREDICTIONS")
    print("=" * 78)

    total_prediction_seconds = 0.0

    for gei_path in gei_paths:
        feature_vector = load_gei_feature(gei_path)

        expected_features = getattr(model, "n_features_in_", None)
        if (
            expected_features is not None
            and feature_vector.shape[1] != expected_features
        ):
            raise ValueError(
                f"{gei_path.name} produced {feature_vector.shape[1]} features, "
                f"but the model expects {expected_features}."
            )

        start = perf_counter()
        predicted_player = str(model.predict(feature_vector)[0])
        probabilities = model.predict_proba(feature_vector)[0]
        prediction_seconds = perf_counter() - start
        total_prediction_seconds += prediction_seconds

        probability_by_class = {
            str(player): float(probability)
            for player, probability in zip(model.classes_, probabilities)
        }
        ranked_results = sorted(
            probability_by_class.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        probability_winner, highest_probability = ranked_results[0]
        predicted_probability = probability_by_class[predicted_player]

        print(f"GEI: {gei_path.name}")
        print(f"SVM prediction: {predicted_player}")
        print(f"Probability assigned to prediction: {predicted_probability:.2%}")
        print("Class probabilities:")
        for player_name, probability in ranked_results:
            print(f"  {player_name}: {probability:.2%}")

        if probability_winner != predicted_player:
            print(
                "Warning: SVM decision prediction and probability ranking "
                f"disagree. Highest probability class: {probability_winner}."
            )

        if highest_probability < args.low_confidence_threshold:
            print(
                "Warning: the highest calibrated probability is below the "
                f"heuristic {args.low_confidence_threshold:.0%} threshold."
            )

        print(f"Prediction time: {prediction_seconds:.6f} seconds")
        print("-" * 78)

    mean_prediction_seconds = total_prediction_seconds / len(gei_paths)
    print(
        f"Mean prediction time per GEI: {mean_prediction_seconds:.6f} seconds"
    )
    print(
        "\nImportant: this is a closed-set model. It always selects one of "
        "its trained player classes. The confidence threshold is descriptive "
        "only and is not a validated unknown-person rejection rule."
    )


if __name__ == "__main__":
    main()
