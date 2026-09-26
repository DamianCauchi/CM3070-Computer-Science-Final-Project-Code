from pathlib import Path
from time import perf_counter

import cv2
import joblib
import numpy as np


# Prediction_Testing.py is stored inside "SVM algorithm".
SVM_FOLDER = Path(__file__).resolve().parent
PROJECT_FILES_FOLDER = SVM_FOLDER.parent

MODEL_PATH = (
    SVM_FOLDER
    / "Models"
    / "Three_player_svm.pkl"
)

UNKNOWN_GEI_FOLDER = (
    PROJECT_FILES_FOLDER
    / "Database"
    / "Unknown Recordings"
    / "Unknown GEIs"
)

IMAGE_SIZE = (64, 64)
LOW_CONFIDENCE_THRESHOLD = 0.65


def find_test_geis() -> list[Path]:
    """Find every unseen GEI in the Unknown GEIs folder."""

    if not UNKNOWN_GEI_FOLDER.is_dir():
        raise FileNotFoundError(
            f"Unknown GEI folder not found:\n{UNKNOWN_GEI_FOLDER}"
        )

    gei_paths = sorted(
        path
        for path in UNKNOWN_GEI_FOLDER.rglob("*.png")
        if path.is_file()
    )

    if not gei_paths:
        raise FileNotFoundError(
            f"No PNG GEIs were found in:\n{UNKNOWN_GEI_FOLDER}"
        )

    return gei_paths


def load_gei_feature(gei_path: Path) -> np.ndarray:
    """Convert one GEI into the same feature format used during training."""

    image = cv2.imread(str(gei_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read GEI: {gei_path}")

    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    feature_vector = image.astype(np.float32).flatten() / 255.0
    return feature_vector.reshape(1, -1)


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"SVM model not found:\n{MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "The saved SVM does not support probabilities. "
            "Train it using SVC(probability=True)."
        )

    gei_paths = find_test_geis()
    prediction_times = []

    print(f"Model: {MODEL_PATH}")
    print(f"Unknown GEI folder: {UNKNOWN_GEI_FOLDER}")
    print(f"GEIs found: {len(gei_paths)}")
    print(f"Model classes: {list(model.classes_)}")

    print("\nSVM UNSEEN-RECORDING PREDICTIONS")
    print("=" * 78)

    for gei_path in gei_paths:
        feature_vector = load_gei_feature(gei_path)

        start = perf_counter()
        predicted_player = model.predict(feature_vector)[0]
        probabilities = model.predict_proba(feature_vector)[0]
        prediction_seconds = perf_counter() - start

        prediction_times.append(prediction_seconds)

        ranked_results = sorted(
            zip(model.classes_, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )

        probability_by_class = dict(
            zip(model.classes_, probabilities)
        )

        predicted_probability = float(
            probability_by_class[predicted_player]
        )
        highest_probability = float(max(probabilities))

        print(f"GEI: {gei_path.name}")
        print(f"SVM prediction: {predicted_player}")
        print(
            "Probability assigned to prediction: "
            f"{predicted_probability:.2%}"
        )
        print("Class probabilities:")

        for player_name, probability in ranked_results:
            print(f"  {player_name}: {probability:.2%}")

        if highest_probability < LOW_CONFIDENCE_THRESHOLD:
            print(
                "Warning: the highest calibrated probability is below "
                "the heuristic 65% threshold."
            )

        print(
            f"Prediction time: {prediction_seconds:.6f} seconds"
        )
        print("-" * 78)

    mean_prediction_time = float(np.mean(prediction_times))

    print(
        "Mean prediction time per GEI: "
        f"{mean_prediction_time:.6f} seconds"
    )
    print(
        "\nImportant: these recordings were excluded from training. "
        "The SVM must still select one of its three enrolled players; "
        "the 65% threshold is only a warning and not a validated "
        "unknown-person detector."
    )


if __name__ == "__main__":
    main()