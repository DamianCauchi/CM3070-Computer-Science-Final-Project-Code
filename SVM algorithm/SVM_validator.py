from pathlib import Path

import cv2
import joblib
import numpy as np


# Store this file inside:
# Project Files/SVM algorithm/Prediction.py
SVM_FOLDER = Path(__file__).resolve().parent
PROJECT_FILES_FOLDER = SVM_FOLDER.parent
DATABASE_ROOT = PROJECT_FILES_FOLDER / "Database" / "Player Recordings"
MODEL_PATH = SVM_FOLDER / "Models" / "Three_player_svm.pkl"
IMAGE_SIZE = (64, 64)

# Shared database folder name -> class label stored in the SVM model.
PLAYER_FOLDERS = {
    "Dani Alves BFC 22": "Dani Alves",
    "Iniesta BFC 8": "Iniesta",
    "Pique BFC 3": "Pique",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_gei_images(gei_folder: Path):
    """Find all GEI images stored directly inside one player's GEI folder."""
    return sorted(
        path
        for path in gei_folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_feature_vector(gei_path: Path):
    """Apply the same 64 x 64 normalisation used by SVM_Training.py."""
    image = cv2.imread(str(gei_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    feature_vector = image.astype(np.float32).reshape(-1) / 255.0
    return feature_vector


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"SVM model not found: {MODEL_PATH}\n"
            "Run SVM_Training.py before running this database validator."
        )

    if not DATABASE_ROOT.is_dir():
        raise FileNotFoundError(
            f"Shared database not found: {DATABASE_ROOT}\n"
            "Expected: Project Files\\Database\\Player Recordings"
        )

    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise TypeError("The saved file is not a compatible SVM model.")

    features = []
    expected_labels = []
    gei_paths = []

    print(f"Model: {MODEL_PATH}")
    print(f"Shared database: {DATABASE_ROOT}\n")

    for folder_name, player_label in PLAYER_FOLDERS.items():
        gei_folder = DATABASE_ROOT / folder_name / "GEI"

        if not gei_folder.is_dir():
            raise FileNotFoundError(
                f"GEI folder not found for {player_label}: {gei_folder}\n"
                "Run GEI.py from the Project Files folder first."
            )

        image_paths = find_gei_images(gei_folder)
        if not image_paths:
            raise ValueError(f"No GEI images found for {player_label}: {gei_folder}")

        for gei_path in image_paths:
            feature_vector = load_feature_vector(gei_path)
            if feature_vector is None:
                print(f"WARNING: Skipping unreadable image: {gei_path}")
                continue

            features.append(feature_vector)
            expected_labels.append(player_label)
            gei_paths.append(gei_path)

    if not features:
        raise ValueError("No readable GEIs were found in the shared database.")

    X = np.asarray(features, dtype=np.float32)
    expected = np.asarray(expected_labels)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    correct_count = 0
    class_correct = {label: 0 for label in PLAYER_FOLDERS.values()}
    class_total = {label: 0 for label in PLAYER_FOLDERS.values()}

    print("SVM DATABASE VALIDATION RESULTS")
    print("=" * 72)

    for gei_path, expected_player, predicted_player, scores in zip(
        gei_paths, expected, predictions, probabilities
    ):
        is_correct = predicted_player == expected_player
        result = "CORRECT" if is_correct else "INCORRECT"

        class_total[expected_player] += 1
        if is_correct:
            correct_count += 1
            class_correct[expected_player] += 1

        predicted_index = list(model.classes_).index(predicted_player)
        predicted_score = scores[predicted_index]

        print(
            f"{result:9} | Expected: {expected_player:11} "
            f"| Predicted: {predicted_player:11} "
            f"| Score: {predicted_score:6.2%} "
            f"| {gei_path.name}"
        )

    total_count = len(expected)
    accuracy = correct_count / total_count

    print("=" * 72)
    print(
        f"Overall database check: {correct_count}/{total_count} correct "
        f"({accuracy:.2%})"
    )

    print("\nPer-player database check:")
    for player_label in PLAYER_FOLDERS.values():
        total = class_total[player_label]
        correct = class_correct[player_label]
        player_accuracy = correct / total if total else 0.0
        print(f"  {player_label}: {correct}/{total} correct ({player_accuracy:.2%})")

    print(
        "\nImportant: these GEIs came from the model's main database. "
        "This validates the database, paths, labels and model fit, but it does "
        "not measure generalisation to unseen clips."
    )


if __name__ == "__main__":
    main()
