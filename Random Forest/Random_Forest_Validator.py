from pathlib import Path

import cv2
import joblib
import numpy as np


# Store this file inside:
# Project Files/Random Forest/Random_Forest_Validator.py
RANDOM_FOREST_FOLDER = Path(__file__).resolve().parent
PROJECT_FILES_FOLDER = RANDOM_FOREST_FOLDER.parent
DATABASE_ROOT = PROJECT_FILES_FOLDER / "Database" / "Player Recordings"
MODEL_PATH = (
    RANDOM_FOREST_FOLDER / "Models" / "random_forest_classifier.pkl"
)
IMAGE_SIZE = (64, 64)

# Shared database folder name -> class label stored in the model.
PLAYER_FOLDERS = {
    "Dani Alves BFC 22": "Dani Alves",
    "Iniesta BFC 8": "Iniesta",
    "Pique BFC 3": "Pique",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_gei_images(gei_folder: Path):
    """Find all GEI images stored directly inside one player's GEI folder."""
    if not gei_folder.is_dir():
        raise FileNotFoundError(
            f"GEI folder not found: {gei_folder}\n"
            "Run GEI.py from the Project Files folder first."
        )

    image_paths = sorted(
        path
        for path in gei_folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise ValueError(f"No GEI images found in: {gei_folder}")

    return image_paths


def load_feature_vector(gei_path: Path):
    """Apply the same 64 x 64 normalisation used during model training."""
    image = cv2.imread(str(gei_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA,
    )
    return image.astype(np.float32).reshape(-1) / 255.0


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Random Forest model not found: {MODEL_PATH}\n"
            "Run Random_Forest_Training.py before running this validator."
        )

    if not DATABASE_ROOT.is_dir():
        raise FileNotFoundError(
            f"Shared database not found: {DATABASE_ROOT}\n"
            "Expected: Project Files\\Database\\Player Recordings"
        )

    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise TypeError("The saved file is not a compatible Random Forest model.")

    features = []
    expected_labels = []
    gei_paths = []

    print(f"Model: {MODEL_PATH}")
    print(f"Shared database: {DATABASE_ROOT}")
    print(f"Model classes: {model.classes_.tolist()}\n")

    for folder_name, player_label in PLAYER_FOLDERS.items():
        gei_folder = DATABASE_ROOT / folder_name / "GEI"

        for gei_path in find_gei_images(gei_folder):
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

    print("RANDOM FOREST DATABASE VALIDATION RESULTS")
    print("=" * 78)

    for gei_path, expected_player, predicted_player, scores in zip(
        gei_paths,
        expected,
        predictions,
        probabilities,
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

    print("=" * 78)
    print(
        f"Overall database check: {correct_count}/{total_count} correct "
        f"({accuracy:.2%})"
    )

    print("\nPer-player database check:")
    for player_label in PLAYER_FOLDERS.values():
        total = class_total[player_label]
        correct = class_correct[player_label]
        player_accuracy = correct / total if total else 0.0
        print(
            f"  {player_label}: {correct}/{total} correct "
            f"({player_accuracy:.2%})"
        )

    print(
        "\nImportant: these GEIs came from the model's training database. "
        "This check confirms that the model associates the database GEIs with "
        "the correct identities. It does not measure performance on unseen clips."
    )


if __name__ == "__main__":
    main()
