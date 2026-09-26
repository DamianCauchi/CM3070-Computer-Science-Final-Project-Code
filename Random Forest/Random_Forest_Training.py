from pathlib import Path
from time import perf_counter
import argparse

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# Store this file inside:
# Project Files/Random Forest/Random_Forest_Training.py
RANDOM_FOREST_FOLDER = Path(__file__).resolve().parent
PROJECT_FILES_FOLDER = RANDOM_FOREST_FOLDER.parent
DEFAULT_DATABASE = PROJECT_FILES_FOLDER / "Database" / "Player Recordings"

MODEL_FOLDER = RANDOM_FOREST_FOLDER / "Models"
MODEL_PATH = MODEL_FOLDER / "random_forest_classifier.pkl"
IMAGE_SIZE = (64, 64)

# Shared database folder name -> class label stored in the model.
PLAYER_FOLDERS = {
    "Dani Alves BFC 22": "Dani Alves",
    "Iniesta BFC 8": "Iniesta",
    "Pique BFC 3": "Pique",
}

EXPECTED_PLAYERS = set(PLAYER_FOLDERS.values())
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Train the Random Forest using GEIs from the shared player database."
        )
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help=(
            "Folder containing the player folders. The default is "
            "Project Files/Database/Player Recordings."
        ),
    )
    return parser.parse_args()


def find_gei_images(gei_folder: Path):
    """Return supported image files stored directly in a player's GEI folder."""
    if not gei_folder.is_dir():
        raise FileNotFoundError(f"GEI folder not found: {gei_folder}")

    image_paths = sorted(
        path
        for path in gei_folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise ValueError(f"No GEI images found in: {gei_folder}")

    return image_paths


def load_player_geis(gei_folder: Path, player_name: str):
    """Load one player's GEIs as normalised 4,096-value feature vectors."""
    image_paths = find_gei_images(gei_folder)
    features = []
    labels = []

    print(f"\nLoading {player_name}")
    print(f"Folder: {gei_folder}")

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"  WARNING: Skipping unreadable image: {image_path.name}")
            continue

        image = cv2.resize(
            image,
            IMAGE_SIZE,
            interpolation=cv2.INTER_AREA,
        )

        feature_vector = image.astype(np.float32).reshape(-1) / 255.0
        features.append(feature_vector)
        labels.append(player_name)
        print(f"  Loaded: {image_path.name}")

    if not features:
        raise ValueError(
            f"No readable GEIs were found for {player_name}: {gei_folder}"
        )

    return features, labels


def main():
    args = parse_arguments()
    database_root = args.database.expanduser().resolve()

    print("Random Forest football-player training")
    print("--------------------------------------")
    print(f"Shared database: {database_root}")
    print(f"Model output: {MODEL_PATH}")

    if not database_root.is_dir():
        raise FileNotFoundError(
            f"Shared database not found: {database_root}\n"
            "Expected: Project Files\\Database\\Player Recordings"
        )

    all_features = []
    all_labels = []

    for folder_name, player_name in PLAYER_FOLDERS.items():
        gei_folder = database_root / folder_name / "GEI"
        player_features, player_labels = load_player_geis(
            gei_folder,
            player_name,
        )
        all_features.extend(player_features)
        all_labels.extend(player_labels)

    X = np.asarray(all_features, dtype=np.float32)
    y = np.asarray(all_labels)

    classes, counts = np.unique(y, return_counts=True)
    loaded_players = set(classes)

    print("\nTraining dataset")
    print("----------------")
    for player_name, count in zip(classes, counts):
        print(f"{player_name}: {count} GEIs")

    print(f"Total GEIs: {X.shape[0]}")
    print(f"Features per GEI: {X.shape[1]}")

    missing_players = EXPECTED_PLAYERS - loaded_players
    if missing_players:
        raise ValueError(
            "Training stopped because these players were not loaded: "
            + ", ".join(sorted(missing_players))
        )

    model = RandomForestClassifier(
        n_estimators=300,
        criterion="gini",
        max_features="sqrt",
        class_weight="balanced",
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")
    start_time = perf_counter()
    model.fit(X, y)
    training_seconds = perf_counter() - start_time

    trained_players = set(model.classes_)
    missing_from_model = EXPECTED_PLAYERS - trained_players
    if missing_from_model:
        raise RuntimeError(
            "Training finished, but these players are missing from the model: "
            + ", ".join(sorted(missing_from_model))
        )

    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nTraining completed successfully")
    print("-------------------------------")
    print(f"Model classes: {model.classes_.tolist()}")
    print(f"Training samples: {X.shape[0]}")
    print(f"Training time: {training_seconds:.4f} seconds")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
