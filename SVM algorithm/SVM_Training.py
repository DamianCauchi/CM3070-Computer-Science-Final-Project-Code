from pathlib import Path
import argparse

import cv2
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# This file should be stored inside:
# C:\Users\User\Desktop\Project Files\SVM algorithm\SVM_Training.py
SVM_FOLDER = Path(__file__).resolve().parent
PROJECT_FILES_FOLDER = SVM_FOLDER.parent
DEFAULT_DATABASE = PROJECT_FILES_FOLDER / "Database" / "Player Recordings"

MODEL_FOLDER = SVM_FOLDER / "Models"
MODEL_PATH = MODEL_FOLDER / "Three_player_svm.pkl"
IMAGE_SIZE = (64, 64)

# Folder name in the shared database -> label saved in the SVM model.
PLAYER_FOLDERS = {
    "Dani Alves BFC 22": "Dani Alves",
    "Iniesta BFC 8": "Iniesta",
    "Pique BFC 3": "Pique",
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train the SVM using GEIs from the shared player database."
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
    """Return supported GEI image files without including nested output folders."""
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(
        path
        for path in gei_folder.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def load_gei_features(gei_folder: Path, player_name: str):
    """Load one player's GEIs as normalised 4,096-value feature vectors."""
    if not gei_folder.is_dir():
        raise FileNotFoundError(
            f"GEI folder not found for {player_name}: {gei_folder}\n"
            "Run GEI.py from the Project Files folder first."
        )

    image_paths = find_gei_images(gei_folder)
    if not image_paths:
        raise ValueError(f"No GEI images found for {player_name}: {gei_folder}")

    features = []
    labels = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"WARNING: Skipping unreadable image: {image_path}")
            continue

        # INTER_AREA is suitable if an unexpected larger image must be reduced.
        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        feature_vector = image.astype(np.float32).reshape(-1) / 255.0

        features.append(feature_vector)
        labels.append(player_name)
        print(f"Loaded {player_name}: {image_path.name}")

    if not features:
        raise ValueError(f"No readable GEIs found for {player_name}: {gei_folder}")

    return features, labels


def main():
    args = parse_arguments()
    database_root = args.database.expanduser().resolve()

    if not database_root.is_dir():
        raise FileNotFoundError(
            f"Shared database not found: {database_root}\n"
            "Expected: Project Files\\Database\\Player Recordings"
        )

    print(f"Shared database: {database_root}")
    print(f"SVM model output: {MODEL_PATH}")

    all_features = []
    all_labels = []

    for folder_name, player_name in PLAYER_FOLDERS.items():
        gei_folder = database_root / folder_name / "GEI"
        player_features, player_labels = load_gei_features(
            gei_folder, player_name
        )
        all_features.extend(player_features)
        all_labels.extend(player_labels)

    X = np.asarray(all_features, dtype=np.float32)
    y = np.asarray(all_labels)

    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"\nTraining feature shape: {X.shape}")
    for label, count in zip(unique_labels, counts):
        print(f"{label}: {count} GEIs")

    if len(unique_labels) < 2:
        raise ValueError("At least two player classes are required for SVM training.")

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )

    print("\nTraining SVM...")
    model.fit(X, y)

    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nTraining complete.")
    print(f"Model classes: {model.classes_.tolist()}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()