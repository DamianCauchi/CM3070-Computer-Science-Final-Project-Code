from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch

from lstm_common import load_checkpoint, load_sequence


# Store this script and lstm_common.py inside the "LSTM" folder.
LSTM_FOLDER = Path(__file__).resolve().parent
PROJECT_FILES_FOLDER = LSTM_FOLDER.parent
DEFAULT_MODEL_PATH = LSTM_FOLDER / "Models" / "gait_lstm.pt"
DEFAULT_SILHOUETTE_DIRECTORY = (
    PROJECT_FILES_FOLDER
    / "Database"
    / "Unknown Recordings"
    / "Unknown Silhouettes"
)

LOW_CONFIDENCE_THRESHOLD = 0.65
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict every unseen silhouette sequence using the saved LSTM."
        )
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--silhouette-directory",
        type=Path,
        default=DEFAULT_SILHOUETTE_DIRECTORY,
    )
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=LOW_CONFIDENCE_THRESHOLD,
    )
    return parser.parse_args()


def contains_images(directory: Path) -> bool:
    return any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in directory.iterdir()
    )


def find_sequence_folders(directory: Path) -> list[Path]:
    """Find one silhouette folder for each unseen source clip."""
    sequence_folders = [
        path
        for path in directory.iterdir()
        if path.is_dir() and contains_images(path)
    ]

    # This fallback also permits frames to be placed directly in the supplied
    # directory, although one folder per source clip is recommended.
    if not sequence_folders and contains_images(directory):
        sequence_folders = [directory]

    return sorted(sequence_folders, key=lambda path: path.name.lower())


def main() -> None:
    args = arguments()
    model_path = args.model.expanduser().resolve()
    silhouette_directory = args.silhouette_directory.expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(
            f"LSTM model not found: {model_path}\n"
            "Run train_lstm.py before running this predictor."
        )

    if not silhouette_directory.is_dir():
        raise FileNotFoundError(
            "Unknown silhouette directory not found: "
            f"{silhouette_directory}\n"
            "Run create_unknown_silhouettes.py from the Project Files "
            "folder first."
        )

    if not 0.0 <= args.low_confidence_threshold <= 1.0:
        raise ValueError("--low-confidence-threshold must be between 0 and 1")

    sequence_folders = find_sequence_folders(silhouette_directory)
    if not sequence_folders:
        raise FileNotFoundError(
            "No silhouette sequence folders containing image frames were "
            f"found in: {silhouette_directory}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, config, checkpoint_information = load_checkpoint(
        model_path,
        device,
    )
    model.eval()

    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Unknown silhouette directory: {silhouette_directory}")
    print(f"Players known by model: {', '.join(map(str, class_names))}")
    print(f"Sequences found: {len(sequence_folders)}")

    if checkpoint_information:
        best_epoch = checkpoint_information.get("best_epoch")
        if best_epoch is not None:
            print(f"Saved best epoch: {best_epoch}")

    print("\nLSTM UNSEEN-RECORDING PREDICTIONS")
    print("=" * 78)

    total_prediction_seconds = 0.0

    for clip_directory in sequence_folders:
        sequence = load_sequence(
            clip_directory,
            config.image_size,
            config.max_frames,
        )

        sequences = sequence.unsqueeze(0).to(device)
        lengths = torch.tensor(
            [sequence.shape[0]],
            dtype=torch.long,
            device=device,
        )

        start = perf_counter()
        with torch.inference_mode():
            logits = model(sequences, lengths)
            probabilities = torch.softmax(logits, dim=1)[0]
        prediction_seconds = perf_counter() - start
        total_prediction_seconds += prediction_seconds

        ranked_indices = torch.argsort(probabilities, descending=True).tolist()
        ranked_results = [
            (str(class_names[index]), float(probabilities[index].item()))
            for index in ranked_indices
        ]

        predicted_player, highest_probability = ranked_results[0]
        second_probability = (
            ranked_results[1][1] if len(ranked_results) > 1 else 0.0
        )
        top_two_margin = highest_probability - second_probability

        print(f"Sequence: {clip_directory.name}")
        print(f"LSTM prediction: {predicted_player}")
        print(f"Probability assigned to prediction: {highest_probability:.2%}")
        print("Class probabilities:")
        for player_name, probability in ranked_results:
            print(f"  {player_name}: {probability:.2%}")
        print(f"Top-two margin: {top_two_margin:.2%}")
        print(f"Frames used: {sequence.shape[0]}")

        if highest_probability < args.low_confidence_threshold:
            print(
                "Warning: the highest softmax probability is below the "
                f"heuristic {args.low_confidence_threshold:.0%} threshold."
            )

        print(f"Prediction time: {prediction_seconds:.6f} seconds")
        print("-" * 78)

    mean_prediction_seconds = (
        total_prediction_seconds / len(sequence_folders)
    )
    print(
        "Mean prediction time per sequence: "
        f"{mean_prediction_seconds:.6f} seconds"
    )
    print(
        "\nImportant: this is a closed-set model. It always selects one of "
        "its trained player classes. The confidence threshold is descriptive "
        "only and is not a validated unknown-person rejection rule."
    )


if __name__ == "__main__":
    main()
