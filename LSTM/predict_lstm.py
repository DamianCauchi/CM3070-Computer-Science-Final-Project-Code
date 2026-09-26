from __future__ import annotations

from pathlib import Path
from time import perf_counter

import torch

from lstm_common import (
    discover_samples,
    load_checkpoint,
    load_sequence,
)


# Store this file and lstm_common.py inside the LSTM algorithm folder.
LSTM_FOLDER = Path(__file__).resolve().parent
PROJECT_FILES_FOLDER = LSTM_FOLDER.parent
DATABASE_ROOT = PROJECT_FILES_FOLDER / "Database" / "Player Recordings"
MODEL_PATH = LSTM_FOLDER / "Models" / "gait_lstm.pt"

# Shared database folder name -> class label stored in the LSTM checkpoint.
PLAYER_FOLDERS = {
    "Dani Alves BFC 22": "Dani Alves",
    "Iniesta BFC 8": "Iniesta",
    "Pique BFC 3": "Pique",
}


def database_player_folders() -> dict[str, Path]:
    """Build the label-to-Silhouettes mapping used by the LSTM data loader."""
    if not DATABASE_ROOT.is_dir():
        raise FileNotFoundError(
            f"Shared database not found: {DATABASE_ROOT}\n"
            "Expected: Project Files\\Database\\Player Recordings"
        )

    player_folders: dict[str, Path] = {}

    for folder_name, player_label in PLAYER_FOLDERS.items():
        silhouettes_folder = DATABASE_ROOT / folder_name / "Silhouettes"

        if not silhouettes_folder.is_dir():
            raise FileNotFoundError(
                f"Silhouettes folder not found for {player_label}: "
                f"{silhouettes_folder}\n"
                "Run silhouette.py from the Project Files folder first."
            )

        player_folders[player_label] = silhouettes_folder

    return player_folders


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"LSTM model not found: {MODEL_PATH}\n"
            "Run train_lstm.py before running this database validator."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, class_names, config, checkpoint_information = load_checkpoint(
        MODEL_PATH,
        device,
    )

    player_folders = database_player_folders()
    samples, discovered_class_names = discover_samples(player_folders)

    missing_classes = set(discovered_class_names) - set(class_names)
    if missing_classes:
        raise ValueError(
            "These database players are missing from the trained model: "
            + ", ".join(sorted(missing_classes))
        )

    correct_count = 0
    total_prediction_seconds = 0.0
    class_correct = {player: 0 for player in discovered_class_names}
    class_total = {player: 0 for player in discovered_class_names}

    print(f"Device: {device}")
    print(f"Model: {MODEL_PATH}")
    print(f"Shared database: {DATABASE_ROOT}")
    print(f"Model classes: {class_names}")

    if checkpoint_information:
        best_epoch = checkpoint_information.get("best_epoch")
        if best_epoch is not None:
            print(f"Saved best epoch: {best_epoch}")

    print("\nLSTM DATABASE VALIDATION RESULTS")
    print("=" * 78)

    for sample in samples:
        sequence = load_sequence(
            sample.clip_dir,
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
        with torch.no_grad():
            logits = model(sequences, lengths)
            probabilities = torch.softmax(logits, dim=1)[0]
        prediction_seconds = perf_counter() - start
        total_prediction_seconds += prediction_seconds

        predicted_index = int(probabilities.argmax().item())
        predicted_player = class_names[predicted_index]
        expected_player = sample.player
        predicted_score = float(probabilities[predicted_index].item())

        is_correct = predicted_player == expected_player
        result = "CORRECT" if is_correct else "INCORRECT"

        class_total[expected_player] += 1
        if is_correct:
            correct_count += 1
            class_correct[expected_player] += 1

        print(
            f"{result:9} | Expected: {expected_player:11} "
            f"| Predicted: {predicted_player:11} "
            f"| Score: {predicted_score:6.2%} "
            f"| Frames used: {sequence.shape[0]:2d} "
            f"| {sample.clip_dir.name}"
        )

    total_count = len(samples)
    accuracy = correct_count / total_count
    mean_prediction_seconds = total_prediction_seconds / total_count

    print("=" * 78)
    print(
        f"Overall database check: {correct_count}/{total_count} correct "
        f"({accuracy:.2%})"
    )

    print("\nPer-player database check:")
    for player_label in discovered_class_names:
        total = class_total[player_label]
        correct = class_correct[player_label]
        player_accuracy = correct / total if total else 0.0
        print(
            f"  {player_label}: {correct}/{total} correct "
            f"({player_accuracy:.2%})"
        )

    print(
        f"\nMean prediction time per sequence: "
        f"{mean_prediction_seconds:.6f} seconds"
    )
    print(
        "\nImportant: these silhouette sequences came from the main database. "
        "This check confirms that the LSTM associates database sequences with "
        "the correct identities. It is not an independent unseen-clip test."
    )


if __name__ == "__main__":
    main()