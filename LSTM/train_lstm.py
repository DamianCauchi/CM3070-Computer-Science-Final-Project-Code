from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader

from lstm_common import (
    GaitLSTM,
    ModelConfig,
    SilhouetteSequenceDataset,
    class_weights,
    collate_sequences,
    discover_samples,
    parse_player_folders,
    save_checkpoint,
    set_seed,
)


# Store this file and lstm_common.py inside the LSTM algorithm folder.
LSTM_FOLDER = Path(__file__).resolve().parent
PROJECT_FILES_FOLDER = LSTM_FOLDER.parent
DEFAULT_DATABASE = PROJECT_FILES_FOLDER / "Database" / "Player Recordings"

# Shared database folder name -> class label stored in the LSTM checkpoint.
PLAYER_FOLDERS = {
    "Dani Alves BFC 22": "Dani Alves",
    "Iniesta BFC 8": "Iniesta",
    "Pique BFC 3": "Pique",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an LSTM on ordered football-player "
            "silhouette sequences."
        )
    )

    parser.add_argument(
        "--player",
        action="append",
        metavar="NAME=FOLDER",
        help=(
            "Optional override. Repeat for each player's "
            "Silhouettes folder. If omitted, the three players "
            "in the shared database are used."
        ),
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

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("Models/gait_lstm.pt"),
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):
    """
    Run one training or evaluation epoch.

    When optimizer is None, the function performs evaluation only.
    """

    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for sequences, lengths, labels, _ in loader:
        sequences = sequences.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(sequences, lengths)
            loss = criterion(logits, labels)

            if training:
                loss.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

        total_loss += (
            loss.item() * labels.size(0)
        )

        total_correct += (
            (logits.argmax(dim=1) == labels)
            .sum()
            .item()
        )

        total_items += labels.size(0)

    return (
        total_loss / total_items,
        total_correct / total_items,
    )


def main() -> None:
    args = parse_args()

    if args.max_frames < 2:
        raise ValueError(
            "--max-frames must be at least 2"
        )

    if args.image_size < 8:
        raise ValueError(
            "--image-size must be at least 8"
        )

    if args.hidden_size < 1:
        raise ValueError(
            "--hidden-size must be positive"
        )

    if args.epochs < 1:
        raise ValueError(
            "--epochs must be at least 1"
        )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be at least 1"
        )

    set_seed(args.seed)

    database_root = args.database.expanduser().resolve()

    if args.player:
        player_folders = parse_player_folders(args.player)
    else:
        if not database_root.is_dir():
            raise FileNotFoundError(
                f"Shared database not found: {database_root}\n"
                "Expected: Project Files\\Database\\Player Recordings"
            )

        player_folders = {}
        for folder_name, player_label in PLAYER_FOLDERS.items():
            silhouettes_folder = (
                database_root / folder_name / "Silhouettes"
            )
            if not silhouettes_folder.is_dir():
                raise FileNotFoundError(
                    f"Silhouettes folder not found for {player_label}: "
                    f"{silhouettes_folder}\n"
                    "Run silhouette.py from the Project Files folder first."
                )
            player_folders[player_label] = silhouettes_folder

    model_path = args.model.expanduser()
    if not model_path.is_absolute():
        model_path = LSTM_FOLDER / model_path
    model_path = model_path.resolve()

    print(f"Shared database: {database_root}")
    print(f"Model output: {model_path}")

    samples, class_names = discover_samples(
        player_folders
    )

    config = ModelConfig(
        image_width=args.image_size,
        image_height=args.image_size,
        max_frames=args.max_frames,
        hidden_size=args.hidden_size,
    )

    train_dataset = SilhouetteSequenceDataset(
        samples,
        config.image_size,
        config.max_frames,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_sequences,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(
        f"Training clips: {len(samples)}"
    )
    print("Validation clips: 0 (fixed epoch count selected by cross-validation)")

    model = GaitLSTM(
        config,
        len(class_names),
    ).to(device)

    weights = class_weights(
        (
            sample.label
            for sample in samples
        ),
        len(class_names),
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    start = perf_counter()

    final_train_loss = float("nan")
    final_train_accuracy = float("nan")

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        final_train_loss, final_train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {final_train_loss:.4f} "
            f"acc {final_train_accuracy:.3f}"
        )

    elapsed = perf_counter() - start

    save_checkpoint(
        model_path,
        model,
        class_names,
        config,
        extra={
            "training_mode": "all_samples_fixed_epochs",
            "epoch_selection": (
                "14 epochs selected as the median best epoch from "
                "three-fold cross-validation"
            ),
            "trained_epochs": args.epochs,
            "final_training_loss": final_train_loss,
            "final_training_accuracy": final_train_accuracy,
            "training_seconds": elapsed,
            "seed": args.seed,
            "training_clips": len(samples),
            "validation_clips": 0,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
        },
    )

    print(f"Completed epochs: {args.epochs}")
    print(
        f"Training time: {elapsed:.2f} seconds"
    )
    print(
        "Model saved to: "
        f"{model_path}"
    )


if __name__ == "__main__":
    main()