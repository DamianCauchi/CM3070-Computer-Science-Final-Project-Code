from __future__ import annotations

import argparse
import copy
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)
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
    set_seed,
    write_json,
)
from train_lstm import run_epoch


# Store this script, lstm_common.py and train_lstm.py inside the LSTM folder.
LSTM_FOLDER = Path(__file__).resolve().parent
PROJECT_FILES_FOLDER = LSTM_FOLDER.parent
DEFAULT_DATABASE_ROOT = (
    PROJECT_FILES_FOLDER / "Database" / "Player Recordings"
)
DEFAULT_OUTPUT_PATH = (
    LSTM_FOLDER / "Results" / "lstm_cv_final_h32_wd001.json"
)
DEFAULT_CONFUSION_MATRIX_PATH = (
    LSTM_FOLDER
    / "Evaluation Results"
    / "lstm_cv_confusion_matrix.png"
)

CONFUSION_MATRIX_MINIMUM = 0
CONFUSION_MATRIX_MAXIMUM = 20
ANNOTATION_COLOUR_THRESHOLD = (
    CONFUSION_MATRIX_MINIMUM + CONFUSION_MATRIX_MAXIMUM
) / 2

PLAYER_FOLDERS = {
    "Dani Alves BFC 22": "Dani Alves",
    "Iniesta BFC 8": "Iniesta",
    "Pique BFC 3": "Pique",
}


def database_player_folders(database_root: Path) -> dict[str, Path]:
    """Return the shared database silhouette folders used by the LSTM."""
    if not database_root.is_dir():
        raise FileNotFoundError(
            f"Shared database not found: {database_root}\n"
            "Expected: Project Files\\Database\\Player Recordings"
        )

    player_folders: dict[str, Path] = {}

    for folder_name, player_label in PLAYER_FOLDERS.items():
        silhouettes_folder = database_root / folder_name / "Silhouettes"

        if not silhouettes_folder.is_dir():
            raise FileNotFoundError(
                f"Silhouettes folder not found for {player_label}: "
                f"{silhouettes_folder}\n"
                "Run silhouette.py from the Project Files folder first."
            )

        player_folders[player_label] = silhouettes_folder

    return player_folders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the LSTM using stratified "
            "cross-validation."
        )
    )

    parser.add_argument(
        "--player",
        action="append",
        metavar="NAME=FOLDER",
        help=(
            "Optional override. Repeat for every player. "
            "If omitted, Dani Alves BFC 22, Iniesta BFC 8 "
            "and Pique BFC 3 are used automatically."
        ),
    )

    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_ROOT,
        help=(
            "Shared Player Recordings directory. This is used when --player "
            "overrides are not supplied."
        ),
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Number of cross-validation folds.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=40,
        help="Maximum number of frames loaded from each clip.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=32,
        help="Width and height used when loading silhouettes.",
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="Number of features in the LSTM hidden state.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs for each fold.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early-stopping patience.",
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

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="JSON file used to save the evaluation results.",
    )

    parser.add_argument(
        "--confusion-matrix-output",
        type=Path,
        default=DEFAULT_CONFUSION_MATRIX_PATH,
        help=(
            "PNG file used to save the combined out-of-fold "
            "confusion matrix."
        ),
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")

    if args.max_frames < 2:
        raise ValueError("--max-frames must be at least 2.")

    if args.image_size < 8:
        raise ValueError("--image-size must be at least 8.")

    if args.hidden_size < 1:
        raise ValueError("--hidden-size must be positive.")

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    if args.patience < 1:
        raise ValueError("--patience must be at least 1.")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")


def predict_loader(
    model: GaitLSTM,
    loader: DataLoader,
    device: torch.device,
) -> tuple[
    list[int],
    list[int],
    list[str],
    float,
]:
    predictions: list[int] = []
    expected: list[int] = []
    paths: list[str] = []

    elapsed = 0.0
    model.eval()

    with torch.no_grad():
        for sequences, lengths, labels, batch_paths in loader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = perf_counter()

            logits = model(
                sequences,
                lengths,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed += perf_counter() - start

            predictions.extend(
                logits.argmax(dim=1).cpu().tolist()
            )

            expected.extend(labels.tolist())
            paths.extend(batch_paths)

    return (
        expected,
        predictions,
        paths,
        elapsed,
    )


def save_confusion_matrix(
    destination: Path,
    matrix: np.ndarray,
    class_names: list[str],
) -> None:
    """Save a report-ready cross-validation confusion matrix."""
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 6))

    image = axis.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
        vmin=CONFUSION_MATRIX_MINIMUM,
        vmax=CONFUSION_MATRIX_MAXIMUM,
    )

    colour_bar = figure.colorbar(image, ax=axis)
    colour_bar.set_label("Number of clips")

    display_names = [
        "Piqué" if name == "Pique" else name
        for name in class_names
    ]

    axis.set(
        title="LSTM: Three-fold Cross-validation",
        xlabel="Predicted player",
        ylabel="Actual player",
        xticks=np.arange(len(display_names)),
        yticks=np.arange(len(display_names)),
        xticklabels=display_names,
        yticklabels=display_names,
    )

    plt.setp(
        axis.get_xticklabels(),
        rotation=30,
        ha="right",
    )

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            text_colour = (
                "white"
                if value > ANNOTATION_COLOUR_THRESHOLD
                else "black"
            )

            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color=text_colour,
                fontsize=12,
                fontweight="bold",
            )

    figure.tight_layout()
    figure.savefig(
        destination,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    validate_arguments(args)
    set_seed(args.seed)

    if args.player:
        player_folders = parse_player_folders(
            args.player
        )
        database_root = None
    else:
        database_root = args.database.expanduser().resolve()
        player_folders = database_player_folders(
            database_root
        )

    samples, class_names = discover_samples(
        player_folders
    )

    labels = np.array(
        [sample.label for sample in samples],
        dtype=np.int64,
    )

    class_counts = np.bincount(
        labels,
        minlength=len(class_names),
    )

    print("\nDataset summary")

    if database_root is not None:
        print(f"  Shared database: {database_root}")

    for class_index, class_name in enumerate(class_names):
        print(
            f"  {class_name}: "
            f"{int(class_counts[class_index])} clips"
        )

    smallest_class = int(class_counts.min())

    if smallest_class < args.folds:
        raise ValueError(
            f"The smallest player class has only "
            f"{smallest_class} clips, but --folds is "
            f"{args.folds}. Each player needs at least one "
            f"clip in every fold. Reduce --folds or add clips."
        )

    indices = np.arange(len(samples))

    config = ModelConfig(
        image_width=args.image_size,
        image_height=args.image_size,
        max_frames=args.max_frames,
        hidden_size=args.hidden_size,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")
    print(f"Total clips: {len(samples)}")
    print(f"Cross-validation folds: {args.folds}")

    splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )

    all_expected: list[int] = []
    all_predictions: list[int] = []
    all_paths: list[str] = []
    all_folds: list[int] = []

    training_times: list[float] = []
    prediction_times: list[float] = []
    fold_results: list[dict] = []

    splits = splitter.split(
        indices,
        labels,
    )

    for fold, (
        outer_train_indices,
        test_indices,
    ) in enumerate(splits, start=1):
        print()
        print("=" * 60)
        print(f"Starting fold {fold} of {args.folds}")
        print("=" * 60)

        fold_seed = args.seed + fold
        set_seed(fold_seed)

        outer_train_labels = labels[
            outer_train_indices
        ]

        try:
            (
                train_indices,
                validation_indices,
            ) = train_test_split(
                outer_train_indices,
                test_size=0.20,
                random_state=fold_seed,
                shuffle=True,
                stratify=outer_train_labels,
            )
        except ValueError as exc:
            raise ValueError(
                "The training portion of a fold could not "
                "be divided into training and validation "
                "sets while retaining every player class. "
                "Add more clips or reduce --folds."
            ) from exc

        train_samples = [
            samples[index]
            for index in train_indices
        ]

        validation_samples = [
            samples[index]
            for index in validation_indices
        ]

        test_samples = [
            samples[index]
            for index in test_indices
        ]

        print(f"Training clips: {len(train_samples)}")
        print(
            "Validation clips: "
            f"{len(validation_samples)}"
        )
        print(f"Test clips: {len(test_samples)}")

        def make_loader(
            selected_samples,
            shuffle: bool = False,
        ) -> DataLoader:
            dataset = SilhouetteSequenceDataset(
                selected_samples,
                config.image_size,
                config.max_frames,
            )

            return DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=shuffle,
                collate_fn=collate_sequences,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
            )

        train_loader = make_loader(
            train_samples,
            shuffle=True,
        )

        validation_loader = make_loader(
            validation_samples
        )

        test_loader = make_loader(
            test_samples
        )

        model = GaitLSTM(
            config,
            len(class_names),
        ).to(device)

        weights = class_weights(
            (
                sample.label
                for sample in train_samples
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

        best_state = copy.deepcopy(
            model.state_dict()
        )

        best_validation_loss = float("inf")
        best_validation_accuracy = 0.0
        best_epoch = 0
        stale_epochs = 0

        training_start = perf_counter()

        for epoch in range(
            1,
            args.epochs + 1,
        ):
            (
                train_loss,
                train_accuracy,
            ) = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
            )

            (
                validation_loss,
                validation_accuracy,
            ) = run_epoch(
                model,
                validation_loader,
                criterion,
                device,
            )

            print(
                f"Fold {fold} | "
                f"Epoch {epoch:03d} | "
                f"train loss {train_loss:.4f}, "
                f"accuracy {train_accuracy:.3f} | "
                f"validation loss "
                f"{validation_loss:.4f}, "
                f"accuracy "
                f"{validation_accuracy:.3f}"
            )

            if (
                validation_loss
                < best_validation_loss - 1e-4
            ):
                best_validation_loss = (
                    validation_loss
                )

                best_validation_accuracy = (
                    validation_accuracy
                )

                best_epoch = epoch

                best_state = copy.deepcopy(
                    model.state_dict()
                )

                stale_epochs = 0
            else:
                stale_epochs += 1

                if stale_epochs >= args.patience:
                    print(
                        "Early stopping in fold "
                        f"{fold} after epoch {epoch}."
                    )
                    break

        training_elapsed = (
            perf_counter() - training_start
        )

        training_times.append(training_elapsed)

        model.load_state_dict(best_state)

        (
            expected,
            predictions,
            paths,
            prediction_elapsed,
        ) = predict_loader(
            model,
            test_loader,
            device,
        )

        all_expected.extend(expected)
        all_predictions.extend(predictions)
        all_paths.extend(paths)
        all_folds.extend([fold] * len(expected))

        mean_prediction_time = (
            prediction_elapsed / len(test_samples)
        )

        prediction_times.append(
            mean_prediction_time
        )

        fold_accuracy = accuracy_score(
            expected,
            predictions,
        )

        correct = sum(
            actual == predicted
            for actual, predicted in zip(
                expected,
                predictions,
            )
        )

        fold_results.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "best_validation_loss": float(
                    best_validation_loss
                ),
                "best_validation_accuracy": float(
                    best_validation_accuracy
                ),
                "test_clips": len(test_samples),
                "correct_predictions": int(correct),
                "accuracy": float(fold_accuracy),
                "training_seconds": float(
                    training_elapsed
                ),
                "prediction_seconds_per_clip": float(
                    mean_prediction_time
                ),
            }
        )

        print()
        print(f"Fold {fold} results")
        print(f"Best epoch: {best_epoch}")
        print(
            "Best validation loss: "
            f"{best_validation_loss:.4f}"
        )
        print(
            "Test accuracy: "
            f"{fold_accuracy * 100:.2f}% "
            f"({correct}/{len(expected)})"
        )
        print(
            "Training time: "
            f"{training_elapsed:.4f} seconds"
        )
        print(
            "Mean prediction time per clip: "
            f"{mean_prediction_time:.6f} seconds"
        )

    accuracy = accuracy_score(
        all_expected,
        all_predictions,
    )

    balanced_accuracy = balanced_accuracy_score(
        all_expected,
        all_predictions,
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        all_expected,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        all_expected,
        all_predictions,
        labels=list(range(len(class_names))),
    )

    report = classification_report(
        all_expected,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    total_correct = sum(
        actual == predicted
        for actual, predicted in zip(
            all_expected,
            all_predictions,
        )
    )

    print()
    print("=" * 60)
    print("Combined cross-validation results")
    print("=" * 60)

    print(
        f"Correct predictions: "
        f"{total_correct} of {len(all_expected)}"
    )

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print(
        "Balanced accuracy: "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print(
        "Macro precision: "
        f"{macro_precision * 100:.2f}%"
    )

    print(
        "Macro recall: "
        f"{macro_recall * 100:.2f}%"
    )

    print(
        "Macro F1-score: "
        f"{macro_f1 * 100:.2f}%"
    )

    print(
        "Mean training time per fold: "
        f"{np.mean(training_times):.4f} seconds"
    )

    print(
        "Mean prediction time per clip: "
        f"{np.mean(prediction_times):.6f} seconds"
    )

    print("\nConfusion matrix:")
    print(matrix)

    print("\nPer-player results:")

    for class_name in class_names:
        class_result = report[class_name]

        print(
            f"  {class_name}: "
            f"precision "
            f"{class_result['precision'] * 100:.2f}%, "
            f"recall "
            f"{class_result['recall'] * 100:.2f}%, "
            f"F1 "
            f"{class_result['f1-score'] * 100:.2f}%"
        )

    results = {
        "classes": class_names,
        "sample_count": len(samples),
        "class_counts": {
            class_name: int(class_counts[index])
            for index, class_name
            in enumerate(class_names)
        },
        "folds": args.folds,
        "correct_predictions": int(total_correct),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "mean_training_seconds_per_fold": float(
            np.mean(training_times)
        ),
        "mean_prediction_seconds_per_clip": float(
            np.mean(prediction_times)
        ),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "fold_results": fold_results,
        "out_of_fold_predictions": [
            {
                "fold": fold,
                "clip": path,
                "expected": class_names[
                    expected_label
                ],
                "predicted": class_names[
                    predicted_label
                ],
                "correct": (
                    expected_label
                    == predicted_label
                ),
            }
            for (
                fold,
                path,
                expected_label,
                predicted_label,
            ) in zip(
                all_folds,
                all_paths,
                all_expected,
                all_predictions,
            )
        ],
        "config": {
            "image_size": args.image_size,
            "max_frames": args.max_frames,
            "hidden_size": args.hidden_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
    }

    write_json(
        args.output,
        results,
    )

    save_confusion_matrix(
        args.confusion_matrix_output,
        matrix,
        class_names,
    )

    print()
    print(
        "Results saved to: "
        f"{args.output.resolve()}"
    )

    print(
        "Confusion matrix saved to: "
        f"{args.confusion_matrix_output.expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()