from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

try:
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence
    from torch.utils.data import Dataset
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required. Install the packages in "
        "requirements-lstm.txt first."
    ) from exc


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_PLAYER_FOLDERS = {
    "Dani Alves": PROJECT_ROOT
    / "Recordings"
    / "Player Recording"
    / "Dani Alves BFC 22"
    / "Silhouettes",
    "Iniesta": PROJECT_ROOT
    / "Recordings"
    / "Player Recording"
    / "Iniesta BFC 8"
    / "Silhouettes",
    "Pique": PROJECT_ROOT
    / "Recordings"
    / "Player Recording"
    / "Pique BFC 3"
    / "Silhouettes",
}


def set_seed(seed: int) -> None:
    """Make training as reproducible as possible."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def natural_key(path: Path) -> list[object]:
    """Sort frame_2 before frame_10."""

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def parse_player_folders(
    values: Sequence[str] | None,
) -> dict[str, Path]:
    """
    Parse command-line values such as:

    --player "Dani Alves=Silhouettes/Dani_Alves"
    """

    if not values:
        result = dict(DEFAULT_PLAYER_FOLDERS)

        missing = [
            path
            for path in result.values()
            if not path.is_dir()
        ]

        if missing:
            formatted = "\n".join(
                f"  - {path}" for path in missing
            )
            raise FileNotFoundError(
                "The default Silhouettes folders were not "
                f"found:\n{formatted}\n"
                "Keep the supplied folder structure or use "
                "--player NAME=FOLDER to provide paths."
            )

        return result

    result: dict[str, Path] = {}

    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --player value {value!r}; "
                "expected Player=folder"
            )

        player, folder = value.split("=", 1)
        player = player.strip()
        path = Path(folder.strip()).expanduser()

        if not player:
            raise ValueError(f"Missing player name in {value!r}")

        if player in result:
            raise ValueError(
                f"Player supplied more than once: {player}"
            )

        if not path.is_dir():
            raise FileNotFoundError(
                f"Silhouette folder does not exist: {path}"
            )

        result[player] = path

    if len(result) < 2:
        raise ValueError(
            "At least two --player entries are required."
        )

    return result


@dataclass(frozen=True)
class ClipSample:
    """Information about one independent video clip."""

    clip_dir: Path
    label: int
    player: str


def discover_samples(
    player_folders: dict[str, Path],
) -> tuple[list[ClipSample], list[str]]:
    """
    Treat every direct child directory of a player's folder as one
    independent silhouette sequence.
    """

    class_names = list(player_folders)
    samples: list[ClipSample] = []

    for label, player in enumerate(class_names):
        root = player_folders[player]

        clip_dirs = sorted(
            (path for path in root.iterdir() if path.is_dir()),
            key=natural_key,
        )

        for clip_dir in clip_dirs:
            frames = [
                path
                for path in clip_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            ]

            if frames:
                samples.append(
                    ClipSample(
                        clip_dir=clip_dir,
                        label=label,
                        player=player,
                    )
                )

        count = sum(
            sample.label == label for sample in samples
        )

        if count == 0:
            raise ValueError(
                f"No silhouette sequences were found for "
                f"{player} in:\n  {root}\n"
                "Each source clip must have its own folder "
                "inside Silhouettes, and that folder must "
                "contain ordered PNG/JPG frames."
            )

        print(f"{player}: {count} silhouette sequences")

    return samples, class_names


def frame_paths(clip_dir: Path) -> list[Path]:
    """Return the ordered silhouette images belonging to one clip."""

    paths = [
        path
        for path in clip_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(paths, key=natural_key)


def choose_uniformly(
    paths: Sequence[Path],
    max_frames: int,
) -> list[Path]:
    """
    Uniformly sample long sequences so that the complete movement
    remains represented.
    """

    if len(paths) <= max_frames:
        return list(paths)

    indices = np.linspace(
        0,
        len(paths) - 1,
        max_frames,
        dtype=np.int64,
    )

    return [paths[int(index)] for index in indices]


def load_sequence(
    clip_dir: Path,
    image_size: tuple[int, int],
    max_frames: int,
) -> torch.Tensor:
    """
    Load one ordered silhouette sequence.

    The returned tensor has the shape:
    sequence length × flattened image features.
    """

    paths = choose_uniformly(
        frame_paths(clip_dir),
        max_frames,
    )

    if not paths:
        raise ValueError(
            f"No silhouette frames found in: {clip_dir}"
        )

    frames: list[np.ndarray] = []

    for path in paths:
        with Image.open(path) as image:
            image = image.convert("L").resize(
                image_size,
                Image.Resampling.BILINEAR,
            )

            array = (
                np.asarray(image, dtype=np.float32) / 255.0
            )

        frames.append(array.reshape(-1))

    return torch.from_numpy(np.stack(frames))


class SilhouetteSequenceDataset(Dataset):
    """PyTorch dataset for ordered silhouette sequences."""

    def __init__(
        self,
        samples: Sequence[ClipSample],
        image_size: tuple[int, int],
        max_frames: int,
    ) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        sequence = load_sequence(
            sample.clip_dir,
            self.image_size,
            self.max_frames,
        )

        return (
            sequence,
            sequence.shape[0],
            sample.label,
            str(sample.clip_dir),
        )


def collate_sequences(batch):
    """
    Pad sequences within a batch.

    Padding is ignored by the LSTM because the original sequence
    lengths are supplied to pack_padded_sequence.
    """

    sequences, lengths, labels, paths = zip(*batch)

    max_length = max(lengths)
    feature_count = sequences[0].shape[1]

    padded = torch.zeros(
        len(batch),
        max_length,
        feature_count,
        dtype=torch.float32,
    )

    for index, sequence in enumerate(sequences):
        padded[index, : sequence.shape[0]] = sequence

    return (
        padded,
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        list(paths),
    )


@dataclass(frozen=True)
class ModelConfig:
    """Configuration saved with the trained model."""

    image_width: int = 32
    image_height: int = 32
    max_frames: int = 40
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.35
    bidirectional: bool = False

    @property
    def image_size(self) -> tuple[int, int]:
        return self.image_width, self.image_height


class GaitLSTM(nn.Module):
    """LSTM classifier for football-player silhouette sequences."""

    def __init__(
        self,
        config: ModelConfig,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.config = config

        input_size = (
            config.image_width * config.image_height
        )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=(
                config.dropout
                if config.num_layers > 1
                else 0.0
            ),
            bidirectional=config.bidirectional,
        )

        directions = 2 if config.bidirectional else 1

        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(
                config.hidden_size * directions,
                num_classes,
            ),
        )

    def forward(
        self,
        sequences: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        packed = pack_padded_sequence(
            sequences,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden, _) = self.lstm(packed)

        if self.config.bidirectional:
            representation = torch.cat(
                (hidden[-2], hidden[-1]),
                dim=1,
            )
        else:
            representation = hidden[-1]

        return self.classifier(representation)


def class_weights(
    labels: Iterable[int],
    num_classes: int,
) -> torch.Tensor:
    """Calculate balanced weights for CrossEntropyLoss."""

    counts = np.bincount(
        np.fromiter(labels, dtype=np.int64),
        minlength=num_classes,
    )

    if np.any(counts == 0):
        raise ValueError(
            "A training split is missing a class: "
            f"counts={counts.tolist()}"
        )

    weights = counts.sum() / (num_classes * counts)

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def save_checkpoint(
    path: Path,
    model: GaitLSTM,
    class_names: Sequence[str],
    config: ModelConfig,
    extra: dict | None = None,
) -> None:
    """Save the model, class labels and preprocessing settings."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": list(class_names),
            "config": asdict(config),
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(
    path: Path,
    device: torch.device,
):
    """Load a model and its saved preprocessing configuration."""

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    config = ModelConfig(**checkpoint["config"])
    class_names = checkpoint["class_names"]

    model = GaitLSTM(
        config,
        len(class_names),
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    return (
        model,
        class_names,
        config,
        checkpoint.get("extra", {}),
    )


def write_json(path: Path, data: dict) -> None:
    """Save evaluation results as formatted JSON."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )
