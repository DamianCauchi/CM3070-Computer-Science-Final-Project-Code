from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = PROJECT_ROOT / "Recordings" / "Player Recording"
DEFAULT_PLAYERS = [
    "Dani Alves BFC 22",
    "Iniesta BFC 8",
    "Pique BFC 3",
]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one GEI from every LSTM silhouette sequence."
    )
    parser.add_argument("--player-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--player",
        action="append",
        help="Player folder name. Repeat to process multiple players.",
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def natural_key(path: Path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def create_gei(clip_dir: Path, output_path: Path, size: int) -> int:
    frames = sorted(
        (
            path
            for path in clip_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_key,
    )
    if not frames:
        return 0

    arrays = []
    for frame_path in frames:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            print(f"  WARNING: Could not read {frame_path}")
            continue
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_NEAREST)
        arrays.append(frame.astype(np.float32) / 255.0)

    if not arrays:
        return 0

    gei = np.mean(np.stack(arrays), axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.clip(gei * 255.0, 0, 255).astype(np.uint8))
    return len(arrays)


def main() -> None:
    args = parse_args()
    if args.image_size < 16:
        raise ValueError("--image-size must be at least 16")

    root = args.player_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Player Recording folder not found: {root}")

    players = args.player or DEFAULT_PLAYERS
    created = 0

    for player in players:
        player_dir = root / player
        silhouette_root = player_dir / "Silhouettes"
        gei_root = player_dir / "GEI"
        if not silhouette_root.is_dir():
            print(f"WARNING: Missing Silhouettes folder: {silhouette_root}")
            continue

        clip_dirs = sorted(
            (path for path in silhouette_root.iterdir() if path.is_dir()),
            key=natural_key,
        )
        print(f"\n{player}: {len(clip_dirs)} silhouette sequence(s)")

        for clip_dir in clip_dirs:
            output = gei_root / f"{clip_dir.name}_GEI.png"
            if output.exists() and not args.overwrite:
                print(f"  Skipped existing: {output.name}")
                continue
            count = create_gei(clip_dir, output, args.image_size)
            if count:
                created += 1
                print(f"  Created {output.name} from {count} frames")
            else:
                print(f"  WARNING: No readable frames in {clip_dir}")

    print(f"\nFinished. {created} GEI image(s) created.")


if __name__ == "__main__":
    main()