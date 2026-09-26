from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_ROOT = PROJECT_ROOT / "Database" / "Player Recordings"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create shared 64x64 GEIs from database silhouettes."
    )
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument("--player", action="append")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def create_gei(sequence: Path, output: Path, size: int) -> int:
    paths = sorted(
        (path for path in sequence.iterdir()
         if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=natural_key,
    )
    frames = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"  WARNING: unreadable image: {path.name}")
            continue
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)
        frames.append(image.astype(np.float32) / 255.0)
    if not frames:
        return 0

    gei = np.mean(np.stack(frames), axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.clip(gei * 255, 0, 255).astype(np.uint8))
    return len(frames)


def main() -> None:
    args = arguments()
    root = args.database.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Database not found: {root}")
    if args.size < 16:
        raise ValueError("--size must be at least 16")

    players = [root / name for name in args.player] if args.player else [
        path for path in root.iterdir() if path.is_dir()
    ]
    created = 0
    for player in sorted(players):
        silhouettes = player / "Silhouettes"
        if not silhouettes.is_dir():
            print(f"WARNING: no Silhouettes folder for {player.name}")
            continue
        sequences = sorted(
            (path for path in silhouettes.iterdir() if path.is_dir()),
            key=natural_key,
        )
        print(f"\n{player.name}: {len(sequences)} sequence(s)")
        for sequence in sequences:
            output = player / "GEI" / f"{sequence.name}_GEI.png"
            if output.exists() and not args.overwrite:
                print(f"  Skipped {output.name}: already exists")
                continue
            count = create_gei(sequence, output, args.size)
            if count:
                created += 1
                print(f"  Created {output.name} from {count} frames")
            else:
                print(f"  WARNING: no readable frames in {sequence.name}")
    print(f"\nFinished: {created} GEI image(s) created.")


if __name__ == "__main__":
    main()
