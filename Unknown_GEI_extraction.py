from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


# Store this script in the main "Project Files" directory.
PROJECT_ROOT = Path(__file__).resolve().parent
UNKNOWN_ROOT = PROJECT_ROOT / "Database" / "Unknown Recordings"
SILHOUETTES_ROOT = UNKNOWN_ROOT / "Unknown Silhouettes"
GEI_ROOT = UNKNOWN_ROOT / "Unknown GEIs"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one Gait Energy Image (GEI) for each unseen-recording "
            "silhouette sequence without modifying the training database."
        )
    )
    parser.add_argument("--silhouettes", type=Path, default=SILHOUETTES_ROOT)
    parser.add_argument("--output", type=Path, default=GEI_ROOT)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clip_directories(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and list(path.glob("frame_*.png"))
    )


def load_frames(clip_directory: Path, size: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []

    for frame_path in sorted(clip_directory.glob("frame_*.png")):
        image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read silhouette frame: {frame_path}")

        if image.shape != (size, size):
            image = cv2.resize(
                image,
                (size, size),
                interpolation=cv2.INTER_NEAREST,
            )

        # Preserve the binary foreground/background representation used by
        # the silhouette pipeline before calculating the temporal average.
        binary = (image >= 128).astype(np.float32)
        frames.append(binary)

    return frames


def create_gei(clip_directory: Path, output: Path, size: int) -> int:
    frames = load_frames(clip_directory, size)
    if not frames:
        raise ValueError(f"No silhouette frames found in: {clip_directory}")

    # GEI(x, y) = (1 / T) * sum of the T binary silhouettes at pixel (x, y).
    gei_normalised = np.mean(np.stack(frames, axis=0), axis=0)
    gei_image = np.rint(gei_normalised * 255.0).astype(np.uint8)

    if not cv2.imwrite(str(output), gei_image):
        raise OSError(f"Could not write GEI: {output}")

    return len(frames)


def main() -> None:
    args = arguments()
    silhouettes = args.silhouettes.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not silhouettes.is_dir():
        raise FileNotFoundError(
            f"Unknown silhouettes folder not found: {silhouettes}\n"
            "Run create_unknown_silhouettes.py first."
        )

    if args.size < 16:
        raise ValueError("--size must be at least 16")

    clips = clip_directories(silhouettes)
    if not clips:
        raise FileNotFoundError(
            f"No silhouette sequences were found in: {silhouettes}"
        )

    output.mkdir(parents=True, exist_ok=True)

    print(f"Silhouettes folder: {silhouettes}")
    print(f"GEI folder: {output}")
    print(f"Silhouette sequences found: {len(clips)}")

    created = 0
    skipped = 0

    for clip_directory in clips:
        output_path = output / f"{clip_directory.name}_GEI.png"

        if output_path.exists() and not args.overwrite:
            print(f"  Skipped {output_path.name}: already exists")
            skipped += 1
            continue

        frame_count = create_gei(clip_directory, output_path, args.size)
        print(f"  Created {output_path.name} from {frame_count} frames")
        created += 1

    print(
        f"\nFinished: {created} GEI(s) created, "
        f"{skipped} existing GEI(s) skipped."
    )


if __name__ == "__main__":
    main()