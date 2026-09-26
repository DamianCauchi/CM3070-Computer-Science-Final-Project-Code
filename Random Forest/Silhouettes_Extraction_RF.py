from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        "Install the required packages first:\n"
        "python -m pip install ultralytics opencv-python"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = PROJECT_ROOT / "Recordings" / "Player Recording"
DEFAULT_PLAYERS = [
    "Dani Alves BFC 22",
    "Iniesta BFC 8",
    "Pique BFC 3",
]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
IGNORED_FOLDERS = {"Silhouettes", "GEI", "Models", "Results"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ordered LSTM silhouette sequences from videos."
    )
    parser.add_argument("--player-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--player",
        action="append",
        help="Player folder name. Repeat to process multiple players.",
    )
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_name(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]+', "_", name).strip(" .")
    return name or "clip"


def find_videos(player_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in player_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
        and not any(part in IGNORED_FOLDERS for part in path.parts)
    )


def select_person(result, previous_center):
    if result.boxes is None or result.masks is None or len(result.boxes) == 0:
        return None, None

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy()
    masks = result.masks.data.detach().cpu().numpy()
    height, width = result.orig_shape
    reference = previous_center or (width / 2.0, height / 2.0)
    diagonal = max(float(np.hypot(width, height)), 1.0)
    best_index = 0
    best_score = -1.0

    for index, (box, confidence) in enumerate(zip(boxes, confidences)):
        x1, y1, x2, y2 = box
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        distance = np.hypot(center[0] - reference[0], center[1] - reference[1])
        area = max(float((x2 - x1) * (y2 - y1)), 1.0)
        score = area * float(confidence) / (1.0 + distance / diagonal)
        if score > best_score:
            best_index = index
            best_score = score

    x1, y1, x2, y2 = boxes[best_index]
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return masks[best_index], center


def normalise_mask(mask: np.ndarray, size: int) -> np.ndarray | None:
    binary = (mask >= 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 10:
        return None

    clean = np.zeros_like(binary)
    cv2.drawContours(clean, [largest], -1, 255, cv2.FILLED)
    x, y, width, height = cv2.boundingRect(largest)
    cropped = clean[y : y + height, x : x + width]
    margin = max(2, int(size * 0.08))
    available = size - 2 * margin
    scale = min(available / width, available / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = cv2.resize(
        cropped,
        (new_width, new_height),
        interpolation=cv2.INTER_NEAREST,
    )
    output = np.zeros((size, size), dtype=np.uint8)
    left = (size - new_width) // 2
    top = max(0, size - margin - new_height)
    output[top : top + new_height, left : left + new_width] = resized
    return output


def process_video(video: Path, output_root: Path, model: YOLO, args) -> int:
    clip_dir = output_root / safe_name(video.stem)
    clip_dir.mkdir(parents=True, exist_ok=True)
    existing = list(clip_dir.glob("frame_*.png"))

    if existing and not args.overwrite:
        print(f"  Skipped existing: {clip_dir.name} ({len(existing)} frames)")
        return len(existing)
    if args.overwrite:
        for path in existing:
            path.unlink()

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        print(f"  ERROR: Could not open {video.name}")
        return 0

    frame_number = 0
    saved = 0
    missed = 0
    previous_center = None

    while True:
        success, frame = capture.read()
        if not success:
            break
        current = frame_number
        frame_number += 1
        if current % args.frame_step:
            continue

        result = model.predict(
            source=frame,
            classes=[0],
            conf=args.confidence,
            retina_masks=True,
            verbose=False,
        )[0]
        mask, center = select_person(result, previous_center)
        if mask is None:
            missed += 1
            continue
        silhouette = normalise_mask(mask, args.image_size)
        if silhouette is None:
            missed += 1
            continue

        cv2.imwrite(str(clip_dir / f"frame_{saved:04d}.png"), silhouette)
        saved += 1
        previous_center = center

    capture.release()
    print(f"  Created {clip_dir.name}: {saved} frames ({missed} missed)")
    return saved


def main() -> None:
    args = parse_args()
    if args.frame_step < 1:
        raise ValueError("--frame-step must be at least 1")
    if args.image_size < 16:
        raise ValueError("--image-size must be at least 16")

    root = args.player_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Player Recording folder not found: {root}")

    model = YOLO(args.model)
    players = args.player or DEFAULT_PLAYERS
    total = 0

    for player in players:
        player_dir = root / player
        if not player_dir.is_dir():
            print(f"WARNING: Missing player folder: {player_dir}")
            continue
        videos = find_videos(player_dir)
        print(f"\n{player}: {len(videos)} video(s)")
        output_root = player_dir / "Silhouettes"
        output_root.mkdir(parents=True, exist_ok=True)
        for video in videos:
            print(f"  Processing {video.name}")
            total += process_video(video, output_root, model, args)

    print(f"\nFinished. {total} silhouette frames are available.")


if __name__ == "__main__":
    main()
