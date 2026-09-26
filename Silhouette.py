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
        "Install dependencies with: "
        "python -m pip install ultralytics opencv-python"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_ROOT = PROJECT_ROOT / "Database" / "Player Recordings"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
IGNORED_FOLDERS = {"Silhouettes", "GEI"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the shared database silhouette sequences."
    )
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument("--player", action="append")
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_name(text: str) -> str:
    return re.sub(r'[<>:"/\\|?*]+', "_", text).strip(" .") or "clip"


def videos_in(player: Path) -> list[Path]:
    return sorted(
        path for path in player.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
        and not any(part in IGNORED_FOLDERS for part in path.parts)
    )


def choose_person(result, previous):
    if result.boxes is None or result.masks is None or len(result.boxes) == 0:
        return None, None

    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    masks = result.masks.data.cpu().numpy()
    height, width = result.orig_shape
    reference = previous or (width / 2, height / 2)
    diagonal = max(float(np.hypot(width, height)), 1.0)
    best, best_score = 0, -1.0

    for index, (box, confidence) in enumerate(zip(boxes, confidences)):
        x1, y1, x2, y2 = box
        centre = ((x1 + x2) / 2, (y1 + y2) / 2)
        distance = np.hypot(centre[0] - reference[0], centre[1] - reference[1])
        area = max(float((x2 - x1) * (y2 - y1)), 1.0)
        score = area * float(confidence) / (1 + distance / diagonal)
        if score > best_score:
            best, best_score = index, score

    x1, y1, x2, y2 = boxes[best]
    return masks[best], ((x1 + x2) / 2, (y1 + y2) / 2)


def normalise(mask: np.ndarray, size: int) -> np.ndarray | None:
    binary = (mask >= 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    body = max(contours, key=cv2.contourArea)
    if cv2.contourArea(body) < 10:
        return None

    clean = np.zeros_like(binary)
    cv2.drawContours(clean, [body], -1, 255, cv2.FILLED)
    x, y, width, height = cv2.boundingRect(body)
    crop = clean[y:y + height, x:x + width]
    margin = max(2, round(size * 0.08))
    available = size - 2 * margin
    scale = min(available / width, available / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    crop = cv2.resize(crop, (new_width, new_height),
                      interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size), dtype=np.uint8)
    left = (size - new_width) // 2
    top = max(0, size - margin - new_height)
    canvas[top:top + new_height, left:left + new_width] = crop
    return canvas


def process(video: Path, output: Path, model: YOLO, args) -> int:
    destination = output / safe_name(video.stem)
    destination.mkdir(parents=True, exist_ok=True)
    old_frames = list(destination.glob("frame_*.png"))
    if old_frames and not args.overwrite:
        print(f"  Skipped {destination.name}: already exists")
        return len(old_frames)
    if args.overwrite:
        for frame in old_frames:
            frame.unlink()

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        print(f"  ERROR: cannot open {video.name}")
        return 0

    read_number = saved = missed = 0
    previous = None
    while True:
        successful, frame = capture.read()
        if not successful:
            break
        current = read_number
        read_number += 1
        if current % args.frame_step:
            continue

        result = model.predict(frame, classes=[0], conf=args.confidence,
                               retina_masks=True, verbose=False)[0]
        mask, centre = choose_person(result, previous)
        if mask is None:
            missed += 1
            continue
        image = normalise(mask, args.size)
        if image is None:
            missed += 1
            continue

        cv2.imwrite(str(destination / f"frame_{saved:04d}.png"), image)
        previous = centre
        saved += 1

    capture.release()
    print(f"  {destination.name}: {saved} saved, {missed} missed")
    return saved


def main() -> None:
    args = arguments()
    root = args.database.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Database not found: {root}\n"
            "Create Database\\Player Recordings and put one folder per player inside."
        )
    if args.size < 16 or args.frame_step < 1:
        raise ValueError("--size must be at least 16 and --frame-step at least 1")

    model = YOLO(args.model)
    players = [root / name for name in args.player] if args.player else [
        path for path in root.iterdir() if path.is_dir()
    ]
    total = 0
    for player in sorted(players):
        videos = videos_in(player)
        print(f"\n{player.name}: {len(videos)} video(s)")
        output = player / "Silhouettes"
        output.mkdir(parents=True, exist_ok=True)
        for video in videos:
            print(f"  Processing {video.name}")
            total += process(video, output, model, args)
    print(f"\nFinished: {total} silhouette frames available.")


if __name__ == "__main__":
    main()
