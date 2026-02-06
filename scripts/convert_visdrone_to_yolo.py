#!/usr/bin/env python3
"""
Convert VisDrone annotations to YOLO format.
VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
YOLO format: <class_id> <x_center> <y_center> <width> <height> (all normalized 0-1)


We use categories 1-10 only (map to YOLO classes 0-9).
"""

import argparse
from pathlib import Path

try:
    import cv2
    def _get_image_size(path):
        im = cv2.imread(str(path))
        return (im.shape[1], im.shape[0]) if im is not None else None
except ImportError:
    from PIL import Image
    def _get_image_size(path):
        try:
            im = Image.open(path)
            return im.size  # (width, height)
        except Exception:
            return None


def convert_visdrone_line(line: str, img_w: int, img_h: int):
    """Convert one VisDrone annotation line to YOLO format (class x_center y_center width height normalized)."""
    parts = line.strip().split(',')
    if len(parts) < 6:
        return None
    bbox_left = int(parts[0])
    bbox_top = int(parts[1])
    bbox_width = int(parts[2])
    bbox_height = int(parts[3])
    object_category = int(parts[5])
    # Skip ignored (0) and others (11)
    if object_category == 0 or object_category == 11:
        return None
    # Use categories 1-10 -> YOLO 0-9
    if not (1 <= object_category <= 10):
        return None
    class_id = object_category - 1
    x_center = (bbox_left + bbox_width / 2.0) / img_w
    y_center = (bbox_top + bbox_height / 2.0) / img_h
    width = bbox_width / img_w
    height = bbox_height / img_h
    # Clamp to [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    if width <= 0 or height <= 0:
        return None
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def convert_split(visdrone_root: Path, split_dir: str, images_subdir: str = "images", annotations_subdir: str = "annotations"):
    """
    Convert one VisDrone split (e.g. train_data/VisDrone2019-DET-train) to YOLO labels.
    Creates a 'labels' folder next to 'images' with same-name .txt files in YOLO format.
    """
    # Support both "train_data/VisDrone2019-DET-train" and "VisDrone2019-DET-train" style
    split_path = visdrone_root / split_dir
    if not split_path.exists():
        # Try without extra nesting
        for d in visdrone_root.iterdir():
            if d.is_dir() and split_dir in d.name:
                split_path = d
                break
        else:
            raise FileNotFoundError(f"Split path not found: {split_path}")
    images_dir = split_path / images_subdir
    annotations_dir = split_path / annotations_subdir
    labels_dir = split_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    converted = 0
    skipped_no_ann = 0
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in img_extensions:
            continue
        # Get image size
        size = _get_image_size(img_path)
        if size is None:
            continue
        img_w, img_h = size
        ann_path = annotations_dir / (img_path.stem + ".txt")
        label_path = labels_dir / (img_path.stem + ".txt")
        yolo_lines = []
        if ann_path.exists():
            with open(ann_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    yolo_line = convert_visdrone_line(line, img_w, img_h)
                    if yolo_line is not None:
                        yolo_lines.append(yolo_line)
        # Write YOLO label file (even if empty - some images have no objects)
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))
        if yolo_lines:
            converted += 1
        else:
            skipped_no_ann += 1
    return converted, skipped_no_ann


def main():
    parser = argparse.ArgumentParser(description="Convert VisDrone annotations to YOLO format")
    parser.add_argument(
        "--visdrone-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".." / "VisDrone",
        help="Path to VisDrone dataset root (containing train_data, val_data, etc.)",
    )
    args = parser.parse_args()
    visdrone_root = args.visdrone_dir.resolve()
    if not visdrone_root.exists():
        raise SystemExit(f"VisDrone root not found: {visdrone_root}")

    splits = [
        ("train_data/VisDrone2019-DET-train", "train"),
        ("val_data/VisDrone2019-DET-val", "val"),
    ]
    for split_dir, name in splits:
        full_path = visdrone_root / split_dir
        if not full_path.exists():
            # Try alternate structure (e.g. test_data has images/annotations at top)
            alt = visdrone_root / split_dir.split("/")[0]
            if alt.exists():
                for sub in alt.iterdir():
                    if sub.is_dir() and "images" in [x.name for x in sub.iterdir() if x.is_dir()]:
                        split_path = sub
                        break
                else:
                    print(f"Skipping {name}: {full_path} not found")
                    continue
            else:
                print(f"Skipping {name}: {full_path} not found")
                continue
        else:
            split_path = full_path
        try:
            n_conv, n_empty = convert_split(visdrone_root, split_dir)
            print(f"{name}: converted {n_conv} images with labels, {n_empty} empty")
        except Exception as e:
            print(f"{name}: error - {e}")


if __name__ == "__main__":
    main()
