#!/usr/bin/env python3
"""
Convert VisDrone-MOT dataset to YOLO detection format and merge with DET dataset.

MOT annotation format (10 fields per line):
  <frame_id>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

DET annotation format (8 fields per line):
  <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

This script:
1. Reads MOT sequences (each sequence = folder of JPEGs + one annotation file)
2. Samples every Nth frame to reduce redundancy from consecutive video frames
3. Converts sampled frame annotations to YOLO format
4. Copies sampled images + labels into a combined output directory

Usage:
  # Convert MOT only (sampled every 10 frames)
  python scripts/convert_visdrone_mot_to_yolo.py \
    --mot-dir ../VisDrone-MOT/VisDrone2019-MOT-train \
    --output-dir ../VisDrone/combined_train \
    --sample-rate 10

  # Also merge with existing DET dataset
  python scripts/convert_visdrone_mot_to_yolo.py \
    --mot-dir ../VisDrone-MOT/VisDrone2019-MOT-train \
    --det-dir ../VisDrone/train_data/VisDrone2019-DET-train \
    --output-dir ../VisDrone/combined_train \
    --sample-rate 10
"""

import argparse
import shutil
from collections import defaultdict
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


def convert_mot_bbox(bbox_left, bbox_top, bbox_width, bbox_height, object_category, img_w, img_h):
    """Convert one MOT bounding box to YOLO format string."""
    # Skip ignored (0) and others (11)
    if object_category == 0 or object_category == 11:
        return None
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


def parse_mot_annotation(ann_path):
    """Parse MOT annotation file into dict: frame_id -> list of (bbox_left, bbox_top, bbox_width, bbox_height, object_category)."""
    frames = defaultdict(list)
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 8:
                continue
            frame_id = int(parts[0])
            # target_id = int(parts[1])  # not needed for detection
            bbox_left = int(parts[2])
            bbox_top = int(parts[3])
            bbox_width = int(parts[4])
            bbox_height = int(parts[5])
            # parts[6] = score
            object_category = int(parts[7])
            frames[frame_id].append((bbox_left, bbox_top, bbox_width, bbox_height, object_category))
    return frames


def convert_mot_sequence(seq_dir, ann_path, output_images_dir, output_labels_dir, sample_rate, prefix):
    """Convert one MOT sequence to YOLO format, sampling every Nth frame.

    Args:
        seq_dir: Path to sequence folder (contains 0000001.jpg, 0000002.jpg, ...)
        ann_path: Path to annotation file for this sequence
        output_images_dir: Where to copy sampled images
        output_labels_dir: Where to write YOLO label files
        sample_rate: Sample every Nth frame (1 = all frames, 10 = every 10th)
        prefix: Prefix for output filenames to avoid collisions between sequences

    Returns:
        (n_converted, n_skipped) tuple
    """
    # Parse all annotations for this sequence
    frame_annotations = parse_mot_annotation(ann_path)

    # Find all image files in sequence
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    img_files = sorted([f for f in seq_dir.iterdir() if f.suffix.lower() in img_extensions])

    if not img_files:
        return 0, 0

    # Get image size from first frame
    img_size = _get_image_size(img_files[0])
    if img_size is None:
        return 0, 0
    img_w, img_h = img_size

    converted = 0
    skipped = 0

    for img_path in img_files:
        # Extract frame number from filename (e.g., "0000001" -> 1)
        try:
            frame_id = int(img_path.stem)
        except ValueError:
            continue

        # Sample every Nth frame
        if frame_id % sample_rate != 1 and sample_rate > 1:
            skipped += 1
            continue

        # Get annotations for this frame
        annotations = frame_annotations.get(frame_id, [])

        # Convert to YOLO format
        yolo_lines = []
        for bbox_left, bbox_top, bbox_width, bbox_height, obj_cat in annotations:
            yolo_line = convert_mot_bbox(bbox_left, bbox_top, bbox_width, bbox_height, obj_cat, img_w, img_h)
            if yolo_line is not None:
                yolo_lines.append(yolo_line)

        # Output filename: prefix_framenum to avoid collisions
        out_name = f"{prefix}_{frame_id:07d}"

        # Copy image
        out_img = output_images_dir / f"{out_name}{img_path.suffix}"
        shutil.copy2(img_path, out_img)

        # Write label
        out_label = output_labels_dir / f"{out_name}.txt"
        with open(out_label, 'w') as f:
            f.write('\n'.join(yolo_lines))

        converted += 1

    return converted, skipped


def copy_det_dataset(det_dir, output_images_dir, output_labels_dir):
    """Copy existing DET dataset (already converted to YOLO) into the combined output.

    Expects det_dir to contain images/ and labels/ subdirectories.
    """
    det_images = det_dir / 'images'
    det_labels = det_dir / 'labels'

    if not det_images.exists():
        raise FileNotFoundError(f"DET images dir not found: {det_images}")
    if not det_labels.exists():
        raise FileNotFoundError(f"DET labels dir not found: {det_labels}. Run convert_visdrone_to_yolo.py first.")

    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    copied = 0

    for img_path in sorted(det_images.iterdir()):
        if img_path.suffix.lower() not in img_extensions:
            continue

        # Copy image with "det_" prefix to avoid collisions
        out_name = f"det_{img_path.stem}"
        shutil.copy2(img_path, output_images_dir / f"{out_name}{img_path.suffix}")

        # Copy label if exists
        label_path = det_labels / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, output_labels_dir / f"{out_name}.txt")
        else:
            # Empty label file
            (output_labels_dir / f"{out_name}.txt").write_text('')

        copied += 1

    return copied


def main():
    parser = argparse.ArgumentParser(description="Convert VisDrone MOT to YOLO and merge with DET")
    parser.add_argument('--mot-dir', type=Path, required=True,
                        help='Path to MOT train dir (containing sequences/ and annotations/)')
    parser.add_argument('--det-dir', type=Path, default=None,
                        help='Path to DET train dir (containing images/ and labels/). Optional.')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory (will create images/ and labels/ inside)')
    parser.add_argument('--sample-rate', type=int, default=10,
                        help='Sample every Nth frame from MOT sequences (default: 10)')
    args = parser.parse_args()

    mot_dir = args.mot_dir.resolve()
    output_dir = args.output_dir.resolve()
    sample_rate = args.sample_rate

    # Validate MOT directory
    sequences_dir = mot_dir / 'sequences'
    annotations_dir = mot_dir / 'annotations'
    if not sequences_dir.exists():
        raise SystemExit(f"sequences/ not found in {mot_dir}")
    if not annotations_dir.exists():
        raise SystemExit(f"annotations/ not found in {mot_dir}")

    # Create output directories
    output_images = output_dir / 'images'
    output_labels = output_dir / 'labels'
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Convert MOT sequences ---
    print(f"Converting MOT sequences (sampling every {sample_rate} frames)...")
    total_converted = 0
    total_skipped = 0

    seq_dirs = sorted([d for d in sequences_dir.iterdir() if d.is_dir()])
    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        ann_path = annotations_dir / f"{seq_name}.txt"
        if not ann_path.exists():
            print(f"  {seq_name}: annotation file not found, skipping")
            continue

        n_conv, n_skip = convert_mot_sequence(
            seq_dir, ann_path, output_images, output_labels, sample_rate, prefix=seq_name
        )
        total_converted += n_conv
        total_skipped += n_skip
        print(f"  {seq_name}: {n_conv} frames sampled, {n_skip} skipped")

    print(f"MOT total: {total_converted} frames converted, {total_skipped} skipped")

    # --- Step 2: Copy DET dataset (optional) ---
    if args.det_dir is not None:
        det_dir = args.det_dir.resolve()
        print(f"\nCopying DET dataset from {det_dir}...")
        n_det = copy_det_dataset(det_dir, output_images, output_labels)
        print(f"DET: {n_det} images copied")
        print(f"\nCombined total: {total_converted + n_det} images")
    else:
        print(f"\nNo --det-dir specified, MOT-only output: {total_converted} images")

    print(f"Output: {output_dir}")
    print(f"  images/ : {len(list(output_images.iterdir()))} files")
    print(f"  labels/ : {len(list(output_labels.iterdir()))} files")


if __name__ == '__main__':
    main()
