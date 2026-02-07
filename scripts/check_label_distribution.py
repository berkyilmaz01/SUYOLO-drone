#!/usr/bin/env python3
"""
Diagnose label distribution in YOLO-format dataset.
Checks: class counts, bounding box size distribution, and per-class stats.

Usage:
  python scripts/check_label_distribution.py --labels-dir ../VisDrone/combined_train/labels
  python scripts/check_label_distribution.py --labels-dir ../VisDrone/MOT-train/labels
"""

import argparse
from pathlib import Path
from collections import Counter
import statistics

CLASS_NAMES = {
    0: 'pedestrian',
    1: 'people',
    2: 'bicycle',
    3: 'car',
    4: 'van',
    5: 'truck',
    6: 'tricycle',
    7: 'awning-tricycle',
    8: 'bus',
    9: 'motor',
}


def analyze_labels(labels_dir):
    labels_dir = Path(labels_dir)
    if not labels_dir.exists():
        print(f"ERROR: {labels_dir} does not exist!")
        return

    class_counts = Counter()
    class_widths = {i: [] for i in range(10)}
    class_heights = {i: [] for i in range(10)}
    class_areas = {i: [] for i in range(10)}
    total_files = 0
    empty_files = 0
    files_with_pedestrians = 0
    files_with_cars = 0

    for label_file in sorted(labels_dir.glob('*.txt')):
        total_files += 1
        has_ped = False
        has_car = False
        lines = label_file.read_text().strip().split('\n')
        if not lines or lines == ['']:
            empty_files += 1
            continue
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            w = float(parts[3])  # normalized width
            h = float(parts[4])  # normalized height
            class_counts[cls_id] += 1
            if cls_id < 10:
                class_widths[cls_id].append(w)
                class_heights[cls_id].append(h)
                class_areas[cls_id].append(w * h)
            if cls_id == 0 or cls_id == 1:
                has_ped = True
            if cls_id == 3:
                has_car = True
        if has_ped:
            files_with_pedestrians += 1
        if has_car:
            files_with_cars += 1

    # Print results
    print(f"\n{'='*70}")
    print(f"Label Distribution Analysis: {labels_dir}")
    print(f"{'='*70}")
    print(f"Total label files: {total_files}")
    print(f"Empty label files: {empty_files}")
    print(f"Files with pedestrian/people: {files_with_pedestrians} ({100*files_with_pedestrians/max(total_files,1):.1f}%)")
    print(f"Files with cars: {files_with_cars} ({100*files_with_cars/max(total_files,1):.1f}%)")

    print(f"\n{'Class':<20} {'Count':>8} {'%':>7}  {'Median W':>9} {'Median H':>9} {'Median Area':>12}")
    print('-' * 70)
    total = sum(class_counts.values())
    for cls_id in range(10):
        count = class_counts.get(cls_id, 0)
        pct = 100 * count / max(total, 1)
        name = CLASS_NAMES.get(cls_id, f'class_{cls_id}')
        if count > 0:
            med_w = statistics.median(class_widths[cls_id])
            med_h = statistics.median(class_heights[cls_id])
            med_a = statistics.median(class_areas[cls_id])
            print(f"{name:<20} {count:>8} {pct:>6.1f}%  {med_w:>9.4f} {med_h:>9.4f} {med_a:>12.6f}")
        else:
            print(f"{name:<20} {count:>8} {pct:>6.1f}%  {'N/A':>9} {'N/A':>9} {'N/A':>12}")

    print(f"\nTotal annotations: {total}")

    # Show what bbox sizes mean at different resolutions
    print(f"\n{'='*70}")
    print("Median bbox sizes in PIXELS at different training resolutions:")
    print(f"{'='*70}")
    resolutions = [(1280, 736), (1920, 1080), (640, 640)]
    for cls_id in [0, 1, 3]:  # pedestrian, people, car
        name = CLASS_NAMES[cls_id]
        if class_counts.get(cls_id, 0) == 0:
            continue
        med_w = statistics.median(class_widths[cls_id])
        med_h = statistics.median(class_heights[cls_id])
        sizes = []
        for res_w, res_h in resolutions:
            pw = med_w * res_w
            ph = med_h * res_h
            sizes.append(f"{res_w}x{res_h}: {pw:.0f}x{ph:.0f}px")
        print(f"  {name:<15} → {' | '.join(sizes)}")

    # Check for suspicious patterns
    print(f"\n{'='*70}")
    print("DIAGNOSTICS:")
    print(f"{'='*70}")
    if class_counts.get(0, 0) == 0 and class_counts.get(1, 0) == 0:
        print("WARNING: ZERO pedestrian AND people annotations found!")
        print("  → Check if your raw annotations have categories 1 and 2")
        print("  → Check if conversion script ran correctly")

    # Sample some label files to show raw content
    print(f"\n--- Sample labels from first 3 files with class 0 or 1 ---")
    shown = 0
    for label_file in sorted(labels_dir.glob('*.txt')):
        lines = label_file.read_text().strip().split('\n')
        has_human = any(line.strip().startswith('0 ') or line.strip().startswith('1 ') for line in lines if line.strip())
        if has_human:
            print(f"\n  {label_file.name}:")
            for line in lines[:5]:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_name = CLASS_NAMES.get(int(parts[0]), '?')
                    print(f"    {line.strip()}  → {cls_name}")
            if len(lines) > 5:
                print(f"    ... ({len(lines)} total annotations)")
            shown += 1
            if shown >= 3:
                break
    if shown == 0:
        print("  NO files found with pedestrian/people labels!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels-dir', type=Path, required=True)
    args = parser.parse_args()
    analyze_labels(args.labels_dir)
