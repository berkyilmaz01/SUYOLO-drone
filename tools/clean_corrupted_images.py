"""Scan HazyDet dataset for corrupted images and remove them (with their labels).

Corrupted images are typically 1 KB placeholder files left behind by failed
downloads from Baidu Netdisk or OneDrive.  They cannot be opened by PIL or
OpenCV and will crash training if not removed.

Usage:
  # Dry-run (default) -- only reports corrupted files, deletes nothing:
  python tools/clean_corrupted_images.py --data-root /path/to/HazyDet

  # Actually delete corrupted images and their matching label files:
  python tools/clean_corrupted_images.py --data-root /path/to/HazyDet --delete
"""

import argparse
import os
from pathlib import Path

from PIL import Image

IMG_FORMATS = {'.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png',
               '.tif', '.tiff', '.webp', '.pfm'}

# Files smaller than this are almost certainly corrupted placeholders.
# Real drone-view images are typically hundreds of KB to several MB.
MIN_BYTES = 10 * 1024  # 10 KB


def check_image(path):
    """Return an error string if the image is corrupted, else None."""
    # Quick size check -- catches the common 1 KB placeholder files.
    size = os.path.getsize(path)
    if size < MIN_BYTES:
        return f"too small ({size} bytes)"

    # Deep check -- actually try to decode the pixel data.
    try:
        with Image.open(path) as im:
            im.verify()
        # verify() does not catch all problems; re-open and load pixels.
        with Image.open(path) as im:
            im.load()
    except Exception as e:
        return str(e)

    return None


def find_label_for_image(img_path):
    """Return the matching YOLO label path, or None if it doesn't exist."""
    # .../images/foo.jpg  ->  .../labels/foo.txt
    p = Path(img_path)
    label_path = p.parents[1] / "labels" / (p.stem + ".txt")
    return label_path if label_path.exists() else None


def scan_split(split_dir, delete=False):
    """Scan one split (train / val / test) and return (total, corrupted)."""
    # Look inside hazy_images/ first, fall back to images/
    img_dir = split_dir / "hazy_images"
    if not img_dir.exists():
        img_dir = split_dir / "images"
    if not img_dir.exists():
        print(f"  No image directory found in {split_dir}, skipping")
        return 0, 0

    files = sorted(
        f for f in img_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMG_FORMATS
    )
    total = len(files)
    corrupted = 0

    for f in files:
        reason = check_image(f)
        if reason is None:
            continue

        corrupted += 1
        label = find_label_for_image(f)

        if delete:
            f.unlink()
            msg = "DELETED"
            if label:
                label.unlink()
                msg += " (+ label)"
        else:
            msg = "CORRUPTED (dry-run, use --delete to remove)"
            if label:
                msg += " (+ label)"

        print(f"  {f.name}: {reason} -- {msg}")

    return total, corrupted


def main():
    parser = argparse.ArgumentParser(
        description="Find and optionally remove corrupted images in HazyDet.")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory of HazyDet (contains train/val/test)")
    parser.add_argument("--delete", action="store_true",
                        help="Actually delete corrupted files (default is dry-run)")
    args = parser.parse_args()

    root = Path(args.data_root)
    if not root.exists():
        print(f"ERROR: {root} does not exist")
        return

    total_all, corrupted_all = 0, 0

    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            continue
        print(f"\nScanning {split}/...")
        total, corrupted = scan_split(split_dir, delete=args.delete)
        total_all += total
        corrupted_all += corrupted
        print(f"  {split}: {corrupted}/{total} corrupted images"
              f" {'removed' if args.delete else 'found'}")

    print(f"\nTotal: {corrupted_all}/{total_all} corrupted images"
          f" {'removed' if args.delete else 'found (re-run with --delete to remove)'}")

    if corrupted_all and not args.delete:
        print("\nTo remove them, re-run with --delete:")
        print(f"  python tools/clean_corrupted_images.py "
              f"--data-root {args.data_root} --delete")


if __name__ == "__main__":
    main()
