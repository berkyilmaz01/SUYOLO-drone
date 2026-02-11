"""Create symlink directories so YOLO can load hazy_images with existing labels.

YOLO's img2label_paths() replaces '/images/' with '/labels/' in image paths.
Since hazy images live in 'hazy_images/' (no '/images/' component), we create
a parallel directory structure using symlinks:

  HazyDet/
    hazy_train/
      images -> ../train/hazy_images   (YOLO sees '/images/' in path)
      labels -> ../train/labels         (YOLO resolves labels correctly)
    hazy_val/
      images -> ../val/hazy_images
      labels -> ../val/labels
    hazy_test/
      images -> ../test/hazy_images
      labels -> ../test/labels

Usage:
  python tools/setup_hazy_symlinks.py --data-root /path/to/HazyDet
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory of HazyDet (contains train/val/test)")
    args = parser.parse_args()

    root = Path(args.data_root)

    for split in ["train", "val", "test"]:
        src_hazy = root / split / "hazy_images"
        src_labels = root / split / "labels"

        if not src_hazy.exists():
            print(f"  {src_hazy} not found, skipping {split}")
            continue

        hazy_dir = root / f"hazy_{split}"
        hazy_dir.mkdir(exist_ok=True)

        # images -> ../train/hazy_images
        img_link = hazy_dir / "images"
        if not img_link.exists():
            img_link.symlink_to(f"../{split}/hazy_images")
            print(f"  Created {img_link} -> ../{split}/hazy_images")
        else:
            print(f"  {img_link} already exists")

        # labels -> ../train/labels
        lbl_link = hazy_dir / "labels"
        if not lbl_link.exists():
            if src_labels.exists():
                lbl_link.symlink_to(f"../{split}/labels")
                print(f"  Created {lbl_link} -> ../{split}/labels")
            else:
                print(f"  WARNING: {src_labels} not found — run coco2yolo_hazydet.py first")
        else:
            print(f"  {lbl_link} already exists")

    print("\nDone! You can now train with: data/hazydet-hazy.yaml")


if __name__ == "__main__":
    main()
