"""Convert HazyDet COCO JSON annotations to YOLO format.

Usage:
  python tools/coco2yolo_hazydet.py --data-root /path/to/HazyDet

Expects:
  data-root/
    train/  train_coco.json, hazy_images/
    val/    val_coco.json, hazy_images/
    test/   test_coco.json, hazy_images/

Produces YOLO labels and an 'images' symlink for each split:
  data-root/
    train/  labels/  (one .txt per image)
            images -> hazy_images  (symlink for YOLO label resolution)
    val/    labels/
            images -> hazy_images
    test/   labels/
            images -> hazy_images
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict


def convert_split(split_dir):
    split_dir = Path(split_dir)

    # Find the COCO JSON
    json_files = list(split_dir.glob("*_coco.json"))
    if not json_files:
        print(f"  No COCO JSON found in {split_dir}, skipping")
        return 0

    json_path = json_files[0]
    print(f"  Loading {json_path.name}...")

    with open(json_path) as f:
        coco = json.load(f)

    # Build image id -> (width, height, filename) map
    images = {}
    for img in coco["images"]:
        images[img["id"]] = {
            "w": img["width"],
            "h": img["height"],
            "file_name": img["file_name"],
        }

    # Print categories
    print(f"  Categories: {coco.get('categories', [])}")

    # Group annotations by image_id
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    # Create labels directory
    labels_dir = split_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    count = 0
    for img_id, img_info in images.items():
        w, h = img_info["w"], img_info["h"]
        fname = Path(img_info["file_name"]).stem + ".txt"

        lines = []
        for ann in anns_by_image.get(img_id, []):
            cat_id = ann["category_id"]  # already 0-indexed (0=car, 1=truck, 2=bus)
            bx, by, bw, bh = ann["bbox"]  # COCO format: x_top_left, y_top_left, width, height

            # Convert to YOLO format: x_center, y_center, width, height (normalized)
            x_center = (bx + bw / 2) / w
            y_center = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h

            # Clamp to [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))

            if nw > 0 and nh > 0:
                lines.append(f"{cat_id} {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}")

        with open(labels_dir / fname, "w") as f:
            f.write("\n".join(lines))

        count += 1

    print(f"  Wrote {count} label files to {labels_dir}")

    # Create 'images' symlink -> 'hazy_images' so YOLO img2label_paths
    # can resolve /images/ -> /labels/ correctly.
    images_link = split_dir / "images"
    hazy_dir = split_dir / "hazy_images"
    if hazy_dir.exists() and not images_link.exists():
        images_link.symlink_to("hazy_images")
        print(f"  Created symlink {images_link} -> hazy_images")

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory containing train/val/test folders")
    args = parser.parse_args()

    root = Path(args.data_root)

    for split in ["train", "val", "test"]:
        split_dir = root / split
        if split_dir.exists():
            print(f"\nConverting {split}...")
            convert_split(split_dir)
        else:
            print(f"\n{split}/ not found, skipping")


if __name__ == "__main__":
    main()
