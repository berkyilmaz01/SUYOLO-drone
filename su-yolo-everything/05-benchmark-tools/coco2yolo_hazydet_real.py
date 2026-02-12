"""Convert HazyDet real-world COCO JSON annotations to YOLO format.

Usage:
  python tools/coco2yolo_hazydet_real.py --data-root /path/to/HazyDet/real_world

Expects:
  real_world/
    train/          (images directly inside)
    test/           (images directly inside)
    train_real.json
    test_real.json

Produces:
  real_world/
    train/images/   (images moved here)
    train/labels/   (YOLO .txt files)
    test/images/    (images moved here)
    test/labels/    (YOLO .txt files)
"""
import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict


def convert_split(root, split, json_name):
    split_dir = root / split
    json_path = root / json_name

    if not split_dir.exists():
        print(f"  {split_dir} not found, skipping")
        return 0
    if not json_path.exists():
        print(f"  {json_path} not found, skipping")
        return 0

    print(f"  Loading {json_path.name}...")
    with open(json_path) as f:
        coco = json.load(f)

    # Print categories to verify mapping
    cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
    print(f"  Categories: {cats}")

    # Build category ID remapping → 0-indexed (car=0, truck=1, bus=2)
    name_to_yolo = {"car": 0, "truck": 1, "bus": 2}
    cat_remap = {}
    for cat in coco.get("categories", []):
        if cat["name"].lower() in name_to_yolo:
            cat_remap[cat["id"]] = name_to_yolo[cat["name"].lower()]
        else:
            print(f"  WARNING: unknown category '{cat['name']}' (id={cat['id']}), skipping")

    # Build image id → info map
    images = {}
    for img in coco["images"]:
        images[img["id"]] = {
            "w": img["width"],
            "h": img["height"],
            "file_name": img["file_name"],
        }

    # Group annotations by image_id
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    # Create images/ and labels/ subdirectories
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    # Move images from split_dir/ into split_dir/images/ (skip dirs)
    moved = 0
    for f in sorted(split_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            dst = images_dir / f.name
            if not dst.exists():
                shutil.move(str(f), str(dst))
                moved += 1
    if moved:
        print(f"  Moved {moved} images into {images_dir}")

    # Write YOLO label files
    count = 0
    skipped_cats = 0
    for img_id, img_info in images.items():
        w, h = img_info["w"], img_info["h"]
        fname = Path(img_info["file_name"]).stem + ".txt"

        lines = []
        for ann in anns_by_image.get(img_id, []):
            cat_id = ann["category_id"]
            if cat_id not in cat_remap:
                skipped_cats += 1
                continue
            yolo_cls = cat_remap[cat_id]

            bx, by, bw, bh = ann["bbox"]  # COCO: x_top_left, y_top_left, w, h

            # Convert to YOLO: x_center, y_center, width, height (normalized)
            x_center = max(0, min(1, (bx + bw / 2) / w))
            y_center = max(0, min(1, (by + bh / 2) / h))
            nw = max(0, min(1, bw / w))
            nh = max(0, min(1, bh / h))

            if nw > 0 and nh > 0:
                lines.append(f"{yolo_cls} {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}")

        with open(labels_dir / fname, "w") as f:
            f.write("\n".join(lines))
        count += 1

    print(f"  Wrote {count} label files to {labels_dir}")
    if skipped_cats:
        print(f"  Skipped {skipped_cats} annotations with unknown category IDs")

    total_anns = sum(len(v) for v in anns_by_image.values())
    print(f"  Total annotations: {total_anns}")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to HazyDet/real_world directory")
    args = parser.parse_args()

    root = Path(args.data_root)

    print("\nConverting train...")
    convert_split(root, "train", "train_real.json")

    print("\nConverting test...")
    convert_split(root, "test", "test_real.json")

    print("\nDone! Update data/hazydet-real.yaml paths to:")
    print(f"  path: {root.parent}")
    print(f"  train: real_world/train/images")
    print(f"  val: real_world/test/images")
    print(f"  test: real_world/test/images")


if __name__ == "__main__":
    main()
