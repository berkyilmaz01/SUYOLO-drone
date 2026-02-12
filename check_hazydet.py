"""Quick sanity check for HazyDet dataset paths and label alignment."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HAZYDET = ROOT.parent / "HazyDet"

print(f"HazyDet root: {HAZYDET}")
print(f"Exists: {HAZYDET.exists()}\n")

if not HAZYDET.exists():
    print("ERROR: ../HazyDet not found. Check your dataset location.")
    exit(1)

# ── 1. Check directory structure ──
dirs = {
    "train/images":       "clean training images",
    "train/hazy_images":  "hazy training images",
    "train/labels":       "training labels",
    "val/images":         "val images",
    "val/labels":         "val labels",
    "test/images":        "test images",
    "test/labels":        "test labels",
    "real_train/images":  "real hazy train (Stage 2)",
    "real_train/labels":  "real hazy train labels",
    "real_test/images":   "real hazy test (Stage 2)",
    "real_test/labels":   "real hazy test labels",
}

print("=" * 60)
print("DIRECTORY CHECK")
print("=" * 60)
for d, desc in dirs.items():
    p = HAZYDET / d
    if p.exists():
        n = len(list(p.iterdir()))
        print(f"  OK   {d:30s} ({n:5d} files) — {desc}")
    else:
        print(f"  MISS {d:30s}              — {desc}")

# ── 2. Check label<->image alignment for each split ──
print("\n" + "=" * 60)
print("LABEL <-> IMAGE ALIGNMENT")
print("=" * 60)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

pairs = [
    ("train/hazy_images", "train/labels",      "hazydet.yaml (hazy train)"),
    ("train/images",      "train/labels",      "hazydet-clean.yaml (clean train)"),
    ("val/images",        "val/labels",        "val"),
    ("test/images",       "test/labels",       "test"),
    ("real_train/images", "real_train/labels", "hazydet-real.yaml (real train)"),
    ("real_test/images",  "real_test/labels",  "hazydet-real.yaml (real test)"),
]

for img_dir, lbl_dir, name in pairs:
    img_path = HAZYDET / img_dir
    lbl_path = HAZYDET / lbl_dir
    if not img_path.exists() or not lbl_path.exists():
        print(f"\n  SKIP  {name} — directory missing")
        continue

    imgs = {f.stem for f in img_path.iterdir() if f.suffix.lower() in IMG_EXT}
    lbls = {f.stem for f in lbl_path.iterdir() if f.suffix == ".txt"}

    matched = imgs & lbls
    imgs_only = imgs - lbls
    lbls_only = lbls - imgs

    print(f"\n  {name}:")
    print(f"    Images: {len(imgs):5d}   Labels: {len(lbls):5d}   Matched: {len(matched):5d}")
    if imgs_only:
        print(f"    WARNING: {len(imgs_only)} images have NO label")
        for f in sorted(imgs_only)[:3]:
            print(f"      e.g. {f}")
    if lbls_only:
        print(f"    WARNING: {len(lbls_only)} labels have NO image")
        for f in sorted(lbls_only)[:3]:
            print(f"      e.g. {f}")
    if not imgs_only and not lbls_only:
        print(f"    PERFECT — all images have labels and vice versa")

# ── 3. Check YOLO auto-discovery (images->labels path swap) ──
print("\n" + "=" * 60)
print("YOLO PATH SWAP CHECK")
print("=" * 60)
print("  YOLO replaces 'images' with 'labels' in the path string.")
print()

swaps = [
    ("train/hazy_images", "train/hazy_labels", "hazydet.yaml"),
    ("train/images",      "train/labels",      "hazydet-clean.yaml"),
    ("val/images",        "val/labels",        "val set"),
    ("test/images",       "test/labels",       "test set"),
    ("real_train/images", "real_train/labels",  "hazydet-real.yaml"),
    ("real_test/images",  "real_test/labels",   "hazydet-real.yaml"),
]

for img_dir, expected_lbl_dir, name in swaps:
    # Simulate YOLO's path swap
    yolo_lbl = img_dir.replace("images", "labels")
    lbl_path = HAZYDET / yolo_lbl
    print(f"  {name}:")
    print(f"    Image dir:    {img_dir}")
    print(f"    YOLO expects: {yolo_lbl}")
    if lbl_path.exists():
        print(f"    OK — exists")
    else:
        # Check if a symlink would fix it
        actual = HAZYDET / expected_lbl_dir.replace("hazy_labels", "labels")
        if actual.exists() and "hazy" in yolo_lbl:
            print(f"    FAIL — missing! Fix with:")
            print(f"      cd {HAZYDET / Path(yolo_lbl).parent} && ln -s labels {Path(yolo_lbl).name}")
        else:
            print(f"    FAIL — missing!")
    print()

print("=" * 60)
print("DONE")
print("=" * 60)
