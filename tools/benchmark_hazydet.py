"""Benchmark comparison tool for HazyDet drone detection experiments.

Compares SU-YOLO results against published baselines from:
  "HazyDet: Open-Source Benchmark for Drone-View Object Detection
   with Depth-Cues in Hazy Scenes" (Feng et al., 2025)

Usage:
  # After training, evaluate and compare:
  python tools/benchmark_hazydet.py --weights runs/train/exp/weights/best.pt \
      --data data/hazydet.yaml --img 736 --time-step 4

  # Just print the leaderboard (no evaluation):
  python tools/benchmark_hazydet.py --leaderboard

  # Add a custom entry to the leaderboard:
  python tools/benchmark_hazydet.py --add-entry --name "SU-YOLO-720p (ep300)" \
      --map-test 0.465 --map-real 0.390 --fps 85.0 --params 7.0
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Published baselines from HazyDet paper Table 3
# Format: (name, epochs, fps, params_M, gflops, mAP_synth_test, mAP_real_test)
PAPER_BASELINES = [
    # Specified (haze-aware) detectors
    ("IA-YOLO",          273, 22.7, 61.80,  37.30, 0.383, 0.224),
    ("TogetherNet",      300, 49.5, 69.20,  19.71, 0.446, 0.252),
    ("MS-DAYOLO",        300, 57.9, 40.00,  44.20, 0.483, 0.365),
    # One-stage
    ("YOLOv3",           273, 68.1, 61.63,  20.19, 0.350, 0.307),
    ("GFL",               12, 51.1, 32.26, 198.65, 0.368, 0.325),
    ("YOLOX",            300, 71.2,  8.94,  13.32, 0.423, 0.354),
    ("FCOS",              12, 60.6, 32.11, 191.48, 0.459, 0.327),
    ("VFNet",             12, 59.9, 32.71, 184.32, 0.495, 0.356),
    ("ATSS",              12, 49.6, 32.12, 195.58, 0.504, 0.364),
    ("DDOD",              12, 47.6, 32.20, 173.05, 0.507, 0.371),
    ("TOOD",              12, 48.0, 32.02, 192.51, 0.514, 0.367),
    # Two-stage
    ("Sparse RCNN",       12, 47.1,108.54, 147.45, 0.277, 0.208),
    ("Faster RCNN",       12, 46.4, 41.35, 201.72, 0.487, 0.334),
    ("Libra RCNN",        12, 44.7, 41.62, 209.92, 0.490, 0.345),
    ("Grid RCNN",         12, 47.2, 64.46, 317.44, 0.505, 0.352),
    ("Cascade RCNN",      12, 42.9, 69.15, 230.40, 0.516, 0.372),
    # End-to-end
    ("Conditional DETR",  50, 40.3, 43.55,  94.17, 0.305, 0.258),
    ("DAB-DETR",          50, 41.7, 43.70,  97.02, 0.313, 0.272),
    ("Deformable DETR",   50, 50.9, 40.01, 203.11, 0.515, 0.369),
    # Paper's method
    ("DeCoDet (SOTA)",    12, 59.7, 34.62, 225.37, 0.520, 0.387),
]

# Clean-image training baselines from Table 5 (trained on clean, tested on hazy)
# These show the domain gap when no haze-specific training is used.
# Format: (name, mAP_synth_test, mAP_real_test)
CLEAN_BASELINES = [
    ("Faster RCNN (clean-trained)",    0.395, 0.215),
    ("+ GridDehaze",                   0.389, 0.196),
    ("+ MixDehazeNet",                 0.399, 0.212),
    ("+ DSANet",                       0.408, 0.224),
    ("+ FFA-Net",                      0.412, 0.220),
    ("+ DehazeFormer",                 0.425, 0.219),
    ("+ gUNet",                        0.427, 0.222),
    ("+ C2PNet",                       0.429, 0.224),
    ("+ DCP",                          0.440, 0.206),
    ("+ RIDCP",                        0.448, 0.242),
]


def print_leaderboard(entries=None, sort_by="mAP_synth", include_clean=False):
    """Print a formatted leaderboard comparing all methods."""
    all_entries = []
    for name, ep, fps, params, gflops, map_s, map_r in PAPER_BASELINES:
        all_entries.append({
            "name": name, "epochs": ep, "fps": fps, "params": params,
            "gflops": gflops, "mAP_synth": map_s, "mAP_real": map_r,
            "source": "paper"
        })

    if include_clean:
        for name, map_s, map_r in CLEAN_BASELINES:
            all_entries.append({
                "name": name, "epochs": 12, "fps": 0, "params": 41.35,
                "gflops": 0, "mAP_synth": map_s, "mAP_real": map_r,
                "source": "clean"
            })

    if entries:
        all_entries.extend(entries)

    # Sort
    sort_key = sort_by if sort_by in all_entries[0] else "mAP_synth"
    all_entries.sort(key=lambda x: x.get(sort_key, 0), reverse=True)

    # Find best values for highlighting
    best_map_s = max(e["mAP_synth"] for e in all_entries if e["mAP_synth"] > 0)
    best_map_r = max(e["mAP_real"] for e in all_entries if e["mAP_real"] > 0)

    # Header
    print("\n" + "=" * 105)
    print("  HazyDet Benchmark Leaderboard")
    print("  Paper: 'HazyDet: Drone-View Object Detection with Depth-Cues in Hazy Scenes'")
    print("=" * 105)
    hdr = f"{'Rank':<5} {'Model':<25} {'Epochs':<8} {'FPS':<8} {'Params(M)':<11} {'mAP(Synth)':<12} {'mAP(Real)':<12} {'Source':<8}"
    print(hdr)
    print("-" * 105)

    for rank, e in enumerate(all_entries, 1):
        name = e["name"]
        map_s_str = f"{e['mAP_synth']:.3f}"
        map_r_str = f"{e['mAP_real']:.3f}" if e["mAP_real"] > 0 else "  —"

        # Mark best with asterisk
        if e["mAP_synth"] == best_map_s:
            map_s_str += " *"
        if e["mAP_real"] == best_map_r and e["mAP_real"] > 0:
            map_r_str += " *"

        # Highlight our entries
        marker = ">>>" if e["source"] != "paper" else "   "
        ep_str = str(e["epochs"]) if e["epochs"] > 0 else "—"
        fps_str = f"{e['fps']:.1f}" if e["fps"] > 0 else "—"
        params_str = f"{e['params']:.2f}" if e["params"] > 0 else "—"

        print(f"{marker}{rank:<2} {name:<25} {ep_str:<8} {fps_str:<8} {params_str:<11} {map_s_str:<12} {map_r_str:<12} {e['source']:<8}")

    print("=" * 105)
    print("  * = best in column  |  >>> = your model")
    print()


def evaluate_model(weights, data, imgsz, time_step, device, batch_size):
    """Run validation and return results dict."""
    import val as validate
    from models.experimental import attempt_load
    from utils.general import check_dataset, check_img_size
    from utils.torch_utils import select_device

    device = select_device(device)

    # Load model
    model = attempt_load(weights, device=device)
    gs = 32
    imgsz = check_img_size(imgsz, s=gs)

    # Load dataset config
    data_dict = check_dataset(data)

    results, maps, _ = validate.run(
        data=data_dict,
        time_step=time_step,
        weights=weights,
        batch_size=batch_size,
        imgsz=imgsz,
        model=model,
        device=device,
        single_cls=False,
        plots=False,
        verbose=True,
    )

    # results = (P, R, mAP@0.5, mAP@0.5:0.95, box_loss, obj_loss, cls_loss)
    return {
        "precision": results[0],
        "recall": results[1],
        "mAP_50": results[2],
        "mAP_50_95": results[3],
        "per_class_ap": maps.tolist() if hasattr(maps, 'tolist') else list(maps),
    }


def load_results_log(log_path):
    """Load previous benchmark results from JSON log."""
    if Path(log_path).exists():
        with open(log_path) as f:
            return json.load(f)
    return []


def save_results_log(log_path, entries):
    """Save benchmark results to JSON log."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(entries, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="HazyDet Benchmark Comparison Tool")
    parser.add_argument("--weights", type=str, default="", help="Model weights path (.pt)")
    parser.add_argument("--data", type=str, default="data/hazydet.yaml", help="Dataset yaml path")
    parser.add_argument("--img", type=int, default=736, help="Inference image size")
    parser.add_argument("--time-step", type=int, default=4, help="SNN time steps")
    parser.add_argument("--device", default="", help="CUDA device (e.g. 0)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--leaderboard", action="store_true", help="Print leaderboard only (no eval)")
    parser.add_argument("--sort-by", type=str, default="mAP_synth",
                        choices=["mAP_synth", "mAP_real", "fps", "params"],
                        help="Sort leaderboard by column")
    parser.add_argument("--include-clean", action="store_true",
                        help="Include clean-image training baselines (Table 5 from paper)")
    # Manual entry
    parser.add_argument("--add-entry", action="store_true", help="Add a manual result entry")
    parser.add_argument("--name", type=str, default="SU-YOLO", help="Model name for entry")
    parser.add_argument("--map-test", type=float, default=0.0, help="mAP on synthetic test set")
    parser.add_argument("--map-real", type=float, default=0.0, help="mAP on real-world test set")
    parser.add_argument("--fps", type=float, default=0.0, help="Inference FPS")
    parser.add_argument("--params", type=float, default=0.0, help="Model parameters (M)")
    parser.add_argument("--epochs", type=int, default=0, help="Training epochs")
    parser.add_argument("--log", type=str, default="runs/benchmark/hazydet_results.json",
                        help="Path to results log file")

    args = parser.parse_args()

    log_path = ROOT / args.log
    custom_entries = []

    # Load any previous custom entries
    saved = load_results_log(log_path)
    for s in saved:
        custom_entries.append(s)

    if args.add_entry:
        entry = {
            "name": args.name,
            "epochs": args.epochs,
            "fps": args.fps,
            "params": args.params,
            "gflops": 0,
            "mAP_synth": args.map_test,
            "mAP_real": args.map_real,
            "source": "ours"
        }
        custom_entries.append(entry)
        save_results_log(log_path, custom_entries)
        print(f"Added entry: {args.name} (mAP_synth={args.map_test}, mAP_real={args.map_real})")

    if args.weights and not args.leaderboard:
        print(f"Evaluating {args.weights} on HazyDet...")
        results = evaluate_model(
            args.weights, args.data, args.img, args.time_step, args.device, args.batch_size
        )
        print(f"\nResults:")
        print(f"  Precision:     {results['precision']:.4f}")
        print(f"  Recall:        {results['recall']:.4f}")
        print(f"  mAP@0.5:       {results['mAP_50']:.4f}")
        print(f"  mAP@0.5:0.95:  {results['mAP_50_95']:.4f}")
        if results.get("per_class_ap"):
            class_names = ["car", "truck", "bus"]
            for i, ap in enumerate(results["per_class_ap"]):
                if i < len(class_names):
                    print(f"  AP({class_names[i]}):    {ap:.4f}")

        # Use mAP@0.5 as the comparable metric (paper's mAP)
        entry = {
            "name": args.name if args.name != "SU-YOLO" else f"SU-YOLO ({Path(args.weights).parent.parent.name})",
            "epochs": args.epochs,
            "fps": args.fps,
            "params": args.params,
            "gflops": 0,
            "mAP_synth": results["mAP_50"],
            "mAP_real": args.map_real,
            "source": "ours"
        }
        custom_entries.append(entry)
        save_results_log(log_path, custom_entries)

    print_leaderboard(custom_entries, sort_by=args.sort_by, include_clean=args.include_clean)


if __name__ == "__main__":
    main()
