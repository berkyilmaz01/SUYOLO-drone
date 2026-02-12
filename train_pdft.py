"""Progressive Domain Fine-Tuning (PDFT) for HazyDet.

Implements Algorithm 1 from:
  "HazyDet: Open-Source Benchmark for Drone-View Object Detection
   with Depth-Cues in Hazy Scenes" (Feng et al., 2025)

Two-stage progressive adaptation:
  Stage 1: Normal/clean -> Simulated haze (intermediate domain)
    - Initialize from pretrained weights (e.g. COCO or ImageNet-pretrained)
    - Fine-tune on synthetic hazy data while freezing early backbone layers
    - Bridges the gap between clear and hazy feature distributions

  Stage 2: Simulated haze -> Real haze (target domain)
    - Initialize from Stage 1 checkpoint
    - Fine-tune on real hazy data with more frozen layers + reduced LR
    - Prevents overfitting on the smaller real-world dataset
    - Preserves learned haze-aware features from Stage 1

Usage:
  # Full PDFT pipeline (both stages):
  python train_pdft.py --weights pretrained.pt \
      --cfg models/detect/su-yolo-720p.yaml \
      --data-synth data/hazydet.yaml \
      --data-real data/hazydet-real.yaml \
      --hyp data/hyps/hyp.hazydet.yaml \
      --epochs-s1 200 --epochs-s2 100 \
      --img 736 --time-step 4

  # Stage 1 only (synthetic haze fine-tuning):
  python train_pdft.py --weights pretrained.pt \
      --cfg models/detect/su-yolo-720p.yaml \
      --data-synth data/hazydet.yaml \
      --hyp data/hyps/hyp.hazydet.yaml \
      --epochs-s1 200 --stage 1 \
      --img 736 --time-step 4

  # Stage 2 only (real haze fine-tuning from Stage 1 checkpoint):
  python train_pdft.py --weights runs/train/pdft_stage1/weights/best.pt \
      --cfg models/detect/su-yolo-720p.yaml \
      --data-real data/hazydet-real.yaml \
      --hyp data/hyps/hyp.hazydet.yaml \
      --epochs-s2 100 --stage 2 \
      --img 736 --time-step 4
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import train, parse_opt
from utils.general import LOGGER, colorstr, increment_path


def parse_pdft_opt():
    parser = argparse.ArgumentParser(description="PDFT: Progressive Domain Fine-Tuning for HazyDet")

    # Model
    parser.add_argument("--weights", type=str, default="", help="Initial weights (pretrained or Stage 1 checkpoint)")
    parser.add_argument("--cfg", type=str, default="models/detect/su-yolo-720p.yaml", help="Model architecture yaml")
    parser.add_argument("--hyp", type=str, default="data/hyps/hyp.hazydet.yaml", help="Hyperparameters yaml")

    # Data
    parser.add_argument("--data-synth", type=str, default="data/hazydet.yaml",
                        help="Synthetic hazy dataset yaml (Stage 1)")
    parser.add_argument("--data-real", type=str, default="data/hazydet-real.yaml",
                        help="Real hazy dataset yaml (Stage 2)")

    # Training
    parser.add_argument("--epochs-s1", type=int, default=200,
                        help="Epochs for Stage 1 (synthetic haze fine-tuning)")
    parser.add_argument("--epochs-s2", type=int, default=100,
                        help="Epochs for Stage 2 (real haze fine-tuning)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", "--img", type=int, default=736, help="Training image size")
    parser.add_argument("--time-step", type=int, default=4, help="SNN time steps")
    parser.add_argument("--device", default="", help="CUDA device (e.g. 0 or 0,1,2,3)")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam", "AdamW", "LION"])
    parser.add_argument("--cos-lr", action="store_true", help="Cosine LR scheduler")

    # PDFT-specific
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2],
                        help="Run specific stage only (0=both, 1=synth only, 2=real only)")
    parser.add_argument("--freeze-s1", nargs="+", type=int, default=[1],
                        help="Layers to freeze in Stage 1 (default: first backbone block)")
    parser.add_argument("--freeze-s2", nargs="+", type=int, default=[3],
                        help="Layers to freeze in Stage 2 (default: first 3 backbone blocks)")
    parser.add_argument("--lr-decay", type=float, default=0.1,
                        help="LR reduction factor for Stage 2 (gamma in Algorithm 1, default 0.1)")
    parser.add_argument("--patience-s1", type=int, default=50,
                        help="Early stopping patience for Stage 1")
    parser.add_argument("--patience-s2", type=int, default=30,
                        help="Early stopping patience for Stage 2")

    # Output
    parser.add_argument("--project", default="runs/train", help="Save to project/name")
    parser.add_argument("--name", default="pdft", help="Experiment name prefix")

    return parser.parse_args()


def build_train_opt(pdft_args, stage, data, epochs, freeze, lr_scale=1.0, weights="", name_suffix=""):
    """Build a train.py-compatible argparse.Namespace for a PDFT stage."""
    # Start from default train opts to get all required fields
    sys_argv_backup = sys.argv
    sys.argv = ["train.py"]  # minimal argv for parse_opt defaults
    opt = parse_opt(known=True)
    sys.argv = sys_argv_backup

    opt.weights = weights or pdft_args.weights
    opt.cfg = pdft_args.cfg
    opt.data = data
    opt.hyp = pdft_args.hyp
    opt.time_step = pdft_args.time_step
    opt.epochs = epochs
    opt.batch_size = pdft_args.batch_size
    opt.imgsz = pdft_args.imgsz
    opt.device = pdft_args.device
    opt.workers = pdft_args.workers
    opt.optimizer = pdft_args.optimizer
    opt.cos_lr = pdft_args.cos_lr
    opt.freeze = freeze
    opt.project = pdft_args.project
    opt.name = f"{pdft_args.name}_{name_suffix}"
    opt.patience = pdft_args.patience_s1 if stage == 1 else pdft_args.patience_s2
    opt.exist_ok = False
    opt.resume = False
    opt.evolve = None
    opt.save_period = -1
    opt.close_mosaic = 15
    opt.noval = False
    opt.nosave = False
    opt.noautoanchor = True
    opt.noplots = False
    opt.rect = False
    opt.multi_scale = False
    opt.single_cls = False
    opt.sync_bn = False
    opt.image_weights = False
    opt.quad = False
    opt.flat_cos_lr = False
    opt.fixed_lr = False
    opt.label_smoothing = 0.0
    opt.seed = 0
    opt.local_rank = -1
    opt.min_items = 0
    opt.cache = None
    opt.bucket = ""
    opt.entity = None
    opt.upload_dataset = False
    opt.bbox_interval = -1
    opt.artifact_alias = "latest"

    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=False))

    return opt


def run_pdft(pdft_args):
    """Execute PDFT pipeline."""

    stage1_best = ""

    # ── Stage 1: Normal/Clean -> Simulated Haze ──
    if pdft_args.stage in (0, 1):
        LOGGER.info(colorstr("bold", "magenta", "\n=== PDFT Stage 1: Synthetic Haze Fine-Tuning ==="))
        LOGGER.info(f"  Data:    {pdft_args.data_synth}")
        LOGGER.info(f"  Weights: {pdft_args.weights}")
        LOGGER.info(f"  Epochs:  {pdft_args.epochs_s1}")
        LOGGER.info(f"  Freeze:  first {pdft_args.freeze_s1} layer(s)")
        LOGGER.info(f"  LR:      base (1x)")

        opt_s1 = build_train_opt(
            pdft_args,
            stage=1,
            data=pdft_args.data_synth,
            epochs=pdft_args.epochs_s1,
            freeze=pdft_args.freeze_s1,
            name_suffix="stage1_synth",
        )

        import yaml
        from utils.callbacks import Callbacks
        from utils.general import check_file, check_yaml
        from utils.torch_utils import select_device

        opt_s1.data = check_file(opt_s1.data)
        opt_s1.cfg = check_yaml(opt_s1.cfg)
        opt_s1.hyp = check_yaml(opt_s1.hyp)

        device = select_device(opt_s1.device, batch_size=opt_s1.batch_size)
        callbacks = Callbacks()
        train(opt_s1.hyp, opt_s1, device, callbacks)

        stage1_best = str(Path(opt_s1.save_dir) / "weights" / "best.pt")
        LOGGER.info(colorstr("bold", "green", f"\n  Stage 1 complete. Best weights: {stage1_best}"))

    # ── Stage 2: Simulated Haze -> Real Haze ──
    if pdft_args.stage in (0, 2):
        LOGGER.info(colorstr("bold", "magenta", "\n=== PDFT Stage 2: Real Haze Fine-Tuning ==="))

        # Use Stage 1 output or provided weights
        s2_weights = stage1_best if stage1_best else pdft_args.weights
        if not s2_weights:
            LOGGER.error("Stage 2 requires weights from Stage 1 or --weights flag")
            return

        # Scale learning rate down for Stage 2
        lr_factor = pdft_args.lr_decay

        LOGGER.info(f"  Data:    {pdft_args.data_real}")
        LOGGER.info(f"  Weights: {s2_weights}")
        LOGGER.info(f"  Epochs:  {pdft_args.epochs_s2}")
        LOGGER.info(f"  Freeze:  first {pdft_args.freeze_s2} layer(s)")
        LOGGER.info(f"  LR:      {lr_factor}x (reduced)")

        opt_s2 = build_train_opt(
            pdft_args,
            stage=2,
            data=pdft_args.data_real,
            epochs=pdft_args.epochs_s2,
            freeze=pdft_args.freeze_s2,
            weights=s2_weights,
            name_suffix="stage2_real",
        )

        import yaml
        from utils.callbacks import Callbacks
        from utils.general import check_file, check_yaml
        from utils.torch_utils import select_device

        opt_s2.data = check_file(opt_s2.data)
        opt_s2.cfg = check_yaml(opt_s2.cfg)
        opt_s2.hyp = check_yaml(opt_s2.hyp)

        # Apply LR reduction for Stage 2
        with open(opt_s2.hyp) as f:
            hyp = yaml.safe_load(f)
        hyp["lr0"] *= lr_factor
        LOGGER.info(f"  Adjusted lr0: {hyp['lr0']}")

        device = select_device(opt_s2.device, batch_size=opt_s2.batch_size)
        callbacks = Callbacks()
        train(hyp, opt_s2, device, callbacks)

        stage2_best = str(Path(opt_s2.save_dir) / "weights" / "best.pt")
        LOGGER.info(colorstr("bold", "green", f"\n  Stage 2 complete. Best weights: {stage2_best}"))
        LOGGER.info(colorstr("bold", "green", "  PDFT pipeline finished."))


def main():
    pdft_args = parse_pdft_opt()

    LOGGER.info(colorstr("bold", "cyan", """
    ╔══════════════════════════════════════════════════╗
    ║  PDFT: Progressive Domain Fine-Tuning           ║
    ║  for HazyDet Drone Detection                    ║
    ║                                                  ║
    ║  Stage 1: Clean/ImageNet -> Synthetic Haze       ║
    ║  Stage 2: Synthetic Haze -> Real Haze            ║
    ╚══════════════════════════════════════════════════╝
    """))

    run_pdft(pdft_args)


if __name__ == "__main__":
    main()
