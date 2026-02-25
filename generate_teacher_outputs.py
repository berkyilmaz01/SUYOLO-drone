"""Generate teacher soft labels and feature maps for offline knowledge distillation.

Runs a trained GELAN-C teacher model on the training set and saves per-image:
  - P3/8, P4/16, P5/32 neck features (before detection head)
  - Raw detection logits per scale (cls + bbox, before sigmoid/softmax)
  - Teacher metadata (reg_max, nc, feature_layers)

Usage:
    python generate_teacher_outputs.py \
        --weights runs/train/hazydet-teacher-gelanc/weights/best.pt \
        --data data/hazydet.yaml --img 1920 --device 0 \
        --output-dir teacher_outputs
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.experimental import attempt_load
from models.yolo import DDetect
from models.spike import set_time_step
from spikingjelly.activation_based.functional import reset_net
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, check_dataset, check_img_size, check_yaml,
                           colorstr)
from utils.torch_utils import select_device


def extract_features_and_logits(model, img, feature_layers):
    """Forward pass with hooks to capture intermediate feature maps.

    Args:
        model: GELAN-C teacher model
        img: input tensor (B, 3, H, W)
        feature_layers: list of layer indices to capture features from

    Returns:
        features: dict mapping layer_idx -> feature tensor
        det_logits: list of raw detection logits per scale
    """
    features = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            features[layer_idx] = output.detach()
        return hook_fn

    for idx in feature_layers:
        h = model.model[idx].register_forward_hook(make_hook(idx))
        hooks.append(h)

    with torch.no_grad():
        pred = model(img)

    for h in hooks:
        h.remove()

    # Extract raw detection logits from the last layer output
    # DDetect returns (y, x) where x is the list of raw per-scale logits
    if isinstance(pred, tuple) and len(pred) == 2:
        det_logits = pred[1]  # list of raw logits per detection scale
    else:
        det_logits = pred

    return features, det_logits


def run(weights, data, imgsz, device, output_dir, batch_size=1, workers=4):
    set_time_step(1)
    device = select_device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load teacher model
    LOGGER.info(f"Loading teacher model from {weights}")
    model = attempt_load(weights, device=device)
    model.eval()
    model.half()
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(imgsz, s=gs)

    # Identify detection head and neck feature layer indices
    # GELAN-C: layer 15 = P3/8 neck output, layer 18 = P4/16 neck output
    detect_module = model.model[-1]
    if not isinstance(detect_module, DDetect):
        LOGGER.warning("Teacher model's last layer is not DDetect, feature indices may be wrong")

    # The detect head's .f attribute tells us which layers feed into it
    # For GELAN-C: DDetect.f = [15, 18, 21] -> P3, P4, P5 neck outputs
    detect_from = detect_module.f
    LOGGER.info(f"Detection head reads from layers: {detect_from}")
    feature_layers = list(detect_from)  # capture all neck outputs

    # Load dataset (no augmentation, just letterbox)
    data_dict = check_dataset(check_yaml(data))
    train_path = data_dict['train']

    dataloader, dataset = create_dataloader(
        train_path, imgsz, batch_size, gs,
        single_cls=False, hyp=None, augment=False,
        cache=False, rect=False, rank=-1,
        workers=workers, image_weights=False,
        close_mosaic=False, quad=False,
        prefix=colorstr('teacher: '))

    LOGGER.info(f"Processing {len(dataset)} images -> {output_dir}")

    num_saved = 0
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc="Generating teacher outputs")):
        imgs = imgs.to(device, non_blocking=True).half() / 255.0
        reset_net(model)

        features, det_logits = extract_features_and_logits(model, imgs, feature_layers)

        for j in range(imgs.shape[0]):
            img_path = Path(paths[j])
            stem = img_path.stem

            output = {}

            # Save neck features at each scale
            for k, layer_idx in enumerate(feature_layers):
                feat = features[layer_idx][j].half().cpu()
                output[f'feat_{k}'] = feat

            # Save layer indices so the student knows which is which
            output['feature_layers'] = feature_layers

            # Save raw detection logits per scale
            for k in range(len(det_logits)):
                logit = det_logits[k][j].half().cpu()
                output[f'det_logits_{k}'] = logit

            output['num_det_scales'] = len(det_logits)
            output['teacher_reg_max'] = detect_module.reg_max
            output['teacher_nc'] = detect_module.nc

            save_path = output_dir / f'{stem}.pt'
            torch.save(output, save_path)
            num_saved += 1

    LOGGER.info(f"Saved {num_saved} teacher output files to {output_dir}")
    total_size = sum(f.stat().st_size for f in output_dir.glob('*.pt')) / 1e9
    LOGGER.info(f"Total disk usage: {total_size:.1f} GB")


def parse_opt():
    parser = argparse.ArgumentParser(description='Generate teacher outputs for knowledge distillation')
    parser.add_argument('--weights', type=str, required=True, help='teacher model weights path')
    parser.add_argument('--data', type=str, required=True, help='dataset.yaml path')
    parser.add_argument('--img', '--imgsz', type=int, default=1920, help='inference image size')
    parser.add_argument('--device', default='0', help='cuda device')
    parser.add_argument('--output-dir', type=str, default='teacher_outputs', help='output directory for .pt files')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (1 recommended for 1080p)')
    parser.add_argument('--workers', type=int, default=4, help='dataloader workers')
    return parser.parse_args()


def main():
    opt = parse_opt()
    run(opt.weights, opt.data, opt.img, opt.device, opt.output_dir,
        opt.batch_size, opt.workers)


if __name__ == '__main__':
    main()
