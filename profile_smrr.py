"""SMRR Profiling Script — Measure sub-threshold membrane potentials at false-negative locations.

Answers the question: "What fraction of missed objects are threshold-limited (recoverable by SMRR)?"

Usage:
    python profile_smrr.py --weights results/hazydet/hazydet4/weights/best.pt \
                           --data data/hazydet.yaml --img 640 --time-step 4

Output:
    - Histogram of max sub-threshold voltage at FN vs TP locations
    - Fraction of FN that are threshold-limited (v > 0.5)
    - Per-class breakdown
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_dataset, check_img_size,
                           check_yaml, colorstr, non_max_suppression, scale_boxes,
                           xywh2xyxy)
from utils.metrics import box_iou
from utils.torch_utils import select_device
from spikingjelly.activation_based.functional import reset_net
from spikingjelly.activation_based import neuron
from models.spike import set_time_step, BasicBlock2, BasicBlock1

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class MembraneVoltageProfiler:
    """Hooks into IFNode.act3 of BasicBlock2 layers to capture membrane voltages
    across all timesteps."""

    def __init__(self, model):
        self.hooks = []
        self.voltage_snapshots = {}  # layer_name → list of [T tensors]
        self._attach_hooks(model)

    def _attach_hooks(self, model):
        """Find all BasicBlock2 (and BasicBlock1) act3 IFNodes that feed SDDetect."""
        for name, module in model.named_modules():
            if isinstance(module, (BasicBlock2, BasicBlock1)):
                act3 = module.act3
                layer_name = name
                self.voltage_snapshots[layer_name] = []

                def make_hook(lname):
                    def hook_fn(m, inp, out):
                        # After each IFNode forward call (one per timestep),
                        # m.v holds the post-reset membrane potential.
                        # For non-firing neurons: v = accumulated voltage (sub-threshold)
                        # For firing neurons: v = 0 (hard reset)
                        self.voltage_snapshots[lname].append(m.v.detach().clone())
                    return hook_fn

                h = act3.register_forward_hook(make_hook(layer_name))
                self.hooks.append(h)
                LOGGER.info(f'  Hooked: {layer_name}.act3')

    def get_voltages(self):
        """Return captured voltages as {layer_name: [v_t1, v_t2, ..., v_tT]}."""
        return deepcopy(self.voltage_snapshots)

    def clear(self):
        """Clear captured voltages for next batch."""
        for k in self.voltage_snapshots:
            self.voltage_snapshots[k] = []

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def map_boxes_to_feature_grid(boxes, img_shape, feat_shape):
    """Map bounding box centers from image space to feature map grid cells.

    Args:
        boxes: [N, 4] xyxy in image pixel coords
        img_shape: (H_img, W_img)
        feat_shape: (H_feat, W_feat)

    Returns:
        grid_y, grid_x: [N] integer grid coordinates
    """
    cx = (boxes[:, 0] + boxes[:, 2]) / 2  # center x
    cy = (boxes[:, 1] + boxes[:, 3]) / 2  # center y

    # Scale to feature map
    scale_x = feat_shape[1] / img_shape[1]
    scale_y = feat_shape[0] / img_shape[0]

    grid_x = (cx * scale_x).long().clamp(0, feat_shape[1] - 1)
    grid_y = (cy * scale_y).long().clamp(0, feat_shape[0] - 1)

    return grid_y, grid_x


def extract_voltage_at_locations(voltages_per_layer, grid_y, grid_x):
    """Extract max-over-time, mean-over-channel voltage at given grid locations.

    Args:
        voltages_per_layer: list of T tensors, each [B, C, H, W] (single image → B=1 slice)
        grid_y, grid_x: [N] grid coordinates

    Returns:
        v_max: [N] max sub-threshold voltage across timesteps at each location
    """
    if len(voltages_per_layer) == 0 or len(grid_y) == 0:
        return torch.zeros(len(grid_y))

    T = len(voltages_per_layer)
    N = len(grid_y)

    v_at_locs = torch.zeros(T, N)
    for t in range(T):
        v_t = voltages_per_layer[t]  # [B, C, H, W] or [C, H, W]
        if v_t.dim() == 4:
            v_t = v_t[0]  # take first image in batch slice (we pass single image)
        # Mean over channels at each grid location
        for n in range(N):
            gy, gx = grid_y[n].item(), grid_x[n].item()
            if gy < v_t.shape[1] and gx < v_t.shape[2]:
                v_at_locs[t, n] = v_t[:, gy, gx].mean()

    # Max voltage across timesteps (the "closest miss")
    v_max, _ = v_at_locs.max(dim=0)
    return v_max


def run_profiling(
    weights,
    data,
    time_step=4,
    batch_size=1,
    imgsz=640,
    conf_thres=0.001,
    iou_thres=0.5,
    device='',
    workers=4,
    save_dir='runs/profile_smrr',
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    set_time_step(time_step)
    device = select_device(device, batch_size=batch_size)

    # Load model
    model = DetectMultiBackend(weights, device=device, data=data, fp16=False)
    stride, pt = model.stride, model.pt
    imgsz = check_img_size(imgsz, s=stride)
    model.eval()
    model.float()

    # Data
    data_dict = check_dataset(data)
    nc = int(data_dict['nc'])
    names = data_dict.get('names', {i: str(i) for i in range(nc)})
    if isinstance(names, list):
        names = dict(enumerate(names))

    # Dataloader (batch_size=1 for precise spatial mapping)
    dataloader = create_dataloader(
        data_dict['val'], imgsz, 1, 32, False,
        hyp='data/hyps/hyp.scratch-high.yaml',
        pad=0.5, rect=True, rank=-1, workers=workers,
        prefix=colorstr('profile: ')
    )[0]

    # Attach profiler hooks
    LOGGER.info('Attaching membrane voltage hooks...')
    inner_model = model.model if hasattr(model, 'model') else model
    profiler = MembraneVoltageProfiler(inner_model)

    # Storage for results
    fn_voltages = []          # max sub-threshold voltage at each false-negative location
    tp_voltages = []          # max sub-threshold voltage at each true-positive location
    fn_voltages_by_class = defaultdict(list)
    tp_voltages_by_class = defaultdict(list)
    fn_box_areas = []         # normalized box area of each FN
    tp_box_areas = []

    iouv = torch.tensor([0.5], device=device)  # single IoU threshold for FN/TP split

    LOGGER.info(f'Profiling {len(dataloader)} batches...')
    reset_net(inner_model)

    for batch_i, (im, targets, paths, shapes) in enumerate(tqdm(dataloader, bar_format=TQDM_BAR_FORMAT)):
        profiler.clear()

        im = im.to(device).float() / 255.0
        nb, _, height, width = im.shape

        # Forward pass (hooks capture voltages)
        preds = model(im)
        if isinstance(preds, tuple):
            preds = preds[0]

        # NMS
        targets_px = targets.clone()
        targets_px[:, 2:] *= torch.tensor((width, height, width, height), device='cpu')
        preds_nms = non_max_suppression(preds, conf_thres, iou_thres, max_det=300)

        # Get captured voltages
        voltages = profiler.get_voltages()

        # Process each image in batch (batch_size=1, so just si=0)
        for si in range(nb):
            pred = preds_nms[si]
            labels = targets[targets[:, 0] == si, 1:].to(device)  # [M, 5]: cls, cx, cy, w, h (normalized)

            if labels.shape[0] == 0:
                continue

            # Convert GT to pixel xyxy
            tbox_norm = labels[:, 1:5].clone()  # cx, cy, w, h normalized
            # Convert to xyxy in pixel coords
            tbox_px = xywh2xyxy(tbox_norm * torch.tensor([width, height, width, height], device=device))

            gt_classes = labels[:, 0].long()
            n_gt = labels.shape[0]

            # Compute box areas (normalized by image area for scale analysis)
            box_areas = ((tbox_px[:, 2] - tbox_px[:, 0]) * (tbox_px[:, 3] - tbox_px[:, 1])) / (width * height)

            # Determine which GT boxes are matched (TP) vs unmatched (FN)
            matched = torch.zeros(n_gt, dtype=torch.bool, device=device)

            if pred.shape[0] > 0:
                iou = box_iou(tbox_px, pred[:, :4])  # [M_gt, N_pred]
                correct_class = gt_classes[:, None] == pred[:, 5][None, :]  # [M_gt, N_pred]
                valid = (iou >= 0.5) & correct_class

                for gi in range(n_gt):
                    if valid[gi].any():
                        matched[gi] = True

            fn_mask = ~matched
            tp_mask = matched

            # Extract voltage at GT locations from each hooked layer
            for layer_name, v_list in voltages.items():
                if len(v_list) == 0:
                    continue

                feat_h, feat_w = v_list[0].shape[-2], v_list[0].shape[-1]

                # Slice voltages for this image in batch
                v_for_img = []
                for v_t in v_list:
                    if v_t.dim() == 4:
                        v_for_img.append(v_t[si])
                    else:
                        v_for_img.append(v_t)

                # Map GT boxes to this feature grid
                gy, gx = map_boxes_to_feature_grid(tbox_px, (height, width), (feat_h, feat_w))

                # Extract voltages
                v_max = extract_voltage_at_locations(v_for_img, gy, gx)

                # Store FN voltages
                for i in range(n_gt):
                    cls_id = gt_classes[i].item()
                    area = box_areas[i].item()
                    v = v_max[i].item()

                    if fn_mask[i]:
                        fn_voltages.append(v)
                        fn_voltages_by_class[cls_id].append(v)
                        fn_box_areas.append(area)
                    elif tp_mask[i]:
                        tp_voltages.append(v)
                        tp_voltages_by_class[cls_id].append(v)
                        tp_box_areas.append(area)

                # Only use the best (largest feature map) layer per GT box
                break

        reset_net(inner_model)

    profiler.remove_hooks()

    # ===== Analysis =====
    fn_v = np.array(fn_voltages) if fn_voltages else np.array([0.0])
    tp_v = np.array(tp_voltages) if tp_voltages else np.array([0.0])

    LOGGER.info('\n' + '=' * 70)
    LOGGER.info('SMRR PROFILING RESULTS')
    LOGGER.info('=' * 70)

    LOGGER.info(f'\nTotal ground truth boxes analyzed:')
    LOGGER.info(f'  True Positives (matched):    {len(tp_voltages)}')
    LOGGER.info(f'  False Negatives (missed):    {len(fn_voltages)}')

    # Threshold-limited analysis
    thresholds = [0.25, 0.5, 0.75, 0.9]
    LOGGER.info(f'\nFalse Negatives by sub-threshold voltage range:')
    LOGGER.info(f'  {"Voltage Range":<25} {"Count":>8} {"Fraction":>10} {"Meaning"}')
    LOGGER.info(f'  {"-"*70}')

    if len(fn_voltages) > 0:
        for tau in thresholds:
            count = (fn_v >= tau).sum()
            frac = count / len(fn_v)
            label = {0.25: 'some signal', 0.5: 'SMRR recoverable',
                     0.75: 'highly recoverable', 0.9: 'almost fired'}[tau]
            LOGGER.info(f'  v >= {tau:<20.2f} {count:>8d} {frac:>10.1%}   {label}')

        LOGGER.info(f'\n  Mean FN voltage:  {fn_v.mean():.3f}')
        LOGGER.info(f'  Median FN voltage: {np.median(fn_v):.3f}')
        LOGGER.info(f'  Mean TP voltage:  {tp_v.mean():.3f}')

    # Per-class breakdown
    LOGGER.info(f'\nPer-class breakdown (v >= 0.5 = threshold-limited):')
    LOGGER.info(f'  {"Class":<15} {"FN total":>10} {"FN v>=0.5":>10} {"Fraction":>10}')
    LOGGER.info(f'  {"-"*50}')
    for cls_id in sorted(set(list(fn_voltages_by_class.keys()) + list(tp_voltages_by_class.keys()))):
        cls_name = names.get(cls_id, str(cls_id))
        fn_cls = np.array(fn_voltages_by_class.get(cls_id, [0.0]))
        n_fn = len(fn_voltages_by_class.get(cls_id, []))
        n_recoverable = (fn_cls >= 0.5).sum() if n_fn > 0 else 0
        frac = n_recoverable / n_fn if n_fn > 0 else 0
        LOGGER.info(f'  {cls_name:<15} {n_fn:>10d} {n_recoverable:>10d} {frac:>10.1%}')

    # SMRR expected impact
    LOGGER.info(f'\n{"=" * 70}')
    if len(fn_voltages) > 0:
        recoverable_frac = (fn_v >= 0.5).sum() / len(fn_v)
        total_objects = len(fn_voltages) + len(tp_voltages)
        current_recall = len(tp_voltages) / total_objects if total_objects > 0 else 0
        recall_ceiling = current_recall + (1 - current_recall) * recoverable_frac
        LOGGER.info(f'SMRR IMPACT ESTIMATE:')
        LOGGER.info(f'  Current recall:                {current_recall:.3f}')
        LOGGER.info(f'  Threshold-limited FN fraction:  {recoverable_frac:.1%}')
        LOGGER.info(f'  Recall ceiling with SMRR:      {recall_ceiling:.3f}')
        LOGGER.info(f'  Max possible recall gain:      +{(recall_ceiling - current_recall):.3f}')
        if recoverable_frac > 0.4:
            LOGGER.info(f'\n  >>> SMRR is STRONGLY recommended for this model.')
        elif recoverable_frac > 0.15:
            LOGGER.info(f'\n  >>> SMRR is worth trying for this model.')
        else:
            LOGGER.info(f'\n  >>> SMRR unlikely to help — bottleneck is backbone capacity, not threshold.')
    LOGGER.info('=' * 70)

    # ===== Plots =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SMRR Profiling: Sub-Threshold Membrane Voltages', fontsize=14)

    # Plot 1: FN vs TP voltage histograms
    ax = axes[0, 0]
    bins = np.linspace(0, 1.0, 40)
    if len(fn_voltages) > 0:
        ax.hist(fn_v, bins=bins, alpha=0.7, label=f'False Negatives (n={len(fn_voltages)})',
                color='red', density=True)
    if len(tp_voltages) > 0:
        ax.hist(tp_v, bins=bins, alpha=0.7, label=f'True Positives (n={len(tp_voltages)})',
                color='green', density=True)
    ax.axvline(x=0.5, color='orange', linestyle='--', linewidth=2, label='SMRR threshold (tau=0.5)')
    ax.set_xlabel('Max sub-threshold voltage')
    ax.set_ylabel('Density')
    ax.set_title('Membrane Voltage Distribution: FN vs TP')
    ax.legend(fontsize=8)

    # Plot 2: FN voltage CDF
    ax = axes[0, 1]
    if len(fn_voltages) > 0:
        sorted_v = np.sort(fn_v)
        cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax.plot(sorted_v, cdf, color='red', linewidth=2)
        ax.axvline(x=0.5, color='orange', linestyle='--', label='tau=0.5')
        ax.axhline(y=(fn_v >= 0.5).mean(), color='blue', linestyle=':', alpha=0.5,
                   label=f'{(fn_v >= 0.5).mean():.0%} above tau')
    ax.set_xlabel('Max sub-threshold voltage')
    ax.set_ylabel('Cumulative fraction of FN')
    ax.set_title('FN Voltage CDF (above tau = recoverable)')
    ax.legend(fontsize=8)

    # Plot 3: Voltage vs box area (scatter)
    ax = axes[1, 0]
    if len(fn_voltages) > 0:
        ax.scatter(fn_box_areas, fn_v[:len(fn_box_areas)], alpha=0.3, s=8,
                   color='red', label='FN')
    if len(tp_voltages) > 0:
        ax.scatter(tp_box_areas, tp_v[:len(tp_box_areas)], alpha=0.3, s=8,
                   color='green', label='TP')
    ax.set_xlabel('Normalized box area')
    ax.set_ylabel('Max sub-threshold voltage')
    ax.set_title('Voltage vs Object Size')
    ax.legend(fontsize=8)
    ax.set_xscale('log')

    # Plot 4: Per-class bar chart
    ax = axes[1, 1]
    cls_ids = sorted(fn_voltages_by_class.keys())
    cls_labels = [names.get(c, str(c)) for c in cls_ids]
    recov_fracs = []
    for c in cls_ids:
        fn_cls = np.array(fn_voltages_by_class[c])
        recov_fracs.append((fn_cls >= 0.5).mean() if len(fn_cls) > 0 else 0)
    if cls_labels:
        bars = ax.bar(cls_labels, recov_fracs, color='steelblue')
        ax.axhline(y=0.4, color='orange', linestyle='--', label='Strong benefit threshold')
        ax.set_ylabel('Fraction of FN with v >= 0.5')
        ax.set_title('SMRR Recoverability by Class')
        ax.legend(fontsize=8)
        for bar, frac in zip(bars, recov_fracs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{frac:.0%}', ha='center', fontsize=9)

    plt.tight_layout()
    plot_path = save_dir / 'smrr_profile.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    LOGGER.info(f'\nPlots saved to {plot_path}')

    # Save raw data
    np.savez(save_dir / 'smrr_profile_data.npz',
             fn_voltages=fn_v,
             tp_voltages=tp_v,
             fn_box_areas=np.array(fn_box_areas),
             tp_box_areas=np.array(tp_box_areas))
    LOGGER.info(f'Raw data saved to {save_dir / "smrr_profile_data.npz"}')


def parse_opt():
    parser = argparse.ArgumentParser(description='SMRR Sub-Threshold Voltage Profiler')
    parser.add_argument('--weights', type=str, required=True, help='model checkpoint path')
    parser.add_argument('--data', type=str, default='data/hazydet.yaml', help='dataset yaml')
    parser.add_argument('--img', '--imgsz', type=int, default=640, help='inference image size')
    parser.add_argument('--time-step', type=int, default=4, help='SNN timesteps (must match training)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device (e.g. 0 or cpu)')
    parser.add_argument('--workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--save-dir', type=str, default='runs/profile_smrr', help='output directory')
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    run_profiling(
        weights=opt.weights,
        data=opt.data,
        time_step=opt.time_step,
        batch_size=1,
        imgsz=opt.img,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        device=opt.device,
        workers=opt.workers,
        save_dir=opt.save_dir,
    )
