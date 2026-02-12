"""SMRR Profiling Script — Measure sub-threshold membrane potentials at false-negative locations.

Answers the question: "What fraction of missed objects are threshold-limited (recoverable by SMRR)?"

Usage:
    python profile_smrr.py --weights results/hazydet/hazydet4/weights/best.pt \
                           --data data/hazydet.yaml --img 640 --time-step 1

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
from models.spike import set_time_step

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class MembraneVoltageProfiler:
    """Hooks into ALL IFNode instances in the model to capture membrane voltages."""

    def __init__(self, model):
        self.hooks = []
        self.voltage_snapshots = {}  # layer_name → list of (v_tensor, shape) tuples
        self.hook_count = 0
        self._attach_hooks(model)

    def _attach_hooks(self, model):
        """Find ALL IFNode instances and hook them."""
        for name, module in model.named_modules():
            if isinstance(module, neuron.IFNode):
                layer_name = name
                self.voltage_snapshots[layer_name] = []

                def make_hook(lname):
                    def hook_fn(m, inp, out):
                        # m.v holds post-reset membrane potential
                        # For non-firing neurons: v = accumulated voltage (sub-threshold)
                        # For firing neurons: v = 0 (hard reset)
                        v = m.v
                        if isinstance(v, torch.Tensor):
                            self.voltage_snapshots[lname].append(v.detach().cpu().clone())
                        # else: v is scalar (0.0 after reset) — skip
                    return hook_fn

                h = module.register_forward_hook(make_hook(layer_name))
                self.hooks.append(h)
                self.hook_count += 1

        LOGGER.info(f'  Attached {self.hook_count} hooks on IFNode instances')
        for name in self.voltage_snapshots:
            LOGGER.info(f'    - {name}')

    def get_voltages(self):
        """Return captured voltages (already on CPU)."""
        # Return direct reference — no deepcopy needed since we clone+cpu in hook
        result = {}
        for k, v_list in self.voltage_snapshots.items():
            result[k] = list(v_list)  # shallow copy of list
        return result

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
        boxes: [N, 4] xyxy in image pixel coords (can be on any device)
        img_shape: (H_img, W_img)
        feat_shape: (H_feat, W_feat)

    Returns:
        grid_y, grid_x: [N] integer grid coordinates (CPU)
    """
    boxes_cpu = boxes.detach().cpu().float()
    cx = (boxes_cpu[:, 0] + boxes_cpu[:, 2]) / 2
    cy = (boxes_cpu[:, 1] + boxes_cpu[:, 3]) / 2

    scale_x = feat_shape[1] / img_shape[1]
    scale_y = feat_shape[0] / img_shape[0]

    grid_x = (cx * scale_x).long().clamp(0, feat_shape[1] - 1)
    grid_y = (cy * scale_y).long().clamp(0, feat_shape[0] - 1)

    return grid_y, grid_x


def extract_voltage_at_locations(voltages_list, grid_y, grid_x, si=0):
    """Extract mean-over-channel voltage at given grid locations.

    Args:
        voltages_list: list of T tensors (CPU), each [B, C, H, W] or [C, H, W]
        grid_y, grid_x: [N] grid coordinates (CPU)
        si: batch image index

    Returns:
        v_max: [N] max sub-threshold voltage across timesteps
    """
    if len(voltages_list) == 0 or len(grid_y) == 0:
        return torch.zeros(len(grid_y))

    T = len(voltages_list)
    N = len(grid_y)
    v_at_locs = torch.zeros(T, N)

    for t in range(T):
        v_t = voltages_list[t]
        # Handle different dims
        if v_t.dim() == 4:
            v_t = v_t[si]  # [C, H, W]
        if v_t.dim() != 3:
            continue

        C, H, W = v_t.shape
        for n in range(N):
            gy = grid_y[n].item()
            gx = grid_x[n].item()
            if 0 <= gy < H and 0 <= gx < W:
                v_at_locs[t, n] = v_t[:, gy, gx].float().mean().item()

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

    # Attach profiler hooks — search ALL nesting levels
    LOGGER.info('\nAttaching membrane voltage hooks...')
    inner_model = model.model if hasattr(model, 'model') else model
    profiler = MembraneVoltageProfiler(inner_model)

    if profiler.hook_count == 0:
        LOGGER.error('NO IFNode hooks attached! Model structure may be unexpected.')
        LOGGER.info('Model modules found:')
        for name, mod in inner_model.named_modules():
            LOGGER.info(f'  {name}: {type(mod).__name__}')
        return

    # Storage
    fn_voltages = []
    tp_voltages = []
    fn_voltages_by_class = defaultdict(list)
    tp_voltages_by_class = defaultdict(list)
    fn_box_areas = []
    tp_box_areas = []

    # Identify which hooked layers have the largest spatial dims (closest to input)
    # We'll determine this from the first batch
    best_layers = None  # will be set after first batch

    LOGGER.info(f'Profiling {len(dataloader)} batches...\n')
    reset_net(inner_model)
    debug_printed = False

    for batch_i, (im, targets, paths, shapes) in enumerate(tqdm(dataloader, bar_format=TQDM_BAR_FORMAT)):
        profiler.clear()

        im = im.to(device).float() / 255.0
        nb, _, height, width = im.shape

        # Forward pass — hooks capture voltages
        with torch.no_grad():
            preds = model(im)

        # Handle model output format
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        # NMS
        preds_nms = non_max_suppression(preds, conf_thres, iou_thres, max_det=300)

        # Get captured voltages (already on CPU from hook)
        voltages = profiler.get_voltages()

        # Debug: print info for first batch
        if not debug_printed:
            LOGGER.info(f'\n--- DEBUG (batch 0) ---')
            LOGGER.info(f'  Image shape: {im.shape}')
            LOGGER.info(f'  Targets shape: {targets.shape}')
            LOGGER.info(f'  Preds after NMS: {[p.shape for p in preds_nms]}')
            LOGGER.info(f'  Voltage snapshots per layer:')
            layer_sizes = {}
            for lname, vlist in voltages.items():
                if len(vlist) > 0:
                    shapes_str = [str(tuple(v.shape)) for v in vlist]
                    LOGGER.info(f'    {lname}: {len(vlist)} snapshots, shapes={shapes_str}')
                    # Track spatial size for layer selection
                    spatial = vlist[0].shape[-2] * vlist[0].shape[-1]
                    layer_sizes[lname] = spatial
                else:
                    LOGGER.info(f'    {lname}: EMPTY (no voltages captured)')

            if not layer_sizes:
                LOGGER.error('\n  ALL voltage snapshots are EMPTY — hooks did not fire!')
                LOGGER.error('  This likely means the model architecture does not use IFNode')
                LOGGER.error('  in the expected way. Aborting.\n')
                profiler.remove_hooks()
                return

            # Pick layers with largest spatial dims (best for small objects)
            sorted_layers = sorted(layer_sizes.items(), key=lambda x: -x[1])
            # Use top-2 largest spatial layers (likely the FPN outputs before detect)
            best_layers = [name for name, _ in sorted_layers[:2]]
            LOGGER.info(f'  Using layers: {best_layers}')

            # Debug target info
            labels_0 = targets[targets[:, 0] == 0, 1:]
            LOGGER.info(f'  Labels for image 0: {labels_0.shape}')
            if labels_0.shape[0] > 0:
                LOGGER.info(f'    First label: cls={labels_0[0, 0]:.0f}, '
                           f'xywh={labels_0[0, 1:].tolist()}')
            LOGGER.info(f'--- END DEBUG ---\n')
            debug_printed = True

        # Process each image in batch
        for si in range(nb):
            pred = preds_nms[si].cpu()  # [N_pred, 6] on CPU
            labels = targets[targets[:, 0] == si, 1:]  # [M, 5] cls,cx,cy,w,h (normalized)

            if labels.shape[0] == 0:
                continue

            n_gt = labels.shape[0]
            gt_classes = labels[:, 0].long()

            # Convert GT boxes to pixel xyxy
            tbox_px = xywh2xyxy(
                labels[:, 1:5] * torch.tensor([width, height, width, height], dtype=torch.float32)
            )

            # Box areas (normalized)
            box_areas = ((tbox_px[:, 2] - tbox_px[:, 0]) *
                         (tbox_px[:, 3] - tbox_px[:, 1])) / (width * height)

            # Match GT to predictions (all on CPU)
            matched = torch.zeros(n_gt, dtype=torch.bool)

            if pred.shape[0] > 0:
                iou = box_iou(tbox_px, pred[:, :4])  # [M_gt, N_pred]
                correct_class = gt_classes[:, None] == pred[:, 5].long()[None, :]
                valid = (iou >= 0.5) & correct_class

                for gi in range(n_gt):
                    if valid[gi].any():
                        matched[gi] = True

            fn_mask = ~matched
            tp_mask = matched

            # Extract voltage from the best layers
            collected_for_this_image = False
            for layer_name in (best_layers or voltages.keys()):
                v_list = voltages.get(layer_name, [])
                if len(v_list) == 0:
                    continue

                feat_h, feat_w = v_list[0].shape[-2], v_list[0].shape[-1]
                gy, gx = map_boxes_to_feature_grid(tbox_px, (height, width), (feat_h, feat_w))
                v_max = extract_voltage_at_locations(v_list, gy, gx, si=si)

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

                collected_for_this_image = True
                break  # use the first (largest) available layer

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

    if len(fn_voltages) == 0 and len(tp_voltages) == 0:
        LOGGER.error('\n  NO objects were processed. Possible causes:')
        LOGGER.error('  1. Voltage hooks captured empty data (check DEBUG output above)')
        LOGGER.error('  2. Dataset labels could not be loaded')
        LOGGER.error('  3. Prediction/label coordinate mismatch')
        return

    # Threshold-limited analysis
    thresholds = [0.25, 0.5, 0.75, 0.9]
    LOGGER.info(f'\nFalse Negatives by sub-threshold voltage range:')
    LOGGER.info(f'  {"Voltage Range":<25} {"Count":>8} {"Fraction":>10} {"Meaning"}')
    LOGGER.info(f'  {"-"*70}')

    if len(fn_voltages) > 0:
        for tau in thresholds:
            count = int((fn_v >= tau).sum())
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
        n_recoverable = int((fn_cls >= 0.5).sum()) if n_fn > 0 else 0
        frac = n_recoverable / n_fn if n_fn > 0 else 0
        LOGGER.info(f'  {cls_name:<15} {n_fn:>10d} {n_recoverable:>10d} {frac:>10.1%}')

    # SMRR expected impact
    LOGGER.info(f'\n{"=" * 70}')
    if len(fn_voltages) > 0:
        recoverable_frac = float((fn_v >= 0.5).sum()) / len(fn_v)
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
        ax.plot(sorted_v, cdf, color='red', linewidth=2, label='FN CDF')
        ax.axvline(x=0.5, color='orange', linestyle='--', label='tau=0.5')
        frac_above = float((fn_v >= 0.5).mean())
        ax.axhline(y=1-frac_above, color='blue', linestyle=':', alpha=0.5,
                   label=f'{frac_above:.0%} above tau')
    ax.set_xlabel('Max sub-threshold voltage')
    ax.set_ylabel('Cumulative fraction of FN')
    ax.set_title('FN Voltage CDF (above tau = recoverable)')
    ax.legend(fontsize=8)

    # Plot 3: Voltage vs box area (scatter)
    ax = axes[1, 0]
    if len(fn_voltages) > 0 and len(fn_box_areas) > 0:
        ax.scatter(fn_box_areas, fn_v[:len(fn_box_areas)], alpha=0.3, s=8,
                   color='red', label='FN')
    if len(tp_voltages) > 0 and len(tp_box_areas) > 0:
        ax.scatter(tp_box_areas, tp_v[:len(tp_box_areas)], alpha=0.3, s=8,
                   color='green', label='TP')
    ax.set_xlabel('Normalized box area')
    ax.set_ylabel('Max sub-threshold voltage')
    ax.set_title('Voltage vs Object Size')
    ax.legend(fontsize=8)
    if len(fn_box_areas) > 0 or len(tp_box_areas) > 0:
        ax.set_xscale('log')

    # Plot 4: Per-class bar chart
    ax = axes[1, 1]
    cls_ids = sorted(fn_voltages_by_class.keys())
    cls_labels = [names.get(c, str(c)) for c in cls_ids]
    recov_fracs = []
    for c in cls_ids:
        fn_cls = np.array(fn_voltages_by_class[c])
        recov_fracs.append(float((fn_cls >= 0.5).mean()) if len(fn_cls) > 0 else 0)
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
