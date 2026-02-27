"""Deep model diagnostics: per-class, per-scale, error type, bottleneck,
sparsity, calibration, and KD gain analysis.

Produces charts and a text report that pinpoint exactly where a model
succeeds, fails, and where knowledge distillation helps.

Usage:
    python tools/diagnose_model.py \
        --weights runs/train/hazydet-student-kd-fixed3/weights/best.pt \
        --baseline runs/train/hazydet-student-kd-scratch/weights/best.pt \
        --teacher runs/train/hazydet-teacher-gelanc5/weights/best.pt \
        --data data/hazydet.yaml --img 1920 --device 0 \
        --save-dir runs/diagnostics/
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from utils.general import (check_dataset, check_img_size, check_yaml,
                            colorstr, non_max_suppression, scale_boxes,
                            xywh2xyxy, xyxy2xywh)
from utils.metrics import ap_per_class, box_iou, ConfusionMatrix
from utils.torch_utils import select_device

CLASS_NAMES = ['car', 'truck', 'bus']


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(weights, device, is_snn=False):
    if is_snn:
        from models.spike import set_time_step
        set_time_step(1)
    ckpt = torch.load(weights, map_location='cpu', weights_only=False)
    model = (ckpt.get('ema') or ckpt['model']).float()
    if not hasattr(model, 'stride'):
        model.stride = torch.tensor([32.])
    if hasattr(model, 'names') and isinstance(model.names, (list, tuple)):
        model.names = dict(enumerate(model.names))
    return model.to(device).eval()


def get_dataloader(data, imgsz, batch_size, task='val'):
    from utils.dataloaders import create_dataloader
    data_dict = check_dataset(check_yaml(data))
    path = data_dict.get('val' if task == 'val' else 'test', data_dict.get('val'))
    gs = 32
    imgsz = check_img_size(imgsz, s=gs)
    dataloader = create_dataloader(
        path, imgsz, batch_size, gs, pad=0.5, rect=True,
        prefix=colorstr(f'{task}: '))[0]
    return dataloader, data_dict


# ---------------------------------------------------------------------------
# 1. Per-class performance comparison
# ---------------------------------------------------------------------------

def collect_predictions(model, dataloader, device, conf_thres=0.001,
                        iou_thres=0.5, is_snn=False):
    """Run inference and collect raw (correct, conf, pred_cls, target_cls) stats."""
    if is_snn:
        from spikingjelly.activation_based.functional import reset_net

    nc = model.nc if hasattr(model, 'nc') else 3
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()
    stats = []
    all_preds = []   # (img_idx, x1,y1,x2,y2, conf, cls)
    all_labels = []  # (img_idx, cls, x1,y1,x2,y2)

    for batch_i, (im, targets, paths, shapes) in enumerate(dataloader):
        if is_snn:
            reset_net(model)
        im = im.to(device, non_blocking=True).float() / 255
        targets = targets.to(device)
        nb, _, height, width = im.shape

        with torch.no_grad():
            preds = model(im)
        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        preds = non_max_suppression(preds, conf_thres, iou_thres, max_det=300)

        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []
            shape = shapes[si][0]

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros((0, niou), dtype=torch.bool, device=device),
                                  torch.Tensor(), torch.Tensor(), labels[:, 0]))
                    for lbl in labels:
                        c, x, y, w, h = lbl.tolist()
                        all_labels.append([batch_i * nb + si, c, x, y, w, h])
                continue

            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])

            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct = _process_batch(predn, labelsn, iouv)
            else:
                correct = torch.zeros(len(pred), niou, dtype=torch.bool, device=device)

            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))

            for p in predn:
                all_preds.append([batch_i * nb + si] + p[:6].cpu().tolist())
            for lbl in labels:
                c, x, y, w, h = lbl.tolist()
                all_labels.append([batch_i * nb + si, c, x, y, w, h])

    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    return stats, all_preds, all_labels


def _process_batch(detections, labels, iouv):
    """Compute correct prediction matrix (from val.py)."""
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=detections.device)


def per_class_metrics(stats, nc=3):
    """Extract per-class P, R, AP50, AP50-95 from stats."""
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(
            *stats, plot=False, save_dir='.', names={i: CLASS_NAMES[i] for i in range(nc)})
        ap50 = ap[:, 0]
        ap95 = ap.mean(1)
        return {
            'classes': [CLASS_NAMES[int(c)] for c in ap_class],
            'P': p.tolist(),
            'R': r.tolist(),
            'AP50': ap50.tolist(),
            'AP50-95': ap95.tolist(),
            'F1': f1.tolist(),
        }
    return None


def plot_per_class_comparison(results_dict, save_dir):
    """Grouped bar chart: per-class AP50 for each model."""
    models = list(results_dict.keys())
    if not models:
        return

    ref = results_dict[models[0]]
    if ref is None:
        return
    classes = ref['classes']
    n_classes = len(classes)
    n_models = len(models)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (metric, title) in enumerate([('AP50', 'AP@0.5'), ('AP50-95', 'AP@0.5:0.95')]):
        ax = axes[ax_idx]
        x = np.arange(n_classes)
        width = 0.8 / n_models

        for i, model_name in enumerate(models):
            data = results_dict[model_name]
            if data is None:
                continue
            vals = data[metric]
            offset = (i - n_models / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=model_name)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7)

        ax.set_xlabel('Class')
        ax.set_ylabel(title)
        ax.set_title(f'Per-Class {title}')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(save_dir / 'per_class_comparison.png', dpi=200)
    plt.close(fig)
    print(f'  Saved per_class_comparison.png')


# ---------------------------------------------------------------------------
# 2. Per-scale detection analysis
# ---------------------------------------------------------------------------

def per_scale_analysis(all_preds, all_labels, model, img_size, nc=3):
    """Assign each prediction to its likely detection scale and compute
    per-scale mAP.  Predictions are assigned by matching anchor stride to
    object size: P2/4 for small, P3/8 for medium, P4/16 for large."""

    strides = model.model[-1].stride.cpu().numpy() if hasattr(model.model[-1], 'stride') else np.array([4, 8, 16])
    scale_names = [f'P{int(np.log2(s))}/{int(s)}' for s in strides]

    if not all_labels:
        return None

    labels_arr = np.array(all_labels)  # (N, 6) img_idx, cls, x,y,w,h
    preds_arr = np.array(all_preds) if all_preds else np.zeros((0, 7))

    areas = labels_arr[:, 4] * labels_arr[:, 5] * img_size * img_size if len(labels_arr) > 0 else np.array([])

    area_thresholds = [0, 32**2, 96**2, float('inf')]
    scale_labels = ['Small (<32px)', 'Medium (32-96px)', 'Large (>96px)']

    result = {}
    for si, (lo, hi, sname) in enumerate(zip(area_thresholds[:-1], area_thresholds[1:], scale_labels)):
        mask = (areas >= lo) & (areas < hi)
        n_gt = mask.sum()
        result[sname] = {
            'n_gt': int(n_gt),
            'scale_stride': strides[min(si, len(strides) - 1)] if si < len(strides) else strides[-1],
        }

    return result, scale_labels


def plot_per_scale(scale_results_dict, save_dir):
    """Bar chart showing GT distribution by object size."""
    if not scale_results_dict:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    model_names = list(scale_results_dict.keys())
    ref = scale_results_dict[model_names[0]]
    if ref is None:
        return

    scales = list(ref.keys())
    x = np.arange(len(scales))
    gt_counts = [ref[s]['n_gt'] for s in scales]

    ax.bar(x, gt_counts, color=['#2196F3', '#FF9800', '#4CAF50'])
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.set_ylabel('Number of Ground Truth Objects')
    ax.set_title('Object Size Distribution (Ground Truth)')
    for i, v in enumerate(gt_counts):
        ax.text(i, v + max(gt_counts) * 0.01, str(v), ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig.savefig(save_dir / 'object_size_distribution.png', dpi=200)
    plt.close(fig)
    print(f'  Saved object_size_distribution.png')


# ---------------------------------------------------------------------------
# 3. Error type decomposition (TIDE-style)
# ---------------------------------------------------------------------------

def error_decomposition(all_preds, all_labels, nc=3, conf_thres=0.25,
                        iou_thres_correct=0.5, iou_thres_loose=0.1):
    """Classify each detection/GT into error categories.

    Returns dict with counts of:
      - correct: right class, IoU >= 0.5
      - cls_error: wrong class, IoU >= 0.5
      - loc_error: right class, 0.1 <= IoU < 0.5
      - both_error: wrong class, 0.1 <= IoU < 0.5
      - duplicate: correct but already matched
      - background_fp: no GT match at all (IoU < 0.1)
      - missed: GT with no matching detection
    """
    preds = np.array(all_preds) if all_preds else np.zeros((0, 7))
    labels = np.array(all_labels) if all_labels else np.zeros((0, 6))

    if len(preds) == 0 or len(labels) == 0:
        return {'correct': 0, 'cls_error': 0, 'loc_error': 0,
                'both_error': 0, 'duplicate': 0, 'background_fp': 0,
                'missed': len(labels)}

    preds_conf = preds[:, 5]
    mask = preds_conf >= conf_thres
    preds = preds[mask]

    results = {
        'correct': 0, 'cls_error': 0, 'loc_error': 0,
        'both_error': 0, 'duplicate': 0, 'background_fp': 0, 'missed': 0
    }

    img_ids = np.unique(np.concatenate([preds[:, 0], labels[:, 0]])) if len(preds) > 0 else np.unique(labels[:, 0])

    for img_id in img_ids:
        p_mask = preds[:, 0] == img_id
        l_mask = labels[:, 0] == img_id
        img_preds = preds[p_mask]   # (M, 7) img_idx, x1,y1,x2,y2, conf, cls
        img_labels = labels[l_mask]  # (N, 6) img_idx, cls, x,y,w,h

        if len(img_labels) == 0:
            results['background_fp'] += len(img_preds)
            continue

        if len(img_preds) == 0:
            results['missed'] += len(img_labels)
            continue

        pred_boxes = torch.tensor(img_preds[:, 1:5], dtype=torch.float32)
        gt_xywh = torch.tensor(img_labels[:, 2:6], dtype=torch.float32)
        gt_boxes = xywh2xyxy(gt_xywh)

        iou = box_iou(gt_boxes, pred_boxes).numpy()

        pred_cls = img_preds[:, 6].astype(int)
        gt_cls = img_labels[:, 1].astype(int)

        gt_matched = set()
        pred_sorted = np.argsort(-img_preds[:, 5])

        for pi in pred_sorted:
            best_iou_idx = np.argmax(iou[:, pi])
            best_iou = iou[best_iou_idx, pi]

            if best_iou >= iou_thres_correct:
                if pred_cls[pi] == gt_cls[best_iou_idx]:
                    if best_iou_idx not in gt_matched:
                        results['correct'] += 1
                        gt_matched.add(best_iou_idx)
                    else:
                        results['duplicate'] += 1
                else:
                    results['cls_error'] += 1
            elif best_iou >= iou_thres_loose:
                if pred_cls[pi] == gt_cls[best_iou_idx]:
                    results['loc_error'] += 1
                else:
                    results['both_error'] += 1
            else:
                results['background_fp'] += 1

        for gi in range(len(img_labels)):
            if gi not in gt_matched:
                results['missed'] += 1

    return results


def plot_error_decomposition(errors_dict, save_dir):
    """Stacked bar chart of error types per model."""
    if not errors_dict:
        return

    categories = ['correct', 'cls_error', 'loc_error', 'both_error',
                   'duplicate', 'background_fp', 'missed']
    labels_nice = ['Correct', 'Cls Error', 'Loc Error', 'Cls+Loc', 'Duplicate', 'BG FP', 'Missed']
    colors = ['#4CAF50', '#F44336', '#FF9800', '#9C27B0', '#795548', '#607D8B', '#E91E63']

    models = list(errors_dict.keys())
    n = len(models)

    fig, ax = plt.subplots(figsize=(max(8, 3 * n), 6))
    x = np.arange(n)
    width = 0.6

    bottoms = np.zeros(n)
    for ci, (cat, label, color) in enumerate(zip(categories, labels_nice, colors)):
        vals = [errors_dict[m].get(cat, 0) for m in models]
        ax.bar(x, vals, width, bottom=bottoms, label=label, color=color)
        bottoms += np.array(vals, dtype=float)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('Count')
    ax.set_title('Error Type Decomposition (conf >= 0.25)')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    fig.savefig(save_dir / 'error_breakdown.png', dpi=200)
    plt.close(fig)
    print(f'  Saved error_breakdown.png')


# ---------------------------------------------------------------------------
# 4. Layer-wise bottleneck profiling
# ---------------------------------------------------------------------------

def layer_bottleneck_profile(model, imgsz, device, is_snn=False):
    """Measure per-layer FLOPs, latency, and params."""
    if is_snn:
        from spikingjelly.activation_based.functional import reset_net
        reset_net(model)

    layer_info = []

    for idx, m in enumerate(model.model):
        name = m.__class__.__name__
        n_params = sum(p.numel() for p in m.parameters())
        layer_info.append({
            'idx': idx, 'name': name, 'params': n_params,
            'macs': 0, 'latency_ms': 0,
        })

    mac_hooks = []
    layer_macs = {}

    def make_mac_hook(layer_idx):
        def hook(mod, inp, out):
            if isinstance(mod, nn.Conv2d) and len(inp) > 0 and inp[0].dim() == 4:
                _, _, H, W = inp[0].shape
                Co, Ci, Kh, Kw = mod.weight.shape
                s = mod.stride if isinstance(mod.stride, tuple) else (mod.stride, mod.stride)
                macs = Kh * Kw * Ci * Co * (H // s[0]) * (W // s[1])
                layer_macs[layer_idx] = layer_macs.get(layer_idx, 0) + macs
            elif isinstance(mod, nn.Linear):
                layer_macs[layer_idx] = layer_macs.get(layer_idx, 0) + mod.in_features * mod.out_features
        return hook

    for idx, m in enumerate(model.model):
        for sub in m.modules():
            if isinstance(sub, (nn.Conv2d, nn.Linear)):
                mac_hooks.append(sub.register_forward_hook(make_mac_hook(idx)))

    inp = torch.randn(1, 3, imgsz, imgsz, device=device)
    if is_snn:
        reset_net(model)
    with torch.no_grad():
        model(inp)

    for h in mac_hooks:
        h.remove()

    for idx in range(len(layer_info)):
        layer_info[idx]['macs'] = layer_macs.get(idx, 0)

    # Latency profiling per top-level layer
    n_runs = 20
    for idx, m in enumerate(model.model):
        y_cache = []

        def run_prefix(stop_at):
            """Run model up to layer stop_at, return input for that layer."""
            if is_snn:
                reset_net(model)
            x = inp
            ys = []
            for i, layer in enumerate(model.model):
                if layer.f != -1:
                    x = ys[layer.f] if isinstance(layer.f, int) else \
                        [x if j == -1 else ys[j] for j in layer.f]
                if i == stop_at:
                    return x
                x = layer(x)
                ys.append(x)
            return x

        try:
            layer_input = run_prefix(idx)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_runs):
                with torch.no_grad():
                    m(layer_input if not isinstance(layer_input, list) else layer_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) / n_runs * 1000
            layer_info[idx]['latency_ms'] = elapsed
        except Exception:
            layer_info[idx]['latency_ms'] = 0

    return layer_info


def plot_layer_bottleneck(layer_info, save_dir, model_name='Model'):
    """Stacked bar chart of per-layer FLOPs and latency."""
    if not layer_info:
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    idxs = [l['idx'] for l in layer_info]
    names = [f"{l['idx']}:{l['name'][:12]}" for l in layer_info]
    macs = [l['macs'] * 2 / 1e9 for l in layer_info]  # GFLOPs
    latencies = [l['latency_ms'] for l in layer_info]
    params = [l['params'] / 1e3 for l in layer_info]  # K params

    # Assign colors by block type
    block_colors = []
    for l in layer_info:
        idx = l['idx']
        if idx <= 3:
            block_colors.append('#2196F3')  # backbone = blue
        elif idx <= 16:
            block_colors.append('#FF9800')  # neck = orange
        else:
            block_colors.append('#4CAF50')  # head = green

    ax1.barh(range(len(idxs)), macs, color=block_colors)
    ax1.set_yticks(range(len(idxs)))
    ax1.set_yticklabels(names, fontsize=7)
    ax1.set_xlabel('GFLOPs')
    ax1.set_title(f'{model_name}: FLOPs per Layer')
    ax1.invert_yaxis()

    ax2.barh(range(len(idxs)), latencies, color=block_colors)
    ax2.set_yticks(range(len(idxs)))
    ax2.set_yticklabels(names, fontsize=7)
    ax2.set_xlabel('Latency (ms)')
    ax2.set_title(f'{model_name}: Latency per Layer')
    ax2.invert_yaxis()

    ax3.barh(range(len(idxs)), params, color=block_colors)
    ax3.set_yticks(range(len(idxs)))
    ax3.set_yticklabels(names, fontsize=7)
    ax3.set_xlabel('Parameters (K)')
    ax3.set_title(f'{model_name}: Params per Layer')
    ax3.invert_yaxis()

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2196F3', label='Backbone (0-3)'),
                       Patch(facecolor='#FF9800', label='Neck (4-16)'),
                       Patch(facecolor='#4CAF50', label='Head')]
    ax1.legend(handles=legend_elements, fontsize=8)

    plt.tight_layout()
    fig.savefig(save_dir / 'layer_bottleneck.png', dpi=200)
    plt.close(fig)
    print(f'  Saved layer_bottleneck.png')


# ---------------------------------------------------------------------------
# 5. Per-block spike sparsity
# ---------------------------------------------------------------------------

def block_sparsity_profile(model, imgsz, device):
    """Measure spike sparsity grouped by architectural block."""
    from spikingjelly.activation_based.functional import reset_net
    from spikingjelly.activation_based import neuron

    reset_net(model)

    block_stats = {}
    hooks = []

    for name, mod in model.named_modules():
        if isinstance(mod, neuron.IFNode):
            def make_hook(n):
                def hook_fn(module, inp, output):
                    if isinstance(output, torch.Tensor) and output.numel() > 0:
                        zeros = (output == 0).sum().item()
                        total = output.numel()
                        block_stats[n] = {
                            'zeros': zeros, 'total': total,
                            'sparsity': zeros / total,
                            'shape': list(output.shape),
                        }
                return hook_fn
            hooks.append(mod.register_forward_hook(make_hook(name)))

    inp = torch.randn(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        model(inp)

    for h in hooks:
        h.remove()

    grouped = {'Backbone': [], 'Neck': [], 'Head': []}
    for name, stat in sorted(block_stats.items()):
        parts = name.split('.')
        try:
            layer_idx = int(parts[1]) if len(parts) > 1 and parts[0] == 'model' else -1
        except (ValueError, IndexError):
            layer_idx = -1

        if layer_idx <= 3:
            grouped['Backbone'].append(stat)
        elif layer_idx <= 16:
            grouped['Neck'].append(stat)
        else:
            grouped['Head'].append(stat)

    summary = {}
    for group, stats_list in grouped.items():
        if stats_list:
            total_z = sum(s['zeros'] for s in stats_list)
            total_e = sum(s['total'] for s in stats_list)
            summary[group] = {
                'sparsity': total_z / total_e if total_e > 0 else 0,
                'n_layers': len(stats_list),
                'total_elements': total_e,
            }

    return block_stats, summary


def plot_sparsity_map(block_stats, summary, save_dir, model_name='Model'):
    """Bar chart of per-IFNode sparsity, colored by block."""
    if not block_stats:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Per-layer detail
    names = []
    sparsities = []
    colors = []
    for name in sorted(block_stats.keys()):
        short = name.replace('model.', '').replace('.act', '')
        names.append(short[:20])
        sparsities.append(block_stats[name]['sparsity'] * 100)
        parts = name.split('.')
        try:
            idx = int(parts[1]) if len(parts) > 1 and parts[0] == 'model' else -1
        except (ValueError, IndexError):
            idx = -1
        if idx <= 3:
            colors.append('#2196F3')
        elif idx <= 16:
            colors.append('#FF9800')
        else:
            colors.append('#4CAF50')

    ax1.barh(range(len(names)), sparsities, color=colors)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=6)
    ax1.set_xlabel('Sparsity (%)')
    ax1.set_title(f'{model_name}: Per-IFNode Spike Sparsity')
    ax1.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax1.invert_yaxis()
    ax1.legend(fontsize=8)

    # Block-level summary
    groups = list(summary.keys())
    group_sparsity = [summary[g]['sparsity'] * 100 for g in groups]
    group_colors = ['#2196F3', '#FF9800', '#4CAF50'][:len(groups)]

    bars = ax2.bar(groups, group_sparsity, color=group_colors)
    ax2.set_ylabel('Sparsity (%)')
    ax2.set_title(f'{model_name}: Block-Level Sparsity')
    ax2.set_ylim(0, 100)
    for bar, val in zip(bars, group_sparsity):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    fig.savefig(save_dir / 'spike_sparsity_map.png', dpi=200)
    plt.close(fig)
    print(f'  Saved spike_sparsity_map.png')


# ---------------------------------------------------------------------------
# 6. Confidence calibration
# ---------------------------------------------------------------------------

def confidence_calibration(all_preds, all_labels, nc=3, n_bins=10):
    """Bin predictions by confidence and compute actual precision per bin."""
    preds = np.array(all_preds) if all_preds else np.zeros((0, 7))
    labels = np.array(all_labels) if all_labels else np.zeros((0, 6))

    if len(preds) == 0:
        return None

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)

    for bi in range(n_bins):
        lo, hi = bins[bi], bins[bi + 1]
        mask = (preds[:, 5] >= lo) & (preds[:, 5] < hi)
        if not mask.any():
            continue

        bin_preds = preds[mask]
        bin_count[bi] = len(bin_preds)
        bin_conf[bi] = bin_preds[:, 5].mean()

        n_correct = 0
        for p in bin_preds:
            img_id, x1, y1, x2, y2, conf, cls = p
            img_labels = labels[labels[:, 0] == img_id]
            if len(img_labels) == 0:
                continue
            gt_xywh = torch.tensor(img_labels[:, 2:6], dtype=torch.float32)
            gt_boxes = xywh2xyxy(gt_xywh)
            pred_box = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
            ious = box_iou(gt_boxes, pred_box).numpy().flatten()
            gt_cls = img_labels[:, 1]
            for gi in range(len(img_labels)):
                if ious[gi] >= 0.5 and int(gt_cls[gi]) == int(cls):
                    n_correct += 1
                    break

        bin_acc[bi] = n_correct / len(bin_preds) if len(bin_preds) > 0 else 0

    ece = np.sum(bin_count * np.abs(bin_acc - bin_conf)) / max(bin_count.sum(), 1)

    return {
        'bin_centers': bin_centers.tolist(),
        'bin_accuracy': bin_acc.tolist(),
        'bin_confidence': bin_conf.tolist(),
        'bin_count': bin_count.tolist(),
        'ECE': float(ece),
    }


def plot_calibration(calib_dict, save_dir):
    """Reliability diagram for each model."""
    if not calib_dict:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for model_name, cal in calib_dict.items():
        if cal is None:
            continue
        centers = cal['bin_centers']
        acc = cal['bin_accuracy']
        ax1.plot(centers, acc, 'o-', label=f'{model_name} (ECE={cal["ECE"]:.3f})', linewidth=2)

    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax1.set_xlabel('Mean Predicted Confidence')
    ax1.set_ylabel('Fraction of Positives (Accuracy)')
    ax1.set_title('Reliability Diagram')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=8)

    # Confidence histogram
    for model_name, cal in calib_dict.items():
        if cal is None:
            continue
        ax2.bar(cal['bin_centers'], cal['bin_count'],
                width=0.08, alpha=0.5, label=model_name)

    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Prediction Confidence Distribution')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(save_dir / 'calibration_curve.png', dpi=200)
    plt.close(fig)
    print(f'  Saved calibration_curve.png')


# ---------------------------------------------------------------------------
# 7. KD gain decomposition
# ---------------------------------------------------------------------------

def plot_kd_gain(baseline_metrics, kd_metrics, save_dir):
    """Per-class delta chart showing where KD helped."""
    if baseline_metrics is None or kd_metrics is None:
        return

    classes = baseline_metrics['classes']
    n = len(classes)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax_idx, (metric, title) in enumerate([('AP50', 'AP@0.5'), ('AP50-95', 'AP@0.5:0.95'), ('F1', 'F1')]):
        ax = axes[ax_idx]
        baseline_vals = np.array(baseline_metrics[metric])
        kd_vals = np.array(kd_metrics[metric])
        delta = kd_vals - baseline_vals

        colors = ['#4CAF50' if d >= 0 else '#F44336' for d in delta]
        bars = ax.bar(classes, delta, color=colors)

        for bar, d in zip(bars, delta):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002 * (1 if d >= 0 else -1),
                    f'{d:+.3f}', ha='center', va='bottom' if d >= 0 else 'top',
                    fontsize=9, fontweight='bold')

        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_ylabel(f'{title} Delta')
        ax.set_title(f'KD Gain: {title}')

    plt.tight_layout()
    fig.savefig(save_dir / 'kd_gain_per_class.png', dpi=200)
    plt.close(fig)
    print(f'  Saved kd_gain_per_class.png')


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_report(save_dir, per_class_results, error_results, layer_results,
                 sparsity_summaries, calib_results, scale_results):
    """Write a comprehensive text report."""
    report = []
    report.append('=' * 80)
    report.append('  DEEP MODEL DIAGNOSTICS REPORT')
    report.append('=' * 80)

    # Per-class
    report.append('\n--- 1. Per-Class Performance ---')
    for model_name, metrics in per_class_results.items():
        if metrics is None:
            continue
        report.append(f'\n  {model_name}:')
        report.append(f'    {"Class":<10s} {"P":>8s} {"R":>8s} {"AP50":>8s} {"AP95":>8s} {"F1":>8s}')
        for i, cls in enumerate(metrics['classes']):
            report.append(f'    {cls:<10s} {metrics["P"][i]:>8.3f} {metrics["R"][i]:>8.3f} '
                          f'{metrics["AP50"][i]:>8.3f} {metrics["AP50-95"][i]:>8.3f} {metrics["F1"][i]:>8.3f}')

    # Error decomposition
    report.append('\n--- 3. Error Decomposition ---')
    for model_name, errors in error_results.items():
        total = sum(errors.values())
        report.append(f'\n  {model_name} (total events: {total}):')
        for cat, count in errors.items():
            pct = count / total * 100 if total > 0 else 0
            report.append(f'    {cat:<20s}: {count:>6d} ({pct:>5.1f}%)')

    # Layer bottleneck
    report.append('\n--- 4. Layer Bottleneck ---')
    for model_name, layers in layer_results.items():
        total_flops = sum(l['macs'] * 2 for l in layers)
        total_lat = sum(l['latency_ms'] for l in layers)
        report.append(f'\n  {model_name}:')
        report.append(f'    {"Idx":<4s} {"Layer":<25s} {"GFLOPs":>8s} {"%FLOPs":>8s} '
                      f'{"ms":>8s} {"%Lat":>8s} {"Params":>10s}')
        for l in layers:
            gf = l['macs'] * 2 / 1e9
            pct_f = l['macs'] * 2 / total_flops * 100 if total_flops > 0 else 0
            pct_l = l['latency_ms'] / total_lat * 100 if total_lat > 0 else 0
            report.append(f'    {l["idx"]:<4d} {l["name"]:<25s} {gf:>8.3f} {pct_f:>7.1f}% '
                          f'{l["latency_ms"]:>8.3f} {pct_l:>7.1f}% {l["params"]:>10,}')

    # Sparsity
    report.append('\n--- 5. Spike Sparsity (Block-Level) ---')
    for model_name, summary in sparsity_summaries.items():
        report.append(f'\n  {model_name}:')
        for group, data in summary.items():
            report.append(f'    {group:<12s}: {data["sparsity"]:.1%} '
                          f'({data["n_layers"]} IFNodes, {data["total_elements"]:,} elements)')

    # Calibration
    report.append('\n--- 6. Confidence Calibration ---')
    for model_name, cal in calib_results.items():
        if cal is None:
            continue
        report.append(f'\n  {model_name}:')
        report.append(f'    ECE (Expected Calibration Error): {cal["ECE"]:.4f}')
        report.append(f'    {"Bin":>8s} {"Conf":>8s} {"Acc":>8s} {"Count":>8s}')
        for i in range(len(cal['bin_centers'])):
            report.append(f'    {cal["bin_centers"][i]:>8.2f} {cal["bin_confidence"][i]:>8.3f} '
                          f'{cal["bin_accuracy"][i]:>8.3f} {cal["bin_count"][i]:>8.0f}')

    # Object size distribution
    if scale_results:
        report.append('\n--- 2. Object Size Distribution ---')
        for model_name, (result, scale_labels) in scale_results.items():
            report.append(f'\n  {model_name}:')
            for s in scale_labels:
                data = result[s]
                report.append(f'    {s:<25s}: {data["n_gt"]:>6d} GT objects')

    report.append('\n' + '=' * 80)
    report.append('  END OF REPORT')
    report.append('=' * 80)

    text = '\n'.join(report)
    (save_dir / 'diagnostics_report.txt').write_text(text)
    print(f'\n  Saved diagnostics_report.txt')
    print(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Deep model diagnostics')
    parser.add_argument('--weights', type=str, required=True, help='Primary model weights (Student+KD)')
    parser.add_argument('--baseline', type=str, default='', help='Student baseline weights')
    parser.add_argument('--teacher', type=str, default='', help='Teacher model weights')
    parser.add_argument('--data', type=str, default='data/hazydet.yaml', help='Dataset yaml')
    parser.add_argument('--img', type=int, default=1920, help='Image size')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--device', default='0', help='CUDA device')
    parser.add_argument('--save-dir', type=str, default='runs/diagnostics/', help='Output directory')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='Confidence threshold for collection')
    parser.add_argument('--skip-bottleneck', action='store_true', help='Skip slow layer-wise profiling')
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)

    print(f'Device: {device}')
    print(f'Image size: {args.img}')
    print(f'Save dir: {save_dir}')

    dataloader, data_dict = get_dataloader(args.data, args.img, args.batch_size)
    nc = int(data_dict.get('nc', 3))

    # Define models to evaluate
    model_configs = []
    if args.teacher:
        model_configs.append(('YOLOv9-C (Teacher)', args.teacher, False))
    if args.baseline:
        model_configs.append(('SU-YOLO (Baseline)', args.baseline, True))
    model_configs.append(('SU-YOLO + KD', args.weights, True))

    per_class_results = {}
    error_results = {}
    layer_results = {}
    sparsity_summaries = {}
    calib_results = {}
    scale_results = {}
    all_model_preds = {}
    all_model_labels = {}

    for model_name, weights_path, is_snn in model_configs:
        print(f'\n{"=" * 70}')
        print(f'  Processing: {model_name}')
        print(f'  Weights: {weights_path}')
        print(f'{"=" * 70}')

        model = load_model(weights_path, device, is_snn=is_snn)

        # --- 1. Per-class metrics ---
        print(f'\n  [1/7] Collecting predictions...')
        stats, preds, labels = collect_predictions(
            model, dataloader, device, conf_thres=args.conf_thres, is_snn=is_snn)
        metrics = per_class_metrics(stats, nc)
        per_class_results[model_name] = metrics
        all_model_preds[model_name] = preds
        all_model_labels[model_name] = labels

        if metrics:
            print(f'    Per-class AP50: {dict(zip(metrics["classes"], [f"{v:.3f}" for v in metrics["AP50"]]))}')

        # --- 2. Per-scale analysis ---
        print(f'  [2/7] Per-scale object size analysis...')
        scale_result = per_scale_analysis(preds, labels, model, args.img, nc)
        if scale_result:
            scale_results[model_name] = scale_result
            for s in scale_result[1]:
                print(f'    {s}: {scale_result[0][s]["n_gt"]} GT objects')

        # --- 3. Error decomposition ---
        print(f'  [3/7] Error decomposition...')
        errors = error_decomposition(preds, labels, nc)
        error_results[model_name] = errors
        total_events = sum(errors.values())
        for cat, count in errors.items():
            pct = count / total_events * 100 if total_events > 0 else 0
            print(f'    {cat:<20s}: {count:>6d} ({pct:.1f}%)')

        # --- 4. Layer bottleneck ---
        if not args.skip_bottleneck:
            print(f'  [4/7] Layer bottleneck profiling...')
            layers = layer_bottleneck_profile(model, args.img, device, is_snn=is_snn)
            layer_results[model_name] = layers
            total_gflops = sum(l['macs'] * 2 / 1e9 for l in layers)
            total_lat = sum(l['latency_ms'] for l in layers)
            print(f'    Total GFLOPs: {total_gflops:.2f}, Total latency: {total_lat:.2f} ms')
        else:
            print(f'  [4/7] Skipped (--skip-bottleneck)')

        # --- 5. Spike sparsity ---
        if is_snn:
            print(f'  [5/7] Spike sparsity profiling...')
            block_stats, summary = block_sparsity_profile(model, args.img, device)
            sparsity_summaries[model_name] = summary
            for group, data in summary.items():
                print(f'    {group}: {data["sparsity"]:.1%}')
            plot_sparsity_map(block_stats, summary, save_dir, model_name)
        else:
            print(f'  [5/7] Skipped (ANN model)')

        # --- 6. Confidence calibration ---
        print(f'  [6/7] Confidence calibration...')
        cal = confidence_calibration(preds, labels, nc)
        calib_results[model_name] = cal
        if cal:
            print(f'    ECE: {cal["ECE"]:.4f}')

        # --- 7. KD gain (computed after all models processed) ---
        print(f'  [7/7] KD gain (computed after all models)')

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # --- Generate comparison plots ---
    print(f'\n{"=" * 70}')
    print(f'  Generating comparison plots...')
    print(f'{"=" * 70}')

    plot_per_class_comparison(per_class_results, save_dir)
    plot_error_decomposition(error_results, save_dir)
    plot_calibration(calib_results, save_dir)

    if scale_results:
        first_key = list(scale_results.keys())[0]
        plot_per_scale({first_key: scale_results[first_key][0]}, save_dir)

    if layer_results:
        for model_name, layers in layer_results.items():
            plot_layer_bottleneck(layers, save_dir, model_name)
            break  # plot the first one (primary model)

    # KD gain: compare baseline vs KD
    baseline_key = next((k for k in per_class_results if 'Baseline' in k), None)
    kd_key = next((k for k in per_class_results if 'KD' in k), None)
    if baseline_key and kd_key:
        plot_kd_gain(per_class_results[baseline_key], per_class_results[kd_key], save_dir)

    # --- Write report ---
    write_report(save_dir, per_class_results, error_results, layer_results,
                 sparsity_summaries, calib_results, scale_results)

    print(f'\nAll outputs saved to: {save_dir}')
    print('Done.')


if __name__ == '__main__':
    main()
