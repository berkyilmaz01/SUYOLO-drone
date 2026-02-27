"""Comprehensive model evaluation: FLOPs, SyOPs, sparsity, energy, size, accuracy.

Compares Teacher (GELAN-C), Student Baseline, and Student+KD side by side
with publication-ready output tables.

Usage:
    python tools/evaluate_full.py \
        --teacher runs/train/hazydet-teacher-gelanc5/weights/best.pt \
        --baseline runs/train/hazydet-student-kd-scratch/weights/best.pt \
        --student-kd runs/train/hazydet-student-kd-fixed3/weights/best.pt \
        --data data/hazydet.yaml --img 1920 --device 0

    # Skip accuracy eval (just compute FLOPs/SyOPs/energy):
    python tools/evaluate_full.py \
        --teacher runs/train/hazydet-teacher-gelanc5/weights/best.pt \
        --student-kd runs/train/hazydet-student-kd-fixed3/weights/best.pt \
        --img 1920 --device 0 --skip-accuracy
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import numpy as np

ANN_MAC_PJ = 4.6    # pJ per multiply-accumulate at 45nm CMOS
SNN_AC_PJ = 0.9     # pJ per accumulate (addition only) at 45nm CMOS
DRAM_ACCESS_PJ = 640 # pJ per DRAM access at 45nm (for context)


def load_model(weights_path, device, is_snn=False):
    """Load a model from checkpoint, preferring EMA weights."""
    if is_snn:
        from models.spike import set_time_step
        set_time_step(1)

    ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
    model = (ckpt.get('ema') or ckpt['model']).float()

    if not hasattr(model, 'stride'):
        model.stride = torch.tensor([32.])
    if hasattr(model, 'names') and isinstance(model.names, (list, tuple)):
        model.names = dict(enumerate(model.names))

    model = model.to(device).eval()
    return model


def count_parameters(model):
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_flops(model, imgsz, device, is_snn=False):
    """Count FLOPs via forward hooks on Conv2d and Linear layers.
    Returns total MACs (multiply-accumulate count)."""
    if is_snn:
        from spikingjelly.activation_based.functional import reset_net
        reset_net(model)

    total_macs = [0]
    per_layer_macs = {}
    hooks = []

    def make_hook(name):
        def hook_fn(m, inp, out):
            if isinstance(m, nn.Conv2d) and len(inp) > 0 and inp[0].dim() == 4:
                _, _, H, W = inp[0].shape
                Cout, Cin_per_g, Kh, Kw = m.weight.shape
                s = m.stride if isinstance(m.stride, tuple) else (m.stride, m.stride)
                Ho = H // s[0]
                Wo = W // s[1]
                macs = Kh * Kw * Cin_per_g * Cout * Ho * Wo
                total_macs[0] += macs
                per_layer_macs[name] = per_layer_macs.get(name, 0) + macs
            elif isinstance(m, nn.Linear):
                macs = m.in_features * m.out_features
                total_macs[0] += macs
                per_layer_macs[name] = per_layer_macs.get(name, 0) + macs
        return hook_fn

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(make_hook(name)))

    inp = torch.randn(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        model(inp)

    for h in hooks:
        h.remove()

    return total_macs[0], per_layer_macs


def measure_spike_sparsity(model, imgsz, device):
    """Hook every IFNode to measure spike firing rates.
    Returns overall sparsity and per-layer breakdown."""
    from spikingjelly.activation_based.functional import reset_net
    from spikingjelly.activation_based import neuron

    reset_net(model)

    spike_stats = {}
    hooks = []

    for name, mod in model.named_modules():
        if isinstance(mod, neuron.IFNode):
            def make_hook(n):
                def hook_fn(module, inp, output):
                    if isinstance(output, torch.Tensor) and output.numel() > 0:
                        zeros = (output == 0).sum().item()
                        total = output.numel()
                        spike_stats[n] = {
                            'zeros': zeros,
                            'total': total,
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

    if not spike_stats:
        return 0.0, {}

    total_zeros = sum(s['zeros'] for s in spike_stats.values())
    total_elements = sum(s['total'] for s in spike_stats.values())
    overall_sparsity = total_zeros / total_elements if total_elements > 0 else 0.0

    return overall_sparsity, spike_stats


def measure_latency(model, imgsz, device, is_snn=False, n_warmup=10, n_runs=50):
    """Measure inference latency at batch=1."""
    if is_snn:
        from spikingjelly.activation_based.functional import reset_net

    inp = torch.randn(1, 3, imgsz, imgsz, device=device)

    for _ in range(n_warmup):
        if is_snn:
            reset_net(model)
        with torch.no_grad():
            model(inp)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        if is_snn:
            reset_net(model)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(inp)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times = sorted(times)
    return {
        'mean_ms': np.mean(times) * 1000,
        'median_ms': np.median(times) * 1000,
        'min_ms': min(times) * 1000,
        'max_ms': max(times) * 1000,
        'fps': 1.0 / np.mean(times),
    }


def compute_energy(total_macs, sparsity, is_snn):
    """Estimate per-frame energy on 45nm CMOS.

    ANN: all MACs are full multiply-accumulate (4.6 pJ each).
    SNN: only (1 - sparsity) fraction of operations execute,
         and those are accumulate-only (0.9 pJ each) since
         inputs are binary spikes.
    """
    if is_snn:
        effective_ops = total_macs * (1 - sparsity)
        energy_pj = effective_ops * SNN_AC_PJ
    else:
        energy_pj = total_macs * ANN_MAC_PJ
    energy_mj = energy_pj / 1e9
    return energy_mj, effective_ops if is_snn else total_macs


def run_accuracy(weights, data, imgsz, time_step, device_str, batch_size=4):
    """Run val.py and return accuracy metrics."""
    try:
        import val as validate
        from utils.general import check_dataset, check_yaml

        data_dict = check_dataset(check_yaml(data))
        device = torch.device(f'cuda:{device_str}' if device_str.isdigit() else device_str)

        results, maps, _ = validate.run(
            data=data_dict,
            time_step=time_step,
            weights=[weights],
            batch_size=batch_size,
            imgsz=imgsz,
            conf_thres=0.001,
            iou_thres=0.5,
            device=device_str,
            single_cls=False,
            plots=False,
            verbose=True,
        )
        return {
            'P': results[0],
            'R': results[1],
            'mAP50': results[2],
            'mAP50-95': results[3],
        }
    except Exception as e:
        print(f'  WARNING: accuracy eval failed: {e}')
        return None


def evaluate_model(name, weights, imgsz, device, is_snn=False,
                   data=None, skip_accuracy=False, accuracy_override=None):
    """Full evaluation of a single model."""
    print(f'\n{"=" * 70}')
    print(f'  Evaluating: {name}')
    print(f'  Weights: {weights}')
    print(f'{"=" * 70}')

    model = load_model(weights, device, is_snn=is_snn)
    total_params, trainable_params = count_parameters(model)

    print(f'  Parameters: {total_params:,} ({total_params / 1e6:.2f}M)')
    print(f'  FP32 size:  {total_params * 4 / 1024 / 1024:.2f} MB')
    print(f'  INT8 size:  {total_params / 1024 / 1024:.2f} MB')

    # FLOPs
    total_macs, per_layer = count_flops(model, imgsz, device, is_snn=is_snn)
    gflops = total_macs * 2 / 1e9
    print(f'  GFLOPs:     {gflops:.2f}')

    # Spike sparsity (SNN only)
    sparsity = 0.0
    spike_details = {}
    if is_snn:
        sparsity, spike_details = measure_spike_sparsity(model, imgsz, device)
        print(f'  Spike sparsity: {sparsity:.1%}')
        gsyops = gflops * (1 - sparsity)
        print(f'  GSyOPs (effective): {gsyops:.2f}  (GFLOPs x {1 - sparsity:.2f})')
    else:
        gsyops = gflops

    # Energy
    energy_mj, effective_ops = compute_energy(total_macs, sparsity, is_snn)
    print(f'  Energy/frame (est.): {energy_mj:.4f} mJ')
    if is_snn:
        print(f'  Energy per op: {SNN_AC_PJ * (1 - sparsity):.2f} pJ (sparse AC)')
    else:
        print(f'  Energy per op: {ANN_MAC_PJ} pJ (full MAC)')

    # Latency
    latency = measure_latency(model, imgsz, device, is_snn=is_snn)
    print(f'  GPU latency (bs=1): {latency["mean_ms"]:.2f} ms ({latency["fps"]:.1f} FPS)')

    # Accuracy
    accuracy = accuracy_override
    if not skip_accuracy and accuracy is None and data:
        device_str = str(device).replace('cuda:', '')
        accuracy = run_accuracy(weights, data, imgsz,
                                time_step=1 if is_snn else 4,
                                device_str=device_str)

    if accuracy:
        print(f'  mAP50:      {accuracy["mAP50"]:.3f}')
        print(f'  mAP50-95:   {accuracy["mAP50-95"]:.3f}')
        print(f'  Precision:  {accuracy["P"]:.3f}')
        print(f'  Recall:     {accuracy["R"]:.3f}')

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return {
        'name': name,
        'params': total_params,
        'params_m': total_params / 1e6,
        'fp32_mb': total_params * 4 / 1024 / 1024,
        'int8_mb': total_params / 1024 / 1024,
        'gflops': gflops,
        'gsyops': gsyops,
        'sparsity': sparsity,
        'energy_mj': energy_mj,
        'latency': latency,
        'accuracy': accuracy,
        'is_snn': is_snn,
        'spike_details': spike_details,
    }


def print_comparison_table(results):
    """Print side-by-side comparison table."""
    sep = '=' * 90
    print(f'\n\n{sep}')
    print('  COMPREHENSIVE MODEL COMPARISON')
    print(sep)

    cols = results
    n = len(cols)
    header = f'{"Metric":<25s}'
    for r in cols:
        header += f' | {r["name"]:>18s}'
    print(header)
    print('-' * (25 + 21 * n))

    def row(label, values, fmt='{:>18s}'):
        line = f'{label:<25s}'
        for v in values:
            line += f' | {fmt.format(v)}'
        print(line)

    row('Parameters', [f'{r["params"]:,}' for r in cols])
    row('Parameters (M)', [f'{r["params_m"]:.2f}M' for r in cols])
    row('FP32 Size', [f'{r["fp32_mb"]:.2f} MB' for r in cols])
    row('INT8 Size', [f'{r["int8_mb"]:.2f} MB' if r["is_snn"] else 'N/A' for r in cols])
    row('GFLOPs', [f'{r["gflops"]:.2f}' for r in cols])
    row('GSyOPs (effective)', [f'{r["gsyops"]:.2f}' if r["is_snn"] else 'N/A (=GFLOPs)' for r in cols])
    row('Spike Sparsity', [f'{r["sparsity"]:.1%}' if r["is_snn"] else 'N/A (ANN)' for r in cols])
    row('Energy/frame (est.)', [f'{r["energy_mj"]:.4f} mJ' for r in cols])
    row('GPU Latency (bs=1)', [f'{r["latency"]["mean_ms"]:.1f} ms' for r in cols])
    row('GPU FPS (bs=1)', [f'{r["latency"]["fps"]:.1f}' for r in cols])

    if any(r['accuracy'] for r in cols):
        print('-' * (25 + 21 * n))
        row('mAP50', [f'{r["accuracy"]["mAP50"]:.3f}' if r['accuracy'] else '—' for r in cols])
        row('mAP50-95', [f'{r["accuracy"]["mAP50-95"]:.3f}' if r['accuracy'] else '—' for r in cols])
        row('Precision', [f'{r["accuracy"]["P"]:.3f}' if r['accuracy'] else '—' for r in cols])
        row('Recall', [f'{r["accuracy"]["R"]:.3f}' if r['accuracy'] else '—' for r in cols])

    print(sep)

    # Efficiency ratios
    teacher = next((r for r in cols if not r['is_snn']), None)
    kd = next((r for r in cols if r['is_snn'] and 'KD' in r['name']), None)
    baseline = next((r for r in cols if r['is_snn'] and 'KD' not in r['name']), None)

    if teacher and kd:
        print(f'\n  --- Efficiency Ratios (Student+KD vs Teacher) ---')
        print(f'  Parameter reduction:  {teacher["params"] / kd["params"]:.0f}x smaller')
        print(f'  Model size reduction: {teacher["fp32_mb"] / kd["int8_mb"]:.0f}x (FP32 vs INT8)')
        print(f'  GFLOPs reduction:     {teacher["gflops"] / kd["gflops"]:.1f}x')
        print(f'  Effective ops (SyOPs): {teacher["gflops"] / kd["gsyops"]:.1f}x')
        print(f'  Energy reduction:     {teacher["energy_mj"] / kd["energy_mj"]:.1f}x')
        if kd['accuracy'] and teacher['accuracy']:
            gap = teacher['accuracy']['mAP50-95'] - kd['accuracy']['mAP50-95']
            print(f'  Accuracy gap:         {gap:.3f} mAP50-95 ({gap / teacher["accuracy"]["mAP50-95"] * 100:.1f}% relative)')

    if baseline and kd:
        print(f'\n  --- KD Improvement (Student+KD vs Baseline) ---')
        if kd['accuracy'] and baseline['accuracy']:
            d50 = kd['accuracy']['mAP50'] - baseline['accuracy']['mAP50']
            d95 = kd['accuracy']['mAP50-95'] - baseline['accuracy']['mAP50-95']
            print(f'  mAP50 gain:    +{d50:.3f} ({d50 / baseline["accuracy"]["mAP50"] * 100:.1f}% relative)')
            print(f'  mAP50-95 gain: +{d95:.3f} ({d95 / baseline["accuracy"]["mAP50-95"] * 100:.1f}% relative)')

    print()


def print_sparsity_breakdown(results):
    """Print per-layer spike sparsity for SNN models."""
    for r in results:
        if not r['is_snn'] or not r['spike_details']:
            continue
        print(f'\n  --- Spike Sparsity Breakdown: {r["name"]} ---')
        print(f'  {"Layer":<55s} {"Sparsity":>10s} {"Shape":>20s}')
        print(f'  {"-" * 85}')
        for name, stats in sorted(r['spike_details'].items()):
            print(f'  {name:<55s} {stats["sparsity"]:>9.1%} {str(stats["shape"]):>20s}')
        print(f'  {"-" * 85}')
        print(f'  {"Overall":<55s} {r["sparsity"]:>9.1%}')


def print_latex_table(results):
    """Print a LaTeX-ready table for paper inclusion."""
    print('\n  --- LaTeX Table ---')
    print('  \\begin{table}[h]')
    print('  \\centering')
    print('  \\caption{Model comparison on HazyDet.}')
    print('  \\label{tab:model_comparison}')
    print('  \\begin{tabular}{l' + 'r' * len(results) + '}')
    print('  \\toprule')

    header = '  Metric'
    for r in results:
        header += f' & {r["name"]}'
    print(header + ' \\\\')
    print('  \\midrule')

    print(f'  Parameters' + ''.join(f' & {r["params_m"]:.2f}M' for r in results) + ' \\\\')
    print(f'  GFLOPs' + ''.join(f' & {r["gflops"]:.2f}' for r in results) + ' \\\\')
    print(f'  GSyOPs' + ''.join(f' & {r["gsyops"]:.2f}' if r["is_snn"] else f' & {r["gflops"]:.2f}' for r in results) + ' \\\\')
    print(f'  Sparsity' + ''.join(f' & {r["sparsity"]*100:.1f}\\%' if r["is_snn"] else ' & --' for r in results) + ' \\\\')
    print(f'  Energy (mJ)' + ''.join(f' & {r["energy_mj"]:.4f}' for r in results) + ' \\\\')

    if any(r['accuracy'] for r in results):
        print('  \\midrule')
        print(f'  mAP@50' + ''.join(f' & {r["accuracy"]["mAP50"]:.3f}' if r["accuracy"] else ' & --' for r in results) + ' \\\\')
        print(f'  mAP@50:95' + ''.join(f' & {r["accuracy"]["mAP50-95"]:.3f}' if r["accuracy"] else ' & --' for r in results) + ' \\\\')

    print('  \\bottomrule')
    print('  \\end{tabular}')
    print('  \\end{table}')


def main():
    parser = argparse.ArgumentParser(description='Comprehensive model evaluation')
    parser.add_argument('--teacher', type=str, default='', help='Teacher model weights')
    parser.add_argument('--baseline', type=str, default='', help='Student baseline weights (no KD)')
    parser.add_argument('--student-kd', type=str, default='', help='Student+KD weights')
    parser.add_argument('--data', type=str, default='data/hazydet.yaml', help='Dataset yaml')
    parser.add_argument('--img', type=int, default=1920, help='Image size')
    parser.add_argument('--device', default='0', help='CUDA device')
    parser.add_argument('--skip-accuracy', action='store_true', help='Skip accuracy evaluation')
    parser.add_argument('--latex', action='store_true', help='Print LaTeX table')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if args.device.isdigit() and torch.cuda.is_available()
                          else 'cpu')
    print(f'Device: {device}')
    print(f'Image size: {args.img}x{args.img}')

    # Pre-computed accuracy from val.py runs (used when --skip-accuracy)
    known_accuracy = {
        'teacher': {'P': 0.873, 'R': 0.792, 'mAP50': 0.862, 'mAP50-95': 0.660},
        'baseline': {'P': 0.527, 'R': 0.468, 'mAP50': 0.478, 'mAP50-95': 0.290},
        'student_kd': {'P': 0.591, 'R': 0.491, 'mAP50': 0.515, 'mAP50-95': 0.323},
    }

    all_results = []

    if args.teacher:
        r = evaluate_model(
            'YOLOv9-C (Teacher)', args.teacher, args.img, device,
            is_snn=False, data=args.data, skip_accuracy=args.skip_accuracy,
            accuracy_override=known_accuracy['teacher'] if args.skip_accuracy else None)
        all_results.append(r)

    if args.baseline:
        r = evaluate_model(
            'SU-YOLO Ghost (Baseline)', args.baseline, args.img, device,
            is_snn=True, data=args.data, skip_accuracy=args.skip_accuracy,
            accuracy_override=known_accuracy['baseline'] if args.skip_accuracy else None)
        all_results.append(r)

    if args.student_kd:
        r = evaluate_model(
            'SU-YOLO Ghost + KD', args.student_kd, args.img, device,
            is_snn=True, data=args.data, skip_accuracy=args.skip_accuracy,
            accuracy_override=known_accuracy['student_kd'] if args.skip_accuracy else None)
        all_results.append(r)

    if len(all_results) > 1:
        print_comparison_table(all_results)
        print_sparsity_breakdown(all_results)

        if args.latex:
            print_latex_table(all_results)

    print('\nDone.')


if __name__ == '__main__':
    main()
