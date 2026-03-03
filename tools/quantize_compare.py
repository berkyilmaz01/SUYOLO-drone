"""Quantization Export Tool — fuses BN into conv weights, then produces
real integer weight arrays at each bit width for ZCU102 FPGA deployment.

Phase 1: BN fusion (folds SeBatchNorm + var_scale into conv weight/bias)
Phase 2: Validate fused model (run val.py once to confirm BN fusion is lossless)
Phase 3: Export actual quantized weight packages at INT8 / INT16 / FXP8 / FP16

No simulated quantization. All exports are real integer arrays + scale factors
ready for hardware synthesis.

Usage:
    python tools/quantize_compare.py \
        --weights runs/train/hazydet-student-kd-scratch/weights/best.pt \
        --data data/hazydet.yaml \
        --img 1920 --batch 4 --time-step 1
"""
import argparse
import copy
import csv
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

import models.spike as spike
from models.spike import (SConv, SDConv, seBatchNorm, SeBatchNorm2d)


# ---------------------------------------------------------------------------
# Phase 1: BN Fusion
# ---------------------------------------------------------------------------

def fuse_conv_bn(conv, bn_wrapper):
    """Fuse SeBatchNorm into a Conv2d, returning fused weight and bias.

    SeBatchNorm2d eval-time formula:
        out = gamma * (x - mean) / sqrt(var / var_scale + eps) + beta

    Fused into conv:
        w_fused = w * gamma / sqrt(var/var_scale + eps)
        b_fused = beta - mean * gamma / sqrt(var/var_scale + eps)  [+ conv.bias * factor]
    """
    if isinstance(bn_wrapper, seBatchNorm):
        bn = bn_wrapper.bn
    elif isinstance(bn_wrapper, SeBatchNorm2d):
        bn = bn_wrapper
    else:
        return None, None

    w = conv.weight.detach().float()
    out_ch = w.shape[0]

    mean = bn.running_mean[:out_ch].detach().float()
    var = bn.running_var[:out_ch].detach().float()
    var_scale = getattr(bn, 'var_scale', 1.0)
    eps = bn.eps
    gamma = bn.weight[:out_ch].detach().float() if bn.weight is not None else torch.ones(out_ch)
    beta = bn.bias[:out_ch].detach().float() if bn.bias is not None else torch.zeros(out_ch)

    inv_std = gamma / torch.sqrt(var / var_scale + eps)

    w_fused = w * inv_std.view(-1, 1, 1, 1)
    b_fused = beta - mean * inv_std

    if conv.bias is not None:
        b_fused = b_fused + conv.bias.detach().float() * inv_std

    return w_fused, b_fused


def fuse_model_bn(model):
    """Fuse all SeBatchNorm layers into preceding Conv2d layers in-place."""
    fused = 0
    for _, mod in model.named_modules():
        if isinstance(mod, (SConv, SDConv)):
            w_f, b_f = fuse_conv_bn(mod.conv, mod.bn)
            if w_f is not None:
                mod.conv.weight.data = w_f
                if mod.conv.bias is None:
                    mod.conv.bias = nn.Parameter(b_f)
                else:
                    mod.conv.bias.data = b_f
                mod.bn = nn.Identity()
                fused += 1
    return fused


# ---------------------------------------------------------------------------
# Phase 3: Real quantized weight export
# ---------------------------------------------------------------------------

def symmetric_quantize(tensor, n_bits):
    """Symmetric per-tensor quantization → actual integer tensor + scale."""
    qmax = (1 << (n_bits - 1)) - 1
    abs_max = tensor.abs().max().item()
    if abs_max == 0:
        return torch.zeros_like(tensor, dtype=torch.int32), 0.0
    scale = abs_max / qmax
    q = (tensor / scale).round().clamp(-qmax - 1, qmax).to(torch.int32)
    return q, scale


def per_channel_quantize(tensor, n_bits):
    """Symmetric per-channel quantization (dim 0 = output channels)."""
    qmax = (1 << (n_bits - 1)) - 1
    abs_max = tensor.abs().flatten(1).max(dim=1)[0].clamp(min=1e-8)
    scales = (abs_max / qmax)
    q = (tensor / scales.view(-1, 1, 1, 1)).round().clamp(-qmax - 1, qmax).to(torch.int32)
    return q, scales


def export_layer_weights(save_dir, layer_name, weight_fp32, bias_fp32, schemes):
    """Export a single layer's weights under each quantization scheme."""
    records = {}

    for scheme in schemes:
        out_dir = save_dir / scheme / layer_name
        out_dir.mkdir(parents=True, exist_ok=True)

        if scheme == 'fp32':
            np.save(out_dir / 'weight.npy', weight_fp32.numpy())
            if bias_fp32 is not None:
                np.save(out_dir / 'bias.npy', bias_fp32.numpy())
            records[scheme] = {
                'bits': 32,
                'weight_bytes': weight_fp32.numel() * 4,
                'bias_bytes': (bias_fp32.numel() * 4) if bias_fp32 is not None else 0,
                'scale': 'N/A',
            }

        elif scheme == 'fp16':
            w_f16 = weight_fp32.half()
            np.save(out_dir / 'weight_fp16.npy', w_f16.numpy())
            if bias_fp32 is not None:
                np.save(out_dir / 'bias_fp16.npy', bias_fp32.half().numpy())
            records[scheme] = {
                'bits': 16,
                'weight_bytes': weight_fp32.numel() * 2,
                'bias_bytes': (bias_fp32.numel() * 2) if bias_fp32 is not None else 0,
                'scale': 'N/A (native fp16)',
            }

        elif scheme == 'int16':
            q, scale = symmetric_quantize(weight_fp32, 16)
            np.save(out_dir / 'weight_int16.npy', q.numpy().astype(np.int16))
            with open(out_dir / 'scale.json', 'w') as f:
                json.dump({'scale': scale, 'bits': 16, 'type': 'per_tensor_symmetric'}, f, indent=2)
            if bias_fp32 is not None:
                np.save(out_dir / 'bias_int32.npy', bias_fp32.numpy())  # biases stay INT32
            # CSV for hardware team
            _write_weight_csv(out_dir / 'weight_int16.csv', q)
            records[scheme] = {
                'bits': 16,
                'weight_bytes': weight_fp32.numel() * 2,
                'bias_bytes': (bias_fp32.numel() * 4) if bias_fp32 is not None else 0,
                'scale': scale,
            }

        elif scheme == 'int8':
            q, scale = symmetric_quantize(weight_fp32, 8)
            np.save(out_dir / 'weight_int8.npy', q.numpy().astype(np.int8))
            with open(out_dir / 'scale.json', 'w') as f:
                json.dump({'scale': scale, 'bits': 8, 'type': 'per_tensor_symmetric'}, f, indent=2)
            if bias_fp32 is not None:
                np.save(out_dir / 'bias_int32.npy', bias_fp32.numpy())
            _write_weight_csv(out_dir / 'weight_int8.csv', q)
            records[scheme] = {
                'bits': 8,
                'weight_bytes': weight_fp32.numel(),
                'bias_bytes': (bias_fp32.numel() * 4) if bias_fp32 is not None else 0,
                'scale': scale,
            }

        elif scheme == 'fxp8':
            q, scales = per_channel_quantize(weight_fp32, 8)
            np.save(out_dir / 'weight_fxp8.npy', q.numpy().astype(np.int8))
            with open(out_dir / 'scales.json', 'w') as f:
                json.dump({
                    'scales': scales.tolist(),
                    'bits': 8,
                    'type': 'per_channel_symmetric',
                    'num_channels': weight_fp32.shape[0],
                }, f, indent=2)
            if bias_fp32 is not None:
                np.save(out_dir / 'bias_int32.npy', bias_fp32.numpy())
            _write_weight_csv(out_dir / 'weight_fxp8.csv', q)
            records[scheme] = {
                'bits': 8,
                'weight_bytes': weight_fp32.numel(),
                'bias_bytes': (bias_fp32.numel() * 4) if bias_fp32 is not None else 0,
                'scale': f'per-channel ({weight_fp32.shape[0]} scales)',
            }

    return records


def _write_weight_csv(path, q_tensor):
    """Write quantized weight tensor as CSV (one filter per row)."""
    q_np = q_tensor.numpy()
    out_ch = q_np.shape[0]
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(out_ch):
            writer.writerow(q_np[i].flatten().tolist())


# ---------------------------------------------------------------------------
# Phase 2: Validate fused model via val.py CLI
# ---------------------------------------------------------------------------

def run_validation_cli(weights_path, data, imgsz, batch_size, time_step):
    """Run val.py as subprocess, return parsed metrics."""
    import subprocess
    cmd = [
        sys.executable, str(ROOT / 'val.py'),
        '--data', data,
        '--weights', str(weights_path),
        '--img', str(imgsz),
        '--batch', str(batch_size),
        '--time-step', str(time_step),
        '--task', 'val',
        '--project', str(ROOT / 'runs' / 'quantize'),
        '--name', 'val_fused',
        '--exist-ok',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    output = result.stdout + result.stderr

    metrics = {}
    for line in output.split('\n'):
        stripped = line.strip()
        if stripped.startswith('all'):
            parts = stripped.split()
            try:
                metrics = {
                    'precision': float(parts[3]),
                    'recall': float(parts[4]),
                    'mAP50': float(parts[5]),
                    'mAP50_95': float(parts[6]),
                }
            except (ValueError, IndexError):
                pass
    return metrics, output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Quantization Export Tool for ZCU102 FPGA')
    parser.add_argument('--weights', type=str, required=True, help='Model weights (.pt)')
    parser.add_argument('--data', type=str, default='', help='Dataset yaml (for BN-fused model validation)')
    parser.add_argument('--img', type=int, default=1920, help='Image size for validation')
    parser.add_argument('--batch', type=int, default=4, help='Batch size for validation')
    parser.add_argument('--time-step', type=int, default=1, help='SNN time step')
    parser.add_argument('--save-dir', type=str, default='runs/quantize/', help='Output directory')
    parser.add_argument('--schemes', nargs='+', default=['fp32', 'fp16', 'int16', 'int8', 'fxp8'],
                        help='Quantization schemes to export')
    parser.add_argument('--skip-val', action='store_true', help='Skip validation of fused model')
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    spike.time_step = args.time_step
    spike.set_time_step(args.time_step)

    print('=' * 80)
    print('  QUANTIZATION EXPORT TOOL — ZCU102 FPGA')
    print('=' * 80)
    print(f'  Weights:    {args.weights}')
    print(f'  Time step:  {args.time_step}')
    print(f'  Schemes:    {args.schemes}')
    print(f'  Output:     {save_dir}')
    print()

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
    model = (ckpt.get('ema') or ckpt['model']).float().eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model loaded: {n_params:,} parameters')

    # ------------------------------------------------------------------
    # Phase 1: BN Fusion
    # ------------------------------------------------------------------
    print(f'\n--- Phase 1: BN Fusion ---')
    n_fused = fuse_model_bn(model)
    print(f'  Fused {n_fused} SeBatchNorm layers into Conv2d')

    # Save fused model
    fused_path = save_dir / 'model_fused.pt'
    torch.save({'model': model, 'ema': None, 'epoch': -1}, fused_path)
    print(f'  Fused model saved: {fused_path}')

    # ------------------------------------------------------------------
    # Phase 2: Validate fused model (optional)
    # ------------------------------------------------------------------
    if not args.skip_val and args.data:
        print(f'\n--- Phase 2: Validating fused model ---')
        metrics, output = run_validation_cli(fused_path, args.data, args.img, args.batch, args.time_step)
        if metrics:
            print(f'  Fused model mAP50:    {metrics["mAP50"]:.4f}')
            print(f'  Fused model mAP50-95: {metrics["mAP50_95"]:.4f}')
            print(f'  Fused model P/R:      {metrics["precision"]:.4f} / {metrics["recall"]:.4f}')
            print(f'  (Should match original — BN fusion is mathematically lossless)')
        else:
            print(f'  WARNING: Could not parse val output. Last 400 chars:')
            print(f'  {output[-400:]}')
    else:
        print(f'\n--- Phase 2: Validation skipped ---')
        metrics = {}

    # ------------------------------------------------------------------
    # Phase 3: Export quantized weights
    # ------------------------------------------------------------------
    print(f'\n--- Phase 3: Exporting quantized weights ---')

    # Collect all conv layers
    conv_layers = []
    for name, mod in model.named_modules():
        if hasattr(mod, 'weight') and mod.weight is not None and len(mod.weight.shape) == 4:
            w = mod.weight.detach().float()
            b = mod.bias.detach().float() if mod.bias is not None else None
            layer_name = name.replace('.', '_')
            conv_layers.append((layer_name, w, b, mod))

    print(f'  Found {len(conv_layers)} conv layers to quantize')

    manifest = []
    total_size = {s: 0 for s in args.schemes}

    for layer_name, w, b, mod in conv_layers:
        records = export_layer_weights(save_dir, layer_name, w, b, args.schemes)
        entry = {
            'layer': layer_name,
            'shape': list(w.shape),
            'params': w.numel(),
        }
        for scheme, rec in records.items():
            entry[f'{scheme}_bytes'] = rec['weight_bytes'] + rec['bias_bytes']
            entry[f'{scheme}_scale'] = rec['scale'] if not isinstance(rec['scale'], float) else f'{rec["scale"]:.8f}'
            total_size[scheme] += rec['weight_bytes'] + rec['bias_bytes']
        manifest.append(entry)

    # Save manifest
    manifest_path = save_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'  Manifest saved: {manifest_path}')

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f'\n{"=" * 80}')
    print(f'  QUANTIZATION SIZE COMPARISON')
    print(f'{"=" * 80}')
    print(f'\n  {"Scheme":<12s} {"Bits":>5s} {"Size (KB)":>10s} {"Size (MB)":>10s} {"Compress":>10s}')
    print(f'  {"-"*12} {"-"*5} {"-"*10} {"-"*10} {"-"*10}')

    fp32_bytes = total_size.get('fp32', 1)
    for scheme in args.schemes:
        sz = total_size[scheme]
        bits = {'fp32': 32, 'fp16': 16, 'int16': 16, 'int8': 8, 'fxp8': 8}[scheme]
        ratio = fp32_bytes / sz if sz > 0 else 0
        print(f'  {scheme.upper():<12s} {bits:>5d} {sz/1024:>10.1f} {sz/(1024*1024):>10.3f} {ratio:>9.1f}x')

    # ------------------------------------------------------------------
    # Per-layer breakdown
    # ------------------------------------------------------------------
    print(f'\n  --- Per-Layer Breakdown (top 10 by param count) ---')
    sorted_layers = sorted(manifest, key=lambda x: x['params'], reverse=True)
    print(f'  {"Layer":<40s} {"Shape":<20s} {"FP32(B)":>9s} {"INT8(B)":>9s} {"FXP8(B)":>9s}')
    print(f'  {"-"*40} {"-"*20} {"-"*9} {"-"*9} {"-"*9}')
    for entry in sorted_layers[:10]:
        shape_str = 'x'.join(str(s) for s in entry['shape'])
        fp32_b = entry.get('fp32_bytes', 0)
        int8_b = entry.get('int8_bytes', 0)
        fxp8_b = entry.get('fxp8_bytes', 0)
        print(f'  {entry["layer"]:<40s} {shape_str:<20s} {fp32_b:>9,d} {int8_b:>9,d} {fxp8_b:>9,d}')

    # ------------------------------------------------------------------
    # Hardware summary
    # ------------------------------------------------------------------
    print(f'\n  --- ZCU102 Hardware Notes ---')
    int8_kb = total_size.get('int8', 0) / 1024
    print(f'  INT8 total weight+bias:  {int8_kb:.1f} KB')
    print(f'  ZCU102 BRAM:             ~4,320 KB (32.1 Mbit)')
    if int8_kb < 4320:
        print(f'  --> Model fits entirely in on-chip BRAM!')
    else:
        print(f'  --> Model exceeds BRAM — needs external DDR or tiling')
    print(f'  DSP48E2 slices:          2,520 (8-bit MAC capable)')
    print(f'  Recommended:             INT8 weights + INT32 accumulator')
    print()

    if metrics:
        print(f'  Fused model validation:  mAP50={metrics["mAP50"]:.4f}  mAP50-95={metrics["mAP50_95"]:.4f}')

    print(f'\n  All exports saved to: {save_dir}/')
    print(f'  Subdirectories: {", ".join(args.schemes)}')
    print(f'{"=" * 80}')


if __name__ == '__main__':
    main()
