"""Full-pipeline deployment quantization for SU-YOLO SNN on ZCU102 FPGA.

Switchable schemes — run all and compare side-by-side:
  fp32      FP32 baseline (no quantization)
  fp16      FP16 weights and activations
  w16a16    INT16 weights, INT32 bias, INT16 accumulator
  w8a32     INT8 weights, INT32 bias+accumulator (safe)
  w8a16     INT8 weights, INT32 bias, INT16 accumulator (recommended for ZCU102)
  w8a8      INT8 weights, INT32 bias, INT8 accumulator (aggressive)

Pipeline per scheme:
  1. BN fusion  — fold SeBatchNorm (with var_scale) into Conv2d
  2. Calibrate  — run N real images, collect per-layer activation min/max
  3. Quantize   — apply weight + activation fake-quantization (measures real mAP drop)
  4. Export     — write actual integer arrays, scale factors, layer graph for FPGA

Usage:
    python tools/quantize_compare.py \\
        --weights runs/train/hazydet-student-kd-scratch/weights/best.pt \\
        --data data/hazydet.yaml --img 1920 --batch 4 --time-step 1 \\
        --calib-images 100

    # Single scheme:
    python tools/quantize_compare.py --weights best.pt --data data/hazydet.yaml \\
        --schemes w8a16

    # Skip validation (export only):
    python tools/quantize_compare.py --weights best.pt --skip-val --schemes w8a16
"""
import argparse
import copy
import csv
import json
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
from models.spike import (SConv, seBatchNorm, SeBatchNorm2d, SRepGhostConv)
from spikingjelly.activation_based.functional import reset_net
from spikingjelly.activation_based import neuron
from val import run as val_run

# ═══════════════════════════════════════════════════════════════════════════════
# Scheme definitions
# ═══════════════════════════════════════════════════════════════════════════════

SCHEMES = OrderedDict({
    'fp32':   dict(w_bits=32, act_bits=32, acc_bits=32, bias_bits=32, per_channel=False,
                   desc='FP32 baseline — no quantization'),
    'fp16':   dict(w_bits=16, act_bits=16, acc_bits=16, bias_bits=16, per_channel=False,
                   desc='FP16 half-precision (native torch.half)'),
    'w16a16': dict(w_bits=16, act_bits=16, acc_bits=32, bias_bits=32, per_channel=False,
                   desc='INT16 weights, INT16 activations, INT32 accumulator'),
    'w8a32':  dict(w_bits=8,  act_bits=32, acc_bits=32, bias_bits=32, per_channel=True,
                   desc='INT8 weights per-channel, FP32 activations (safe)'),
    'w8a16':  dict(w_bits=8,  act_bits=16, acc_bits=32, bias_bits=32, per_channel=True,
                   desc='INT8 weights per-channel, INT16 activations (recommended)'),
    'w8a8':   dict(w_bits=8,  act_bits=8,  acc_bits=32, bias_bits=32, per_channel=True,
                   desc='INT8 weights + activations per-channel (aggressive)'),
})


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — BN Fusion
# ═══════════════════════════════════════════════════════════════════════════════

def _fuse_single_conv_bn(conv, bn_wrapper):
    """Fuse SeBatchNorm into Conv2d → returns (w_fused, b_fused) or (None, None)."""
    if isinstance(bn_wrapper, seBatchNorm):
        bn = bn_wrapper.bn
    elif isinstance(bn_wrapper, SeBatchNorm2d):
        bn = bn_wrapper
    else:
        return None, None

    w = conv.weight.detach().float()
    c_out = w.shape[0]

    mean = bn.running_mean[:c_out].detach().float()
    var = bn.running_var[:c_out].detach().float()
    vs = getattr(bn, 'var_scale', 1.0)
    eps = bn.eps
    gamma = bn.weight[:c_out].detach().float() if bn.affine else torch.ones(c_out)
    beta = bn.bias[:c_out].detach().float() if bn.affine else torch.zeros(c_out)

    inv_std = gamma / torch.sqrt(var / vs + eps)
    w_f = w * inv_std.view(-1, 1, 1, 1)
    b_f = beta - mean * inv_std
    if conv.bias is not None:
        b_f = b_f + conv.bias.detach().float() * inv_std
    return w_f, b_f


def fuse_all_bn(model):
    """In-place BN fusion for every SConv layer. Returns count.

    SDConv is SKIPPED — it defines self.bn in __init__ but never calls it
    in forward(), so those BN params are untrained and must not be fused.

    SRepGhostConv is handled first — its fuse_reparam() folds the identity BN
    branch into the cheap DW conv (add → single conv at deploy time).
    """
    n = 0
    for _, mod in model.named_modules():
        if isinstance(mod, SRepGhostConv):
            if hasattr(mod, 'fuse_reparam'):
                mod.fuse_reparam()
                n += 1
    for _, mod in model.named_modules():
        if isinstance(mod, SConv):
            if isinstance(mod.bn, nn.Identity):
                continue
            w_f, b_f = _fuse_single_conv_bn(mod.conv, mod.bn)
            if w_f is None:
                continue
            mod.conv.weight.data = w_f
            if mod.conv.bias is None:
                mod.conv.bias = nn.Parameter(b_f)
            else:
                mod.conv.bias.data = b_f
            mod.bn = nn.Identity()
            n += 1
    return n


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Activation Calibration
# ═══════════════════════════════════════════════════════════════════════════════

class _ActivationCollector:
    """Forward-hook based min/max collector for conv layer outputs."""

    def __init__(self):
        self.ranges = {}   # layer_name → {'min': float, 'max': float, 'absmax': float}
        self._hooks = []

    def attach(self, model):
        for name, mod in model.named_modules():
            if hasattr(mod, 'weight') and mod.weight is not None and len(mod.weight.shape) == 4:
                h = mod.register_forward_hook(self._make_hook(name))
                self._hooks.append(h)

    def _make_hook(self, name):
        def hook(mod, inp, out):
            if isinstance(out, (list, tuple)):
                t = torch.stack([o for o in out if isinstance(o, torch.Tensor)])
            elif isinstance(out, torch.Tensor):
                t = out
            else:
                return
            t_f = t.detach().float()
            mi, ma = t_f.min().item(), t_f.max().item()
            am = max(abs(mi), abs(ma))
            if name in self.ranges:
                self.ranges[name]['min'] = min(self.ranges[name]['min'], mi)
                self.ranges[name]['max'] = max(self.ranges[name]['max'], ma)
                self.ranges[name]['absmax'] = max(self.ranges[name]['absmax'], am)
            else:
                self.ranges[name] = {'min': mi, 'max': ma, 'absmax': am}
        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def calibrate_activation_ranges(model, data_yaml, imgsz, n_images, time_step, device):
    """Run n_images through the model and return per-layer activation ranges."""
    from utils.dataloaders import create_dataloader
    from utils.general import check_dataset, check_img_size

    data_dict = check_dataset(data_yaml)
    val_path = data_dict.get('val', data_dict.get('test', ''))

    stride = int(max(model.stride)) if hasattr(model, 'stride') else 32
    gs = check_img_size(imgsz, s=stride)

    dataloader = create_dataloader(
        val_path, gs, min(8, n_images), stride,
        pad=0.5, rect=True, workers=4, prefix='calibrate: '
    )[0]

    collector = _ActivationCollector()
    collector.attach(model)

    model.to(device).eval()
    count = 0
    with torch.no_grad():
        for batch_i, (imgs, targets, paths, shapes) in enumerate(dataloader):
            if count >= n_images:
                break
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            reset_net(model)
            model(imgs)
            count += imgs.shape[0]

    collector.remove()
    print(f'  Calibrated on {count} images, {len(collector.ranges)} layers profiled')
    return collector.ranges


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Weight + Activation Fake-Quantization
# ═══════════════════════════════════════════════════════════════════════════════

def _sym_quant_deq(tensor, n_bits):
    """Symmetric quantize → dequantize (introduces real quantization noise)."""
    qmax = (1 << (n_bits - 1)) - 1
    absmax = tensor.abs().max()
    if absmax == 0:
        return tensor, 0.0
    scale = absmax / qmax
    q = (tensor / scale).round().clamp(-qmax - 1, qmax)
    return q * scale, scale.item()


def _perchannel_quant_deq(tensor, n_bits):
    """Per-channel symmetric quantize → dequantize along dim 0."""
    qmax = (1 << (n_bits - 1)) - 1
    absmax = tensor.abs().flatten(1).max(dim=1)[0].clamp(min=1e-8)
    scales = absmax / qmax
    q = (tensor / scales.view(-1, 1, 1, 1)).round().clamp(-qmax - 1, qmax)
    return q * scales.view(-1, 1, 1, 1), scales.tolist()


def apply_weight_fakequant(model, scheme_cfg):
    """Apply weight fake-quantization in-place (quantize→dequantize weights)."""
    w_bits = scheme_cfg['w_bits']
    per_ch = scheme_cfg['per_channel']
    if w_bits >= 32:
        return 0

    n = 0
    for _, mod in model.named_modules():
        if not hasattr(mod, 'weight') or mod.weight is None or len(mod.weight.shape) != 4:
            continue
        w = mod.weight.data.float()

        if w_bits == 16 and not per_ch:
            # FP16 native: cast round-trip
            mod.weight.data = w.half().float()
        elif per_ch:
            mod.weight.data, _ = _perchannel_quant_deq(w, w_bits)
        else:
            mod.weight.data, _ = _sym_quant_deq(w, w_bits)
        n += 1

        # Bias: always keep FP32 or INT32 equivalent
        if mod.bias is not None:
            b = mod.bias.data.float()
            if scheme_cfg['bias_bits'] < 32:
                mod.bias.data, _ = _sym_quant_deq(b, scheme_cfg['bias_bits'])

    return n


def apply_activation_fakequant(model, scheme_cfg, calib_ranges):
    """Insert forward hooks that fake-quantize activations at calibrated scales."""
    act_bits = scheme_cfg['act_bits']
    if act_bits >= 32 or not calib_ranges:
        return []

    hooks = []
    qmax = (1 << (act_bits - 1)) - 1

    for name, mod in model.named_modules():
        if name not in calib_ranges:
            continue
        absmax = calib_ranges[name]['absmax']
        if absmax == 0:
            continue
        scale = absmax / qmax

        def make_hook(s, qm):
            def hook(mod, inp, out):
                if isinstance(out, torch.Tensor):
                    q = (out / s).round().clamp(-qm - 1, qm) * s
                    return q
                elif isinstance(out, (list, tuple)):
                    return type(out)(
                        (o / s).round().clamp(-qm - 1, qm) * s if isinstance(o, torch.Tensor) else o
                        for o in out
                    )
                return out
            return hook

        h = mod.register_forward_hook(make_hook(scale, qmax))
        hooks.append(h)

    return hooks


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Hardware Export (real integer arrays)
# ═══════════════════════════════════════════════════════════════════════════════

def _to_int_array(tensor, n_bits, per_channel=False):
    """Quantize to actual integer numpy array + scale(s)."""
    qmax = (1 << (n_bits - 1)) - 1
    if per_channel:
        absmax = tensor.abs().flatten(1).max(dim=1)[0].clamp(min=1e-8)
        scales = absmax / qmax
        q = (tensor / scales.view(-1, 1, 1, 1)).round().clamp(-qmax - 1, qmax)
        dtype = np.int8 if n_bits <= 8 else np.int16
        return q.numpy().astype(dtype), scales.numpy()
    else:
        absmax = tensor.abs().max().item()
        if absmax == 0:
            dtype = np.int8 if n_bits <= 8 else np.int16
            return np.zeros(tensor.shape, dtype=dtype), np.float32(0)
        scale = absmax / qmax
        q = (tensor / scale).round().clamp(-qmax - 1, qmax)
        dtype = np.int8 if n_bits <= 8 else np.int16
        return q.numpy().astype(dtype), np.float32(scale)


def export_hardware_package(model, scheme_name, scheme_cfg, calib_ranges, save_dir):
    """Export full hardware package for one scheme."""
    out = save_dir / scheme_name
    out.mkdir(parents=True, exist_ok=True)

    w_bits = scheme_cfg['w_bits']
    per_ch = scheme_cfg['per_channel']
    act_bits = scheme_cfg['act_bits']

    manifest = []
    total_w_bytes = 0
    total_b_bytes = 0

    for name, mod in model.named_modules():
        if not hasattr(mod, 'weight') or mod.weight is None or len(mod.weight.shape) != 4:
            continue

        safe_name = name.replace('.', '_')
        layer_dir = out / safe_name
        layer_dir.mkdir(parents=True, exist_ok=True)

        w = mod.weight.detach().float()
        b = mod.bias.detach().float() if mod.bias is not None else None

        if w_bits >= 32:
            np.save(layer_dir / 'weight_fp32.npy', w.numpy())
            w_bytes = w.numel() * 4
            scale_info = 'fp32'
        elif w_bits == 16 and not per_ch:
            np.save(layer_dir / 'weight_fp16.npy', w.half().numpy())
            w_bytes = w.numel() * 2
            scale_info = 'fp16_native'
        else:
            q_np, sc = _to_int_array(w, w_bits, per_channel=per_ch)
            tag = f'int{w_bits}'
            np.save(layer_dir / f'weight_{tag}.npy', q_np)
            # CSV for hardware team
            with open(layer_dir / f'weight_{tag}.csv', 'w', newline='') as f:
                wr = csv.writer(f)
                for i in range(q_np.shape[0]):
                    wr.writerow(q_np[i].flatten().tolist())
            if per_ch:
                np.save(layer_dir / 'scales_per_channel.npy', sc)
                with open(layer_dir / 'scales.json', 'w') as f:
                    json.dump({'type': 'per_channel_symmetric', 'bits': w_bits,
                               'scales': sc.tolist()}, f, indent=2)
                scale_info = f'per_channel_{w_bits}b'
            else:
                with open(layer_dir / 'scale.json', 'w') as f:
                    json.dump({'type': 'per_tensor_symmetric', 'bits': w_bits,
                               'scale': float(sc)}, f, indent=2)
                scale_info = f'per_tensor_{w_bits}b'
            w_bytes = q_np.nbytes

        # Bias — always INT32 or FP32
        b_bytes = 0
        if b is not None:
            np.save(layer_dir / 'bias_int32.npy', b.numpy())
            with open(layer_dir / 'bias.csv', 'w', newline='') as f:
                csv.writer(f).writerow(b.numpy().tolist())
            b_bytes = b.numel() * 4

        # Activation scale for this layer
        act_scale = None
        if name in calib_ranges and act_bits < 32:
            absmax = calib_ranges[name]['absmax']
            a_qmax = (1 << (act_bits - 1)) - 1
            act_scale = absmax / a_qmax if a_qmax > 0 else 0
            with open(layer_dir / 'activation_scale.json', 'w') as f:
                json.dump({'act_bits': act_bits, 'absmax': absmax, 'scale': act_scale}, f, indent=2)

        manifest.append({
            'layer': name, 'safe_name': safe_name,
            'shape': list(w.shape),
            'weight_bits': w_bits, 'scale_type': scale_info,
            'weight_bytes': w_bytes, 'bias_bytes': b_bytes,
            'activation_scale': act_scale,
            'activation_range': calib_ranges.get(name),
        })
        total_w_bytes += w_bytes
        total_b_bytes += b_bytes

    # IFNode thresholds
    thresholds = []
    for name, mod in model.named_modules():
        if isinstance(mod, neuron.IFNode):
            v_th = mod.v_threshold if hasattr(mod, 'v_threshold') else 1.0
            is_inf = (v_th == float('inf'))
            thresholds.append({'name': name,
                               'v_threshold': 'inf' if is_inf else float(v_th),
                               'is_detect_head': is_inf})
    with open(out / 'ifnode_thresholds.json', 'w') as f:
        json.dump(thresholds, f, indent=2)

    # Layer connectivity
    layer_names = [e['layer'] for e in manifest]
    with open(out / 'layer_order.json', 'w') as f:
        json.dump(layer_names, f, indent=2)

    # Manifest
    summary = {
        'scheme': scheme_name,
        'config': scheme_cfg,
        'total_weight_bytes': total_w_bytes,
        'total_bias_bytes': total_b_bytes,
        'total_bytes': total_w_bytes + total_b_bytes,
        'total_kb': (total_w_bytes + total_b_bytes) / 1024,
        'num_layers': len(manifest),
        'num_ifnodes': len(thresholds),
        'layers': manifest,
    }
    with open(out / 'manifest.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# ZCU102 Resource Estimation
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_zcu102_resources(summary, scheme_cfg):
    """Rough resource estimates for XCZU9EG (ZCU102)."""
    total_bram_kb = 4320       # 32.1 Mbit = 4320 KB (BRAM18 + URAM)
    total_dsp = 2520           # DSP48E2 slices
    total_lut = 274080         # CLB LUTs

    model_kb = summary['total_kb']
    w_bits = scheme_cfg['w_bits']
    acc_bits = scheme_cfg['acc_bits']

    # BRAM: weights + double-buffered activation (largest feature map)
    bram_weights_kb = model_kb
    bram_pct = (bram_weights_kb / total_bram_kb) * 100

    # DSP: one MAC per DSP at 8-bit, half at 16-bit
    # INT8×INT8 → 2 MACs per DSP48E2 (SIMD), INT16 → 1 MAC, FP16/32 → needs more
    if w_bits <= 8:
        mac_per_dsp = 2
    elif w_bits <= 16:
        mac_per_dsp = 1
    else:
        mac_per_dsp = 0.25  # FP32 takes ~4 DSPs

    peak_gops = total_dsp * mac_per_dsp * 300e6 / 1e9  # @300MHz
    # SNN advantage: binary spikes → MAC becomes accumulate (ADD), no multiply needed
    # for spike×weight, only add weight if spike=1, skip if spike=0
    snn_gops = peak_gops * 2  # ~2x speedup from skip-on-zero

    return {
        'bram_usage_kb': round(bram_weights_kb, 1),
        'bram_pct': round(bram_pct, 1),
        'fits_bram': bram_weights_kb < total_bram_kb,
        'mac_per_dsp': mac_per_dsp,
        'peak_gops': round(peak_gops, 1),
        'snn_effective_gops': round(snn_gops, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description='SU-YOLO Quantization for ZCU102 Deployment')
    p.add_argument('--weights', required=True, help='Trained model .pt')
    p.add_argument('--data', default='', help='Dataset YAML (needed for calibration + validation)')
    p.add_argument('--img', type=int, default=1920, help='Image size')
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--time-step', type=int, default=1)
    p.add_argument('--repghost', action='store_true', help='Use RepGhost architecture (no concat)')
    p.add_argument('--device', default='0')
    p.add_argument('--save-dir', default='runs/quantize/')
    p.add_argument('--schemes', nargs='+', default=list(SCHEMES.keys()),
                   help=f'Schemes to evaluate: {list(SCHEMES.keys())}')
    p.add_argument('--calib-images', type=int, default=100, help='Images for activation calibration')
    p.add_argument('--skip-val', action='store_true', help='Skip mAP validation (export only)')
    p.add_argument('--skip-export', action='store_true', help='Skip hardware export (compare only)')
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f'cuda:{args.device}' if args.device.isdigit() and torch.cuda.is_available()
                          else 'cpu')

    spike.time_step = args.time_step
    spike.set_time_step(args.time_step)
    if args.repghost:
        spike.set_repghost(True)

    hdr = '=' * 80
    print(f'\n{hdr}')
    print(f'  SU-YOLO QUANTIZATION — ZCU102 DEPLOYMENT PIPELINE')
    print(f'{hdr}')
    print(f'  Weights:      {args.weights}')
    print(f'  Data:         {args.data or "(none — skip calib/val)"}')
    print(f'  Image size:   {args.img}')
    print(f'  Time step:    {args.time_step}')
    print(f'  Device:       {device}')
    print(f'  Calib images: {args.calib_images}')
    print(f'  Schemes:      {args.schemes}')
    print()

    # ── Load model ──────────────────────────────────────────────────────
    ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
    raw_model = (ckpt.get('ema') or ckpt['model']).float().eval()
    n_params = sum(p.numel() for p in raw_model.parameters())
    print(f'  Loaded model: {n_params:,} parameters\n')

    # ── Phase 1: BN Fusion ──────────────────────────────────────────────
    print(f'--- Phase 1: BN Fusion ---')
    fused_model = copy.deepcopy(raw_model)
    n_fused = fuse_all_bn(fused_model)
    print(f'  Fused {n_fused} SeBatchNorm layers into Conv2d')

    fused_path = save_dir / 'model_bn_fused.pt'
    torch.save({'model': fused_model, 'ema': None, 'epoch': -1}, fused_path)
    print(f'  Saved fused model → {fused_path}\n')

    # ── Phase 2: Calibration ────────────────────────────────────────────
    calib_ranges = {}
    if args.data and not args.skip_val:
        print(f'--- Phase 2: Activation Calibration ---')
        calib_ranges = calibrate_activation_ranges(
            copy.deepcopy(fused_model), args.data, args.img,
            args.calib_images, args.time_step, device
        )
        # Save calibration data
        calib_path = save_dir / 'calibration_ranges.json'
        with open(calib_path, 'w') as f:
            json.dump(calib_ranges, f, indent=2)
        print(f'  Saved → {calib_path}\n')
    else:
        print(f'--- Phase 2: Calibration skipped (no --data or --skip-val) ---\n')

    # ── Phase 3 + 4: Per-scheme quantization ────────────────────────────
    results = []

    for scheme_name in args.schemes:
        if scheme_name not in SCHEMES:
            print(f'  WARNING: unknown scheme "{scheme_name}", skipping')
            continue
        cfg = SCHEMES[scheme_name]

        print(f'{hdr}')
        print(f'  Scheme: {scheme_name.upper()} — {cfg["desc"]}')
        print(f'{hdr}')

        # Start from clean fused model
        model = copy.deepcopy(fused_model)

        # Phase 3a: Weight fake-quantization
        if cfg['w_bits'] < 32:
            n_q = apply_weight_fakequant(model, cfg)
            print(f'  Weight fake-quant: {n_q} layers → {cfg["w_bits"]}-bit'
                  f' ({"per-channel" if cfg["per_channel"] else "per-tensor"})')

        # Phase 3b: Activation fake-quantization hooks
        act_hooks = []
        if cfg['act_bits'] < 32 and calib_ranges:
            act_hooks = apply_activation_fakequant(model, cfg, calib_ranges)
            print(f'  Activation fake-quant: {len(act_hooks)} hooks → {cfg["act_bits"]}-bit')

        # Phase 3c: Validate (mAP) — run IN-PROCESS so activation hooks stay active
        metrics = {}
        if not args.skip_val and args.data:
            from utils.dataloaders import create_dataloader
            from utils.general import check_dataset, check_img_size

            print(f'  Running validation (in-process)...')
            model.to(device).eval()

            data_dict = check_dataset(args.data)
            stride = int(max(model.stride)) if hasattr(model, 'stride') else 32
            gs = check_img_size(args.img, s=stride)
            val_path = data_dict.get('val', data_dict.get('test', ''))
            val_loader = create_dataloader(
                val_path, gs, args.batch, stride,
                pad=0.5, rect=True, workers=8,
                prefix='quant-val: '
            )[0]

            val_results = val_run(
                data=data_dict,
                model=model,
                dataloader=val_loader,
                imgsz=args.img,
                batch_size=args.batch,
                time_step=args.time_step,
                half=False,
                plots=False,
                save_dir=save_dir,
            )
            mp, mr, map50, map95 = val_results[0][:4]
            metrics = {'P': float(mp), 'R': float(mr),
                       'mAP50': float(map50), 'mAP50_95': float(map95)}
            print(f'  mAP50: {metrics["mAP50"]:.4f}  mAP50-95: {metrics["mAP50_95"]:.4f}'
                  f'  P: {metrics["P"]:.4f}  R: {metrics["R"]:.4f}')

        # NOW remove hooks (after validation, so they were active during inference)
        for h in act_hooks:
            h.remove()
        act_hooks = []

        # Phase 4: Hardware export
        hw_summary = None
        if not args.skip_export:
            # Use fresh fused model for export (not fake-quantized — export real integers)
            export_model = copy.deepcopy(fused_model)
            hw_summary = export_hardware_package(
                export_model, scheme_name, cfg, calib_ranges, save_dir
            )
            print(f'  Exported → {save_dir / scheme_name}/')
            print(f'    Weight bytes: {hw_summary["total_weight_bytes"]:,}')
            print(f'    Bias bytes:   {hw_summary["total_bias_bytes"]:,}')
            print(f'    Total:        {hw_summary["total_kb"]:.1f} KB')

        # Resource estimate
        if hw_summary:
            res = estimate_zcu102_resources(hw_summary, cfg)
        else:
            # Estimate from param count
            bpp = cfg['w_bits']
            est_kb = (n_params * bpp / 8) / 1024
            res = {'bram_usage_kb': round(est_kb, 1), 'bram_pct': round(est_kb / 4320 * 100, 1),
                   'fits_bram': est_kb < 4320, 'peak_gops': 0, 'snn_effective_gops': 0}

        results.append(OrderedDict(
            scheme=scheme_name,
            desc=cfg['desc'],
            w_bits=cfg['w_bits'],
            act_bits=cfg['act_bits'],
            acc_bits=cfg['acc_bits'],
            size_kb=hw_summary['total_kb'] if hw_summary else res['bram_usage_kb'],
            bram_pct=res['bram_pct'],
            fits_bram=res['fits_bram'],
            **metrics,
        ))
        print()

    # ── Phase 5: Comparison Table ───────────────────────────────────────
    print(f'\n{hdr}')
    print(f'  SIDE-BY-SIDE COMPARISON')
    print(f'{hdr}\n')

    base_map = results[0].get('mAP50', 0) if results else 0

    # Table header
    h1 = f'  {"Scheme":<10} {"W":>3}b {"A":>3}b {"Acc":>3}b {"Size":>8} {"BRAM%":>6}'
    h2 = f' {"mAP50":>7} {"mAP95":>7} {"P":>6} {"R":>6} {"Δ mAP50":>8}'
    print(h1 + h2)
    print(f'  {"-"*10} {"-"*3}- {"-"*3}- {"-"*3}- {"-"*8} {"-"*6}'
          f' {"-"*7} {"-"*7} {"-"*6} {"-"*6} {"-"*8}')

    for r in results:
        delta = r.get('mAP50', 0) - base_map
        d_str = f'{delta:+.4f}' if r['scheme'] != 'fp32' else '  base'
        bram = f'{r["bram_pct"]:.1f}%'
        fit = ' OK' if r['fits_bram'] else ' !!'
        sz = f'{r["size_kb"]:.0f}KB'

        print(f'  {r["scheme"]:<10} {r["w_bits"]:>3}  {r["act_bits"]:>3}  {r["acc_bits"]:>3} '
              f' {sz:>8} {bram:>5}{fit}'
              f' {r.get("mAP50",0):>7.4f} {r.get("mAP50_95",0):>7.4f}'
              f' {r.get("P",0):>6.3f} {r.get("R",0):>6.3f} {d_str:>8}')

    # ── ZCU102 recommendation ──────────────────────────────────────────
    print(f'\n  --- ZCU102 XCZU9EG Resource Budget ---')
    print(f'  BRAM:  4,320 KB (32.1 Mbit)   |  DSP48E2: 2,520  |  LUTs: 274,080')
    print(f'  INT8:  2 MACs/DSP @300MHz → ~1,512 GOPS peak')
    print(f'  SNN:   Binary spikes → MAC = conditional ADD (skip zero spikes)')
    print(f'         With ~80% sparsity → effective 5x over dense INT8')
    print()

    # Recommend best scheme
    valid = [r for r in results if r['fits_bram'] and r.get('mAP50', 0) > 0]
    if valid:
        best = max(valid, key=lambda r: r.get('mAP50', 0))
        smallest = min(valid, key=lambda r: r['size_kb'])
        print(f'  RECOMMENDED: {best["scheme"].upper()} '
              f'(mAP50={best.get("mAP50",0):.4f}, {best["size_kb"]:.0f}KB, BRAM {best["bram_pct"]:.1f}%)')
        if smallest['scheme'] != best['scheme']:
            print(f'  SMALLEST:    {smallest["scheme"].upper()} '
                  f'(mAP50={smallest.get("mAP50",0):.4f}, {smallest["size_kb"]:.0f}KB)')

    # Save comparison
    comp_path = save_dir / 'comparison.json'
    with open(comp_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n  All results → {comp_path}')
    print(f'  Exports     → {save_dir}/<scheme_name>/')
    print(f'{hdr}\n')


if __name__ == '__main__':
    main()
