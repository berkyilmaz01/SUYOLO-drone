"""SNN Signal Tracer — walks a real image through every sub-operation of
the SU-YOLO model, logging tensor statistics at each step.

Designed for the hardware team: shows exactly what happens mathematically
when a real image enters the spiking neural network, from raw pixels to
final detections.

Usage:
    python tools/trace_signal.py \
        --weights runs/train/hazydet-student-kd-fixed3/weights/best.pt \
        --image data/hazydet/images/val/00001.jpg \
        --save-dir runs/trace/ --dump-weights
"""
import argparse
import csv
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn

import models.spike as spike
from models.spike import (SConv, SDConv, SGhostConv, SGhostEncoderLite,
                           GhostBasicBlock1, GhostBasicBlock2, SDDetect,
                           SUpsample, SConcat, seBatchNorm, SDFL)
from spikingjelly.activation_based.functional import reset_net
from spikingjelly.activation_based import neuron
from utils.tal.anchor_generator import make_anchors, dist2bbox


# ---------------------------------------------------------------------------
# Tensor statistics helper
# ---------------------------------------------------------------------------

def tensor_stats(t, name='', is_spike=False):
    """Compute statistics for a single tensor or list-of-tensors (time_step)."""
    if isinstance(t, (list, tuple)):
        t = t[0]  # take first time-step for stats (they are identical for T=1)
    if not isinstance(t, torch.Tensor):
        return {'name': name, 'note': 'not a tensor'}

    t_cpu = t.detach().float().cpu()
    info = OrderedDict()
    info['name'] = name
    info['shape'] = list(t_cpu.shape)
    info['dtype'] = str(t.dtype)
    info['min'] = round(t_cpu.min().item(), 6)
    info['max'] = round(t_cpu.max().item(), 6)
    info['mean'] = round(t_cpu.mean().item(), 6)
    info['std'] = round(t_cpu.std().item(), 6)
    zeros = (t_cpu == 0).sum().item()
    total = t_cpu.numel()
    info['sparsity'] = round(zeros / total, 4) if total > 0 else 0
    info['num_elements'] = total

    if is_spike or name.endswith('spike'):
        uniq = t_cpu.unique().tolist()
        info['unique_values'] = [round(v, 2) for v in uniq[:10]]
        info['is_binary'] = set(uniq) <= {0.0, 1.0}

    # Sample 5x5 patch from spatial center of first channel
    if t_cpu.dim() == 4 and t_cpu.shape[2] >= 5 and t_cpu.shape[3] >= 5:
        h, w = t_cpu.shape[2], t_cpu.shape[3]
        ch, cw = h // 2, w // 2
        patch = t_cpu[0, 0, ch-2:ch+3, cw-2:cw+3].tolist()
        info['center_5x5_ch0'] = [[round(v, 4) for v in row] for row in patch]

    return info


def conv_weight_stats(conv_module, name=''):
    """Extract weight statistics from a Conv2d layer."""
    if not hasattr(conv_module, 'weight'):
        return None
    w = conv_module.weight.detach().float().cpu()
    scale = w.abs().max().item() / 127 if w.abs().max().item() > 0 else 1e-8
    return OrderedDict(
        name=name,
        shape=list(w.shape),
        min=round(w.min().item(), 6),
        max=round(w.max().item(), 6),
        mean=round(w.mean().item(), 6),
        std=round(w.std().item(), 6),
        int8_scale=round(scale, 8),
    )


def bn_params(bn_module, name=''):
    """Extract BN parameters needed for hardware fusion."""
    if not hasattr(bn_module, 'running_mean'):
        return None
    info = OrderedDict(name=name)
    if hasattr(bn_module, 'bn'):
        bn = bn_module.bn  # seBatchNorm wraps SeBatchNorm2d
    else:
        bn = bn_module
    info['running_mean_range'] = [round(bn.running_mean.min().item(), 6),
                                   round(bn.running_mean.max().item(), 6)]
    info['running_var_range'] = [round(bn.running_var.min().item(), 6),
                                  round(bn.running_var.max().item(), 6)]
    if hasattr(bn, 'weight') and bn.weight is not None:
        info['gamma_range'] = [round(bn.weight.min().item(), 6),
                                round(bn.weight.max().item(), 6)]
    if hasattr(bn, 'bias') and bn.bias is not None:
        info['beta_range'] = [round(bn.bias.min().item(), 6),
                               round(bn.bias.max().item(), 6)]
    if hasattr(bn, 'var_scale'):
        info['var_scale'] = bn.var_scale
    return info


# ---------------------------------------------------------------------------
# Sub-operation tracers for each module type
# ---------------------------------------------------------------------------

def trace_sconv(sconv, x, prefix, trace_log):
    """Trace SConv: conv -> bn -> IFNode. Returns output list[Tensor]."""
    # Step 1: Conv2d
    x_conv = [sconv.conv(x[i]) for i in range(spike.time_step)]
    trace_log.append(tensor_stats(x_conv, f'{prefix}.conv2d_out'))
    trace_log.append(conv_weight_stats(sconv.conv, f'{prefix}.conv2d_weights'))

    # Step 2: SeBatchNorm
    x_bn = sconv.bn(x_conv)
    trace_log.append(tensor_stats(x_bn, f'{prefix}.bn_out'))
    trace_log.append(bn_params(sconv.bn, f'{prefix}.bn_params'))

    # Step 3: IFNode activation (spike generation)
    x_spike = [sconv.act(x_bn[i]) for i in range(spike.time_step)]
    trace_log.append(tensor_stats(x_spike, f'{prefix}.ifnode_spike', is_spike=True))

    return x_spike


def trace_sdconv(sdconv, x, prefix, trace_log):
    """Trace SDConv: conv -> stack -> mean. Returns single Tensor."""
    x_conv = [sdconv.conv(x[i]) for i in range(spike.time_step)]
    trace_log.append(tensor_stats(x_conv, f'{prefix}.conv2d_out'))
    trace_log.append(conv_weight_stats(sdconv.conv, f'{prefix}.conv2d_weights'))

    x_stack = torch.stack(x_conv, 0)
    out = x_stack.mean(0)
    trace_log.append(tensor_stats(out, f'{prefix}.temporal_mean_out'))
    return out


def trace_sghost_conv(sghost, x, prefix, trace_log):
    """Trace SGhostConv: primary(SConv) -> cheap(SConv) -> cat."""
    # Primary branch
    p = trace_sconv(sghost.primary, x, f'{prefix}.primary', trace_log)

    if sghost.cheap is None:
        return p

    # Cheap branch (depthwise)
    g = trace_sconv(sghost.cheap, p, f'{prefix}.cheap', trace_log)

    # Concat
    out = [torch.cat([p[i], g[i]], 1) for i in range(spike.time_step)]
    trace_log.append(tensor_stats(out, f'{prefix}.ghost_cat'))
    return out


def trace_sghost_encoder_lite(encoder, x_raw, prefix, trace_log):
    """Trace SGhostEncoderLite: replicate -> conv1(SConv) -> conv2(SGhostConv)."""
    # Step 1: Replicate input for each time step
    x = [x_raw for _ in range(spike.time_step)]
    trace_log.append(tensor_stats(x, f'{prefix}.time_replicate'))

    # Step 2: conv1 (SConv, stride 2)
    out = trace_sconv(encoder.conv1, x, f'{prefix}.conv1', trace_log)

    # Step 3: conv2 (SGhostConv, stride 1)
    out = trace_sghost_conv(encoder.conv2, out, f'{prefix}.conv2', trace_log)
    return out


def trace_ghost_basic_block1(block, x, prefix, trace_log):
    """Trace GhostBasicBlock1: cvres -> cv0 -> chunk -> act2 -> cv2 -> add -> cat -> act3."""
    # Residual branch
    xres = trace_sghost_conv(block.cvres, x, f'{prefix}.cvres', trace_log)

    # Main branch
    x_cv0 = trace_sghost_conv(block.cv0, x, f'{prefix}.cv0', trace_log)

    # Chunk split
    x1, x2 = [], []
    for i in range(spike.time_step):
        y1, y2 = x_cv0[i].chunk(2, 1)
        x1.append(y1)
        x2.append(y2)
    trace_log.append(tensor_stats(x1, f'{prefix}.chunk_branch1'))
    trace_log.append(tensor_stats(x2, f'{prefix}.chunk_branch2'))

    # IFNode on branch 2
    x3 = [block.act2(x2[i]) for i in range(spike.time_step)]
    trace_log.append(tensor_stats(x3, f'{prefix}.act2_spike', is_spike=True))

    # cv2 on activated branch
    x4 = trace_sghost_conv(block.cv2, x3, f'{prefix}.cv2', trace_log)

    # Residual addition
    for i in range(spike.time_step):
        x4[i] = x4[i] + xres[i]
    trace_log.append(tensor_stats(x4, f'{prefix}.residual_add'))

    # Concat branches
    out = [torch.cat([x1[i], x4[i]], 1) for i in range(spike.time_step)]
    trace_log.append(tensor_stats(out, f'{prefix}.concat'))

    # Final IFNode
    out = [block.act3(out[i]) for i in range(spike.time_step)]
    trace_log.append(tensor_stats(out, f'{prefix}.act3_spike', is_spike=True))
    return out


def trace_ghost_basic_block2(block, x, prefix, trace_log):
    """Trace GhostBasicBlock2: cv0 -> chunk -> act2 -> cv2 -> cat -> act3."""
    # cv0
    x_cv0 = trace_sghost_conv(block.cv0, x, f'{prefix}.cv0', trace_log)

    # Chunk split
    x1, x2 = [], []
    for i in range(spike.time_step):
        y1, y2 = x_cv0[i].chunk(2, 1)
        x1.append(y1)
        x2.append(y2)
    trace_log.append(tensor_stats(x1, f'{prefix}.chunk_branch1'))
    trace_log.append(tensor_stats(x2, f'{prefix}.chunk_branch2'))

    # IFNode on branch 2
    x3 = [block.act2(x2[i]) for i in range(spike.time_step)]
    trace_log.append(tensor_stats(x3, f'{prefix}.act2_spike', is_spike=True))

    # cv2
    x4 = trace_sghost_conv(block.cv2, x3, f'{prefix}.cv2', trace_log)

    # Concat
    out = [torch.cat([x1[i], x4[i]], 1) for i in range(spike.time_step)]
    trace_log.append(tensor_stats(out, f'{prefix}.concat'))

    # Final IFNode
    out = [block.act3(out[i]) for i in range(spike.time_step)]
    trace_log.append(tensor_stats(out, f'{prefix}.act3_spike', is_spike=True))
    return out


def trace_supsample(upsample, x, prefix, trace_log):
    """Trace SUpsample."""
    out = [upsample.up(x[i]) for i in range(spike.time_step)]
    trace_log.append(tensor_stats(out, f'{prefix}.upsample_out'))
    return out


def trace_sconcat(concat_mod, x_list, prefix, trace_log):
    """Trace SConcat on a list of spike tensor lists."""
    out = [torch.cat([n[i] for n in x_list], concat_mod.d) for i in range(spike.time_step)]
    trace_log.append(tensor_stats(out, f'{prefix}.concat_out'))
    return out


def trace_detect_branch(seq, x, prefix, trace_log):
    """Trace a cv2 or cv3 detection branch: SConv -> SConv -> SDConv."""
    out = x
    for i, sub in enumerate(seq):
        if isinstance(sub, SConv):
            out = trace_sconv(sub, out, f'{prefix}.sconv{i}', trace_log)
        elif isinstance(sub, SDConv):
            out = trace_sdconv(sub, out, f'{prefix}.sdconv{i}', trace_log)
        else:
            out = sub(out)
            trace_log.append(tensor_stats(out, f'{prefix}.sub{i}'))
    return out


def trace_sddetect(detect, x_scales, prefix, trace_log):
    """Trace SDDetect: cv2/cv3 per scale -> cat -> reshape -> DFL -> dist2bbox -> sigmoid."""
    shape = x_scales[0][0].shape

    scale_outputs = []
    for si in range(detect.nl):
        trace_log.append({'name': f'{prefix}.scale{si}_input',
                          'shape': list(x_scales[si][0].shape) if isinstance(x_scales[si], list) else list(x_scales[si].shape),
                          'note': f'Detection scale {si}'})

        # bbox regression branch
        box_out = trace_detect_branch(detect.cv2[si], x_scales[si],
                                       f'{prefix}.scale{si}.cv2_box', trace_log)
        # classification branch
        cls_out = trace_detect_branch(detect.cv3[si], x_scales[si],
                                       f'{prefix}.scale{si}.cv3_cls', trace_log)
        # Concat box + cls
        combined = torch.cat((box_out, cls_out), 1)
        trace_log.append(tensor_stats(combined, f'{prefix}.scale{si}.box_cls_cat'))
        scale_outputs.append(combined)

    # Anchors
    if detect.dynamic or detect.shape != shape:
        detect.anchors, detect.strides = (a.transpose(0, 1) for a in make_anchors(scale_outputs, detect.stride, 0.5))
        detect.shape = shape

    # Reshape + split
    all_cat = torch.cat([xi.view(shape[0], detect.no, -1) for xi in scale_outputs], 2)
    trace_log.append(tensor_stats(all_cat, f'{prefix}.all_scales_cat'))

    box, cls = all_cat.split((detect.reg_max * 4, detect.nc), 1)
    trace_log.append(tensor_stats(box, f'{prefix}.box_raw'))
    trace_log.append(tensor_stats(cls, f'{prefix}.cls_raw'))

    # DFL
    dfl_out = detect.dfl(box)
    trace_log.append(tensor_stats(dfl_out, f'{prefix}.dfl_out'))

    # dist2bbox
    dbox = dist2bbox(dfl_out, detect.anchors.unsqueeze(0), xywh=True, dim=1) * detect.strides
    trace_log.append(tensor_stats(dbox, f'{prefix}.decoded_boxes_xywh'))

    # Sigmoid on class scores
    cls_sig = cls.sigmoid()
    trace_log.append(tensor_stats(cls_sig, f'{prefix}.cls_sigmoid'))

    # Final output
    y = torch.cat((dbox, cls_sig), 1)
    trace_log.append(tensor_stats(y, f'{prefix}.final_output'))

    return y


# ---------------------------------------------------------------------------
# Top-level model tracer
# ---------------------------------------------------------------------------

def trace_full_model(model, img_tensor, device, trace_log):
    """Walk the image through every layer of the model, tracing sub-operations."""
    reset_net(model)

    y_cache = []  # layer outputs cache (mirrors _forward_once)
    x = img_tensor

    for idx, m in enumerate(model.model):
        layer_name = f'layer{idx}_{m.__class__.__name__}'
        trace_log.append({'name': f'=== {layer_name} ===', 'note': 'LAYER_START'})

        # Resolve input from previous layers
        if m.f != -1:
            if isinstance(m.f, int):
                x = y_cache[m.f]
            else:
                x = [x if j == -1 else y_cache[j] for j in m.f]

        trace_log.append(tensor_stats(x, f'{layer_name}.input'))

        # Dispatch to the appropriate tracer
        if isinstance(m, SGhostEncoderLite):
            x = trace_sghost_encoder_lite(m, x, layer_name, trace_log)

        elif isinstance(m, GhostBasicBlock1):
            x = trace_ghost_basic_block1(m, x, layer_name, trace_log)

        elif isinstance(m, GhostBasicBlock2):
            x = trace_ghost_basic_block2(m, x, layer_name, trace_log)

        elif isinstance(m, SGhostConv):
            x = trace_sghost_conv(m, x, layer_name, trace_log)

        elif isinstance(m, SConv):
            x = trace_sconv(m, x, layer_name, trace_log)

        elif isinstance(m, SUpsample):
            x = trace_supsample(m, x, layer_name, trace_log)

        elif isinstance(m, SConcat):
            x = trace_sconcat(m, x, layer_name, trace_log)

        elif isinstance(m, SDDetect):
            x_out = trace_sddetect(m, x, layer_name, trace_log)
            x = (x_out, x)  # match SDDetect return format

        else:
            x = m(x)
            trace_log.append(tensor_stats(x, f'{layer_name}.output'))

        trace_log.append(tensor_stats(x, f'{layer_name}.OUTPUT'))
        y_cache.append(x if m.i in model.save else None)

    return x


# ---------------------------------------------------------------------------
# Weight export
# ---------------------------------------------------------------------------

def dump_weights_int8(model, save_dir):
    """Export all Conv2d weights as INT8 with scale factors."""
    weight_dir = save_dir / 'weights_int8'
    weight_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and hasattr(mod, 'weight'):
            w = mod.weight.detach().float().cpu()
            scale = w.abs().max().item() / 127 if w.abs().max().item() > 0 else 1e-8
            w_int8 = (w / scale).round().clamp(-128, 127).to(torch.int8)

            safe_name = name.replace('.', '_')
            csv_path = weight_dir / f'{safe_name}.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f'# {name}, shape={list(w.shape)}, scale={scale:.8f}'])
                for oc in range(w_int8.shape[0]):
                    writer.writerow(w_int8[oc].flatten().tolist())

            manifest.append({
                'name': name,
                'shape': list(w.shape),
                'scale': round(scale, 8),
                'csv': str(csv_path.name),
            })

    with open(weight_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'  Exported {len(manifest)} conv layers to {weight_dir}/')
    return manifest


def dump_bn_params(model, save_dir):
    """Export all BN parameters for hardware fusion."""
    bn_dir = save_dir / 'bn_params'
    bn_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for name, mod in model.named_modules():
        if isinstance(mod, seBatchNorm):
            bn = mod.bn
            safe_name = name.replace('.', '_')
            data = {
                'name': name,
                'num_features': bn.num_features,
                'var_scale': getattr(bn, 'var_scale', 1.0),
                'eps': bn.eps,
                'running_mean': bn.running_mean.cpu().tolist(),
                'running_var': bn.running_var.cpu().tolist(),
            }
            if bn.weight is not None:
                data['gamma'] = bn.weight.cpu().tolist()
            if bn.bias is not None:
                data['beta'] = bn.bias.cpu().tolist()

            path = bn_dir / f'{safe_name}.json'
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            manifest.append({'name': name, 'file': str(path.name)})

    with open(bn_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'  Exported {len(manifest)} BN layers to {bn_dir}/')
    return manifest


# ---------------------------------------------------------------------------
# Preprocessing and display
# ---------------------------------------------------------------------------

def preprocess_image(img_path, imgsz=1920):
    """Load and preprocess an image, returning both raw and normalized tensors."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f'Cannot read image: {img_path}')

    raw_info = {
        'path': str(img_path),
        'original_shape': list(img_bgr.shape),
        'dtype': str(img_bgr.dtype),
        'range': [int(img_bgr.min()), int(img_bgr.max())],
    }

    # Resize with letterbox
    h0, w0 = img_bgr.shape[:2]
    r = imgsz / max(h0, w0)
    if r != 1:
        img_bgr = cv2.resize(img_bgr, (int(w0 * r), int(h0 * r)),
                              interpolation=cv2.INTER_LINEAR)

    # Pad to square
    h, w = img_bgr.shape[:2]
    dw, dh = imgsz - w, imgsz - h
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    img_bgr = cv2.copyMakeBorder(img_bgr, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # HWC BGR -> CHW RGB -> float32 / 255
    img_rgb = img_bgr[:, :, ::-1].copy()
    img_chw = img_rgb.transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img_chw).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # (1, 3, H, W)

    norm_info = {
        'shape': list(img_tensor.shape),
        'dtype': 'float32',
        'range': [round(img_tensor.min().item(), 4), round(img_tensor.max().item(), 4)],
        'mean': round(img_tensor.mean().item(), 4),
        'std': round(img_tensor.std().item(), 4),
    }

    return img_tensor, raw_info, norm_info


def print_trace(trace_log):
    """Pretty-print the trace log to console."""
    for entry in trace_log:
        if entry is None:
            continue
        name = entry.get('name', '')
        if 'LAYER_START' in entry.get('note', ''):
            print(f'\n{"=" * 80}')
            print(f'  {name}')
            print(f'{"=" * 80}')
            continue

        if 'shape' in entry:
            shape_str = str(entry['shape'])
            line = f'  {name:<55s} {shape_str:<25s}'
            if 'min' in entry:
                line += f' [{entry["min"]:.4f}, {entry["max"]:.4f}]'
            if 'mean' in entry:
                line += f'  mean={entry["mean"]:.4f}'
            if 'sparsity' in entry:
                line += f'  sparse={entry["sparsity"]:.1%}'
            if entry.get('is_binary'):
                line += '  BINARY'
            print(line)
        elif 'running_mean_range' in entry:
            print(f'  {name:<55s} BN: mean={entry["running_mean_range"]}, var={entry["running_var_range"]}')
        elif 'int8_scale' in entry:
            print(f'  {name:<55s} W: {entry["shape"]} scale={entry["int8_scale"]:.6f} [{entry["min"]:.4f}, {entry["max"]:.4f}]')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SNN Signal Tracer')
    parser.add_argument('--weights', type=str, required=True, help='Model weights (.pt)')
    parser.add_argument('--image', type=str, default='', help='Input image path (uses random if empty)')
    parser.add_argument('--img', type=int, default=1920, help='Image size')
    parser.add_argument('--device', default='0', help='CUDA device')
    parser.add_argument('--save-dir', type=str, default='runs/trace/', help='Output directory')
    parser.add_argument('--dump-weights', action='store_true', help='Export INT8 weights + BN params')
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f'cuda:{args.device}' if args.device.isdigit() and torch.cuda.is_available()
                          else 'cpu')

    # Load model
    spike.time_step = 1
    spike.set_time_step(1)
    print(f'Device: {device}')
    print(f'Time step: {spike.time_step}')

    ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
    model = (ckpt.get('ema') or ckpt['model']).float().to(device).eval()
    print(f'Model loaded: {sum(p.numel() for p in model.parameters()):,} params')

    # Prepare input
    if args.image:
        img_tensor, raw_info, norm_info = preprocess_image(args.image, args.img)
        print(f'\nInput image: {args.image}')
        print(f'  Raw:  {raw_info["original_shape"]} {raw_info["dtype"]} [{raw_info["range"][0]}, {raw_info["range"][1]}]')
        print(f'  Norm: {norm_info["shape"]} {norm_info["dtype"]} [{norm_info["range"][0]}, {norm_info["range"][1]}]')
    else:
        img_tensor = torch.randn(1, 3, args.img, args.img)
        raw_info = {'path': 'random_noise', 'original_shape': [args.img, args.img, 3]}
        norm_info = {'shape': list(img_tensor.shape)}
        print(f'\nUsing random input: {list(img_tensor.shape)}')

    img_tensor = img_tensor.to(device)

    # Run trace
    print(f'\n{"#" * 80}')
    print(f'  SIGNAL TRACE: Image -> Model -> Detections')
    print(f'{"#" * 80}')

    trace_log = []
    trace_log.append({'name': 'INPUT_RAW', **raw_info, 'note': 'Raw image before preprocessing'})
    trace_log.append({'name': 'INPUT_NORMALIZED', **norm_info, 'note': 'After /255, CHW RGB float32'})

    t0 = time.perf_counter()
    with torch.no_grad():
        output = trace_full_model(model, img_tensor, device, trace_log)
    elapsed = time.perf_counter() - t0

    print_trace(trace_log)

    print(f'\n{"=" * 80}')
    print(f'  Trace complete: {len(trace_log)} operations logged in {elapsed:.2f}s')
    print(f'{"=" * 80}')

    # Save trace to JSON
    json_safe_log = []
    for entry in trace_log:
        if entry is None:
            continue
        safe = {}
        for k, v in entry.items():
            if isinstance(v, (int, float, str, bool, list, type(None))):
                safe[k] = v
            else:
                safe[k] = str(v)
        json_safe_log.append(safe)

    with open(save_dir / 'trace_output.json', 'w') as f:
        json.dump(json_safe_log, f, indent=2)
    print(f'  Saved trace_output.json ({len(json_safe_log)} entries)')

    # Save tensor snapshots as NPZ
    snapshot = {}
    for entry in trace_log:
        if entry and 'center_5x5_ch0' in entry:
            key = entry['name'].replace('.', '_').replace(' ', '_')
            snapshot[key] = np.array(entry['center_5x5_ch0'])
    np.savez(save_dir / 'tensor_snapshots.npz', **snapshot)
    print(f'  Saved tensor_snapshots.npz ({len(snapshot)} patches)')

    # Dump weights + BN if requested
    if args.dump_weights:
        print(f'\n  Exporting weights and BN params...')
        dump_weights_int8(model, save_dir)
        dump_bn_params(model, save_dir)

    # Summary stats
    print(f'\n  --- Trace Summary ---')
    spike_ops = [e for e in trace_log if e and e.get('is_binary')]
    if spike_ops:
        avg_sparsity = np.mean([e['sparsity'] for e in spike_ops])
        print(f'  Total spike layers traced: {len(spike_ops)}')
        print(f'  Average spike sparsity: {avg_sparsity:.1%}')

    conv_ops = [e for e in trace_log if e and 'int8_scale' in e]
    print(f'  Total conv layers traced: {len(conv_ops)}')
    print(f'\n  All outputs saved to: {save_dir}')


if __name__ == '__main__':
    main()
