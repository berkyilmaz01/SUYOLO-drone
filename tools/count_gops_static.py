"""Count GOPS by statically walking Conv2d layers and tracking spatial dims.

Does NOT run a forward pass — avoids seBatchNorm bug #11.
Instead, builds the model and estimates GOPS from Conv2d weight shapes
and known spatial resolution flow through the architecture.

Usage:
  python tools/count_gops_static.py --cfg models/detect/su-yolo-720p-mid.yaml --img 1920 1920 --time-step 1
  python tools/count_gops_static.py --cfg models/detect/su-yolo-720p-mid-ghost.yaml --img 1920 1920 --time-step 1
"""
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from models.yolo import DetectionModel
from models.spike import set_time_step


def count_gops(cfg, imgsz, time_step_val):
    set_time_step(time_step_val)
    model = DetectionModel(cfg, ch=3, nc=10, imgsz=max(imgsz))

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Collect all Conv2d layers with their weight shapes
    total_macs = 0
    H, W = imgsz

    # Walk through model.model (the sequential layers)
    # Track spatial resolution at each layer
    spatial = {}  # layer_idx -> (H, W)

    for i, m in enumerate(model.model):
        module_name = m.__class__.__name__

        # Determine input spatial resolution
        if i == 0:
            inp_h, inp_w = H, W
        else:
            # Use from_idx if available
            f = m.f if hasattr(m, 'f') else -1
            if isinstance(f, list):
                inp_h, inp_w = spatial[f[0]]
            elif isinstance(f, int):
                src = i + f if f < 0 else f
                inp_h, inp_w = spatial[src]
            else:
                inp_h, inp_w = spatial[i - 1]

        # Count MACs from all Conv2d inside this module
        layer_macs = 0
        cur_h, cur_w = inp_h, inp_w

        for name, sub in m.named_modules():
            if isinstance(sub, nn.Conv2d):
                Cout, Cin_per_g, Kh, Kw = sub.weight.shape
                groups = sub.groups
                stride = sub.stride if isinstance(sub.stride, tuple) else (sub.stride, sub.stride)
                # Estimate output spatial from stride
                out_h = cur_h // stride[0]
                out_w = cur_w // stride[1]
                macs = Kh * Kw * Cin_per_g * Cout * out_h * out_w
                layer_macs += macs

        total_macs += layer_macs

        # Determine output spatial for this layer
        # Encoder/BasicBlock1 have stride 2, BasicBlock2/SConv depends on config
        out_h, out_w = inp_h, inp_w

        if 'Encoder' in module_name:
            out_h, out_w = inp_h // 2, inp_w // 2
        elif 'BasicBlock1' in module_name or 'GhostBasicBlock1' in module_name:
            out_h, out_w = inp_h // 2, inp_w // 2
        elif 'BasicBlock2' in module_name or 'GhostBasicBlock2' in module_name:
            out_h, out_w = inp_h, inp_w
        elif 'Upsample' in module_name:
            out_h, out_w = inp_h * 2, inp_w * 2
        elif 'Concat' in module_name:
            out_h, out_w = inp_h, inp_w
        elif 'Detect' in module_name:
            out_h, out_w = inp_h, inp_w
        else:
            # SConv / SGhostConv — check for stride in args
            for name, sub in m.named_modules():
                if isinstance(sub, nn.Conv2d):
                    stride = sub.stride if isinstance(sub.stride, tuple) else (sub.stride, sub.stride)
                    if stride[0] == 2:
                        out_h, out_w = inp_h // 2, inp_w // 2
                    break

        spatial[i] = (out_h, out_w)

        gops_layer = 2 * layer_macs / 1e9
        if gops_layer > 0.001:
            print(f"  Layer {i:2d} {module_name:30s} {str(inp_h)+'x'+str(inp_w):>10s} -> {str(out_h)+'x'+str(out_w):>10s}  {gops_layer:8.3f} GOPS")

    total_gops = 2 * total_macs / 1e9

    print(f"\n{'='*60}")
    print(f"Model:      {cfg}")
    print(f"Resolution: {imgsz[0]}x{imgsz[1]}")
    print(f"Time step:  {time_step_val}")
    print(f"Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable:  {trainable:,}")
    print(f"Total GOPS: {total_gops:.2f}")
    print(f"{'='*60}")

    # ZCU102 estimate
    dpu_tops = 3.45
    utilization = 0.60
    effective_gops = dpu_tops * utilization * 1000
    est_fps = effective_gops / total_gops if total_gops > 0 else 0
    print(f"\nZCU102 (3xB4096 @ 60% util): ~{est_fps:.1f} FPS")

    return total_gops, total_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--img', nargs='+', type=int, default=[1920, 1920])
    parser.add_argument('--time-step', type=int, default=1)
    args = parser.parse_args()
    imgsz = args.img if len(args.img) == 2 else [args.img[0], args.img[0]]
    count_gops(args.cfg, imgsz, args.time_step)
