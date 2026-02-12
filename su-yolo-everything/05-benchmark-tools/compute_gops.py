"""Compute estimated GOPS for SU-YOLO model variants at given resolution.

Usage: python tools/compute_gops.py --cfg models/detect/su-yolo-720p-nano.yaml --img 736 1280
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from models.yolo import DetectionModel
from models.spike import set_time_step


def count_conv_gops(module, input_size):
    """Count GOPS for a Conv2d layer given input spatial size (H, W)."""
    H, W = input_size
    if hasattr(module, 'weight'):
        Cout, Cin_per_g, Kh, Kw = module.weight.shape
        groups = getattr(module, 'groups', 1)
        Cin = Cin_per_g * groups
        stride = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
        Ho = H // stride[0]
        Wo = W // stride[1]
        # MACs = Kh * Kw * Cin * Cout / groups * Ho * Wo
        macs = Kh * Kw * Cin_per_g * Cout * Ho * Wo
        return 2 * macs / 1e9  # GOPS (multiply-accumulate = 2 ops)
    return 0


def profile_model(cfg, imgsz, time_step_val):
    set_time_step(time_step_val)
    model = DetectionModel(cfg, ch=3, nc=10, imgsz=max(imgsz))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {cfg}")
    print(f"Image size: {imgsz[0]}x{imgsz[1]}")
    print(f"Time step: {time_step_val}")
    print(f"Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # Count FLOPs via forward hooks (thop breaks on SNN IFNode layers)
    from spikingjelly.activation_based.functional import reset_net
    import torch.nn as nn

    total_macs = [0]  # mutable to allow hook access
    hooks = []

    def make_hook(mod):
        def hook_fn(m, inp, out):
            if isinstance(m, nn.Conv2d):
                # inp[0] shape: [B, C_in, H, W]
                _, _, H, W = inp[0].shape
                Cout, Cin_per_g, Kh, Kw = m.weight.shape
                s = m.stride if isinstance(m.stride, tuple) else (m.stride, m.stride)
                Ho = H // s[0]
                Wo = W // s[1]
                total_macs[0] += Kh * Kw * Cin_per_g * Cout * Ho * Wo
            elif isinstance(m, nn.Linear):
                total_macs[0] += m.in_features * m.out_features
        return hook_fn

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(make_hook(m)))

    inp = torch.randn(1, 3, imgsz[0], imgsz[1])
    reset_net(model)
    with torch.no_grad():
        model(inp)

    for h in hooks:
        h.remove()

    gflops = total_macs[0] * 2 / 1e9  # MACs × 2 = FLOPs, then /1e9 = GFLOPs
    gops = gflops
    print(f"GFLOPs: {gflops:.2f}")
    print(f"GFLOPs per time_step: {gflops:.2f} (time_step={time_step_val})")

    # ZCU102 FPS estimate
    # Conservative: assume 60% DPU utilization due to memory bandwidth
    dpu_tops = 3.45  # INT8 TOPS for 3x B4096 @ 281MHz
    utilization = 0.60  # memory bandwidth limited
    effective_tops = dpu_tops * utilization * 1000  # convert to GOPS
    print(f"\n--- ZCU102 Estimate (3x B4096 @ 281MHz) ---")
    print(f"DPU peak: {dpu_tops} TOPS | Effective (60% util): {effective_tops:.0f} GOPS")
    print(f"Parameters: {total_params/1e6:.2f}M")

    try:
        est_fps = effective_tops / gops
        print(f"Estimated FPS: {est_fps:.1f}")
        print(f"{'✓ MEETS 60 FPS TARGET' if est_fps >= 60 else '✗ BELOW 60 FPS TARGET'}")
    except NameError:
        print("Install thop for FPS estimate: pip install thop")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='model yaml path')
    parser.add_argument('--img', nargs='+', type=int, default=[736, 1280], help='image size h w')
    parser.add_argument('--time-step', type=int, default=1, help='SNN time steps')
    args = parser.parse_args()
    imgsz = args.img if len(args.img) == 2 else [args.img[0], args.img[0]]
    profile_model(args.cfg, imgsz, args.time_step)
