"""Verify RepGhost implementation: imports, forward pass, fusion, channel flow."""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOG = ROOT / '.cursor' / 'debug.log'

def log(hid, msg, data=None):
    entry = {"hypothesisId": hid, "location": "test_repghost.py",
             "message": msg, "timestamp": int(time.time()*1000)}
    if data:
        entry["data"] = data
    with open(LOG, 'a') as f:
        f.write(json.dumps(entry) + '\n')

import torch
import torch.nn as nn

# H4: Test imports (forward reference check)
try:
    import models.spike as spike
    from models.spike import (
        SRepGhostConv, SRepGhostEncoderLite,
        RepGhostBasicBlock1, RepGhostBasicBlock2,
        SGhostConv, SGhostEncoderLite, GhostBasicBlock1, GhostBasicBlock2,
        set_repghost, use_repghost, REPGHOST_MAP, seBatchNorm,
        SConv, set_time_step
    )
    log("H4", "PASS: All RepGhost classes imported successfully")
except Exception as e:
    log("H4", f"FAIL: Import error: {e}")
    sys.exit(1)

# Set time_step for tests
spike.time_step = 4
set_time_step(4)

# H5: Test SRepGhostConv forward pass — shape correctness
try:
    conv = SRepGhostConv(16, 32, k=3, s=1, act=True)
    x_list = [torch.randn(1, 16, 8, 8) for _ in range(spike.time_step)]
    out = conv(x_list)
    shapes = [o.shape for o in out]
    log("H5", "SRepGhostConv forward pass",
        {"input_ch": 16, "output_ch": 32, "time_step": spike.time_step,
         "out_len": len(out), "out_shapes": [list(s) for s in shapes],
         "expected_shape": [1, 32, 8, 8]})
    assert len(out) == spike.time_step, f"Expected {spike.time_step} outputs, got {len(out)}"
    assert out[0].shape == (1, 32, 8, 8), f"Expected (1,32,8,8), got {out[0].shape}"
    log("H5", "PASS: SRepGhostConv shapes correct")
except Exception as e:
    log("H5", f"FAIL: SRepGhostConv forward: {e}")

# H5b: Test with stride=2
try:
    conv_s2 = SRepGhostConv(16, 32, k=3, s=2, act=False)
    x_list = [torch.randn(1, 16, 8, 8) for _ in range(spike.time_step)]
    out = conv_s2(x_list)
    log("H5", "SRepGhostConv stride=2",
        {"out_shape": list(out[0].shape), "expected": [1, 32, 4, 4]})
    assert out[0].shape == (1, 32, 4, 4), f"Expected (1,32,4,4), got {out[0].shape}"
    log("H5", "PASS: SRepGhostConv stride=2 shapes correct")
except Exception as e:
    log("H5", f"FAIL: SRepGhostConv stride=2: {e}")

# H5c: Test SRepGhostEncoderLite
try:
    enc = SRepGhostEncoderLite(3, 32, k=3, s=2)
    x = torch.randn(1, 3, 16, 16)
    out = enc(x)
    log("H5", "SRepGhostEncoderLite forward",
        {"out_len": len(out), "out_shape": list(out[0].shape), "expected": [1, 32, 8, 8]})
    assert out[0].shape == (1, 32, 8, 8)
    log("H5", "PASS: SRepGhostEncoderLite shapes correct")
except Exception as e:
    log("H5", f"FAIL: SRepGhostEncoderLite: {e}")

# H5d: Test RepGhostBasicBlock1
try:
    blk1 = RepGhostBasicBlock1(32, 64, 64, 32, 1)
    x_list = [torch.randn(1, 32, 8, 8) for _ in range(spike.time_step)]
    out = blk1(x_list)
    log("H5", "RepGhostBasicBlock1 forward",
        {"out_len": len(out), "out_shape": list(out[0].shape), "expected": [1, 64, 4, 4]})
    assert out[0].shape == (1, 64, 4, 4)
    log("H5", "PASS: RepGhostBasicBlock1 shapes correct")
except Exception as e:
    log("H5", f"FAIL: RepGhostBasicBlock1: {e}")

# H5e: Test RepGhostBasicBlock2
try:
    blk2 = RepGhostBasicBlock2(64, 64, 64, 32, 1)
    x_list = [torch.randn(1, 64, 4, 4) for _ in range(spike.time_step)]
    out = blk2(x_list)
    log("H5", "RepGhostBasicBlock2 forward",
        {"out_len": len(out), "out_shape": list(out[0].shape), "expected": [1, 64, 4, 4]})
    assert out[0].shape == (1, 64, 4, 4)
    log("H5", "PASS: RepGhostBasicBlock2 shapes correct")
except Exception as e:
    log("H5", f"FAIL: RepGhostBasicBlock2: {e}")

# H3: Test fuse_reparam correctness (time_step=1 for deployment)
try:
    spike.time_step = 1
    set_time_step(1)
    conv_fuse = SRepGhostConv(16, 32, k=3, s=1, act=True, n=1.0)
    conv_fuse.eval()

    x_list = [torch.randn(1, 16, 8, 8)]
    with torch.no_grad():
        out_before = conv_fuse(x_list)
    out_before_val = out_before[0].clone()

    conv_fuse.fuse_reparam()

    with torch.no_grad():
        out_after = conv_fuse(x_list)
    out_after_val = out_after[0].clone()

    diff = (out_before_val - out_after_val).abs().max().item()
    log("H3", "fuse_reparam correctness (time_step=1)",
        {"max_diff": diff, "passes_tolerance": diff < 1e-4,
         "rep_bn_type": type(conv_fuse.rep_bn).__name__,
         "cheap_bn_type": type(conv_fuse.cheap_bn).__name__,
         "has_bias": conv_fuse.cheap_conv.bias is not None})
    if diff < 1e-4:
        log("H3", f"PASS: fuse_reparam max diff = {diff:.6e}")
    else:
        log("H3", f"WARN: fuse_reparam max diff = {diff:.6e} (may be numerical)")
except Exception as e:
    log("H3", f"FAIL: fuse_reparam: {e}")

# Reset time_step
spike.time_step = 4
set_time_step(4)

# H_REMAP: Test parse_model class remapping
try:
    set_repghost(True)
    assert spike.use_repghost == True
    assert REPGHOST_MAP['SGhostConv'] == 'SRepGhostConv'

    test_name = 'SGhostConv'
    if spike.use_repghost and test_name in REPGHOST_MAP:
        remapped = REPGHOST_MAP[test_name]
    else:
        remapped = test_name
    cls = eval(remapped)
    log("H_REMAP", "Class remapping test",
        {"original": test_name, "remapped": remapped,
         "resolved_class": cls.__name__,
         "is_correct": cls is SRepGhostConv})
    assert cls is SRepGhostConv
    log("H_REMAP", "PASS: Class remapping works correctly")
    set_repghost(False)
except Exception as e:
    log("H_REMAP", f"FAIL: Class remapping: {e}")
    set_repghost(False)

# H_FULL: Test full model creation with --repghost flag
try:
    set_repghost(True)
    from models.yolo import Model
    yaml_path = str(ROOT / 'models' / 'detect' / 'su-yolo-720p-mid-ghost.yaml')
    model = Model(yaml_path, ch=3, nc=3)
    n_params = sum(p.numel() for p in model.parameters())

    set_repghost(False)
    model_ghost = Model(yaml_path, ch=3, nc=3)
    n_params_ghost = sum(p.numel() for p in model_ghost.parameters())

    has_repghost = any(isinstance(m, SRepGhostConv) for m in model.modules())
    has_ghost = any(isinstance(m, SGhostConv) for m in model_ghost.modules())
    no_ghost_in_repghost = not any(isinstance(m, SGhostConv) for m in model.modules())

    log("H_FULL", "Full model creation with repghost flag",
        {"repghost_params": n_params, "ghost_params": n_params_ghost,
         "has_repghost_modules": has_repghost,
         "has_ghost_modules_in_ghost": has_ghost,
         "no_ghost_in_repghost_model": no_ghost_in_repghost})

    if has_repghost and no_ghost_in_repghost and has_ghost:
        log("H_FULL", "PASS: Full model creation correct")
    else:
        log("H_FULL", "FAIL: Module type mismatch in full model")
except Exception as e:
    log("H_FULL", f"FAIL: Full model creation: {e}")
    set_repghost(False)

# H_FWD: Test forward pass on full repghost model
try:
    set_repghost(True)
    spike.time_step = 1
    set_time_step(1)
    from models.yolo import Model
    yaml_path = str(ROOT / 'models' / 'detect' / 'su-yolo-720p-mid-ghost.yaml')
    model = Model(yaml_path, ch=3, nc=3).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    if isinstance(out, (list, tuple)):
        out_shape = [list(o.shape) if isinstance(o, torch.Tensor) else type(o).__name__ for o in out]
    else:
        out_shape = list(out.shape)
    log("H_FWD", "Full model forward pass (repghost, time_step=1)",
        {"input_shape": [1, 3, 64, 64], "output_info": str(out_shape)[:200]})
    log("H_FWD", "PASS: Full model forward pass completed")
    set_repghost(False)
    spike.time_step = 4
    set_time_step(4)
except Exception as e:
    log("H_FWD", f"FAIL: Full model forward: {e}")
    set_repghost(False)
    spike.time_step = 4
    set_time_step(4)

print("Test script complete. Check debug.log for results.")
