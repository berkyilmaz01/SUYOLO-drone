# SU-YOLO Comprehensive Code Review

Every file audited. Findings verified and cross-referenced to eliminate false positives.

---

## CRITICAL BUGS (will cause errors/NaN when triggered)

### 1. `ComputeLossLH` division by zero — `loss_tal_dual.py:354`
```python
target_scores_sum = target_scores.sum()          # NOT clamped — can be 0
loss[1] = ... / target_scores_sum                 # NaN
```
Compare with `ComputeLoss` (same file, line 217): `max(target_scores.sum(), 1)`.
When a batch has no positive foreground assignments (early training, sparse labels), `target_scores_sum=0` → NaN → corrupts all weights silently.

### 2. TRT_NMS trailing commas create tuples — `experimental.py:190-191`
```python
self.background_class = -1,   # (-1,) tuple, not int -1
self.box_coding = 1,          # (1,) tuple, not int 1
```
`TRT_NMS.apply()` receives tuples instead of ints → TensorRT export fails.

### 3. Missing `import random` — `experimental.py:80`
```python
num_det = random.randint(0, 100)   # NameError — random not imported
```
`ORT_NMS.forward()` crashes. Only hits ONNX ORT export path.

### 4. NMS autolabel tensor width mismatch — `general.py:938`
```python
v = torch.zeros((len(lb), nc + nm + 5), ...)   # 5 should be 4
x = torch.cat((x, v), 0)                         # x has nc+nm+4 cols → RuntimeError
```
Autolabelling path only. Normal inference unaffected.

---

## MEDIUM BUGS (correctness/performance issues)

### 5. `detect.py:119` selects wrong head for TripleDetect
```python
pred = pred[0][1] if isinstance(pred[0], list) else pred[0]
```
For `TripleDetect`, `pred[0]` is `[y1, y2, y3]`. Index `[1]` selects y2 (2nd head), not y3 (lead head). Should be `[-1]`. Current configs use SDDetect, so not triggered.

### 6. val.py vs detect.py select different heads for dual models
- val.py path: `non_max_suppression` takes `prediction[0]` → **first** head
- detect.py path: `pred[0][1]` → **second** head
Training metrics and inference use different heads → metrics unreliable.

### 7. `model2` wastes GPU memory on DDP ranks — `train.py:126,134`
`model2` is constructed on ALL ranks but EMA only runs on RANK 0. Non-zero ranks waste memory equal to full model size.

### 8. Determinism permanently broken — `train.py:334`
```python
torch.use_deterministic_algorithms(False)   # inside batch loop, global
```
Called every batch inside training loop. Nullifies the `deterministic=True` from line 109. Should be called once before the loop if needed.

### 9. EMA precision degradation with AMP — `val.py:121,330`
```python
model.half()    # ← ema.ema (= model2) loses FP32 precision
...
model.float()   # ← FP16→FP32 round-trip is lossy
```
Since `ema.ema` IS `model2` (no deepcopy), each validation epoch permanently degrades EMA weights when AMP is enabled. Effect is small per epoch but accumulates.

### 10. `loss_tal_triple.py` missing `inner_iou_ratio` — lines 63, 75
`BboxLoss` in triple loss file lacks the `inner_iou_ratio` parameter present in `loss_tal.py` and `loss_tal_dual.py`. Inner-IoU silently disabled for triple-head models.

### 11. `seBatchNorm` bakes time_step at construction — `spike.py:445`
```python
self.bn = SeBatchNorm2d(t * c, n)   # t = time_step at construction
```
But forward uses the global `time_step`. If `--time-step` at inference differs from training, BN dimension mismatch → crash with confusing error.

### 12. Evolve CSV keys mislabeled — `train.py:649-650`
```python
keys = (..., 'val/box_loss', 'val/obj_loss', 'val/cls_loss')
```
Actual loss components are `(box, cls, dfl)` not `(box, obj, cls)`. Columns labeled `obj` contain cls, `cls` contains dfl.

### 13. `attempt_load` misses SDDetect compatibility — `experimental.py:255-258`
Only updates `Detect` and `Model` modules. `SDDetect`, `DDetect`, `DualDetect` etc. don't get `inplace` attribute set.

---

## LOW (dead code, cosmetic, quality)

### 14. `SDConv.bn` and `SDConv.act` allocated but unused — `spike.py:131-134`
### 15. `BasicBlock1.act1` allocated but unused — `spike.py:240`
### 16. `BasicBlock2.act1` allocated but unused — `spike.py:273`
### 17. `c3` param unused in BasicBlock1/2 — accepted but ignored
### 18. `Denoise/FindPoints` dead code (~100 lines) — `spike.py:344-440`
### 19. `loss_tal.py` dead code: `self.cp`, `self.cn`, `self.balance`, `VarifocalLoss`
### 20. `fuse()` is a no-op — `yolo.py:518-531`
### 21. Hardcoded `gs=32` — `train.py:150`
### 22. Default `--source` is empty — `detect.py:247`
### 23. `smart_optimizer` filter `'p' in param_name` too broad — `torch_utils.py:347`

---

## FALSE POSITIVES (verified as NOT bugs)

- **"ModelEMA corrupts training model"**: FALSE. `ModelEMA(model2)` wraps model2, not the training model. `ema.update(model)` reads model state_dict and writes to model2. Training model is never aliased.
- **"seBatchNorm unbind returns tuple"**: Works fine — callers use indexing `out[i]`, not `.append()`.
- **"reset_net placement"**: All 3 files (train/val/detect) call reset_net correctly before each forward pass.
- **"Checkpoint save/load format"**: Correctly saves model+ema, attempt_load prefers ema, strip_optimizer promotes ema→model.
- **"ComputeLoss format compatibility"**: `feats = p[1] if isinstance(p, tuple) else p` handles both eval and train SDDetect outputs.

---

## WHAT WORKS (confirmed correct)

- SNN temporal unfolding: IFNode neurons with ATan surrogate + SeBatchNorm
- All 10 YAML model configs (5 normal + 5 ghost): channel math verified end-to-end
- TAL loss (CIoU + DFL + BCE cls) with TaskAlignedAssigner
- SDDetect spike→tensor bridge: SDConv averages time steps correctly
- ByteTracker: Kalman filter, IoU association, track lifecycle
- Warmup, gradient clipping, AMP, cosine LR
- DDP mode (aside from model2 memory waste)
- Vitis AI / FPGA export path for SDDetect models

---

## GHOST CONVOLUTION INTEGRATION

Added as a fully separate architecture. Normal models are UNCHANGED.

### New modules in `models/spike.py`:
- `SGhostConv` — Primary SConv (half channels) + cheap depthwise SConv → cat
- `SGhostEncoder` — Ghost version of SEncoder
- `SGhostEncoderLite` — Ghost version of SEncoderLite
- `GhostBasicBlock1` — Ghost CSP-ELAN with stride 2
- `GhostBasicBlock2` — Ghost CSP-ELAN without stride
- `GhostTransitionBlock` — Ghost SPP-ELAN

### New YAML configs in `models/detect/`:
| Config | Base | Estimated GOPS |
|--------|------|---------------|
| `su-yolo-ghost.yaml` | `su-yolo.yaml` | ~20 (vs ~40) |
| `su-yolo-720p-ghost.yaml` | `su-yolo-720p.yaml` | ~20 (vs ~40) |
| `su-yolo-720p-nano-ghost.yaml` | `su-yolo-720p-nano.yaml` | ~3.4 (vs ~4.5) |
| `su-yolo-720p-nano-v2-ghost.yaml` | `su-yolo-720p-nano-v2.yaml` | ~4.5 (vs ~7.0) |
| `su-yolo-720p-mid-ghost.yaml` | `su-yolo-720p-mid.yaml` | ~9 (vs ~15) |

### How to train:
```bash
# Normal model (unchanged)
python train.py --cfg models/detect/su-yolo-720p-nano.yaml --data data/visdrone.yaml

# Ghost model (same command, different config)
python train.py --cfg models/detect/su-yolo-720p-nano-ghost.yaml --data data/visdrone.yaml
```
