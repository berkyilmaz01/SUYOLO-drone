# SAVW Feasibility Report: Code-Grounded Diagnosis of Mid-Ghost Recall Saturation

## 1. Diagnosis Validation Against Actual Code

### Claim 1: "4 stages, later stages capped at 128 channels"

**CONFIRMED.** From `models/detect/su-yolo-720p-mid-ghost.yaml`:

```
Stage 0: SGhostEncoderLite  3 → 32ch  (P1/2)
Stage 1: GhostBasicBlock1  32 → 64ch  (P2/4)
Stage 2: GhostBasicBlock1  64 → 128ch (P3/8)
Stage 3: GhostBasicBlock1 128 → 128ch (P4/16)  ← no expansion
```

P3 and P4 share the same 128-channel budget. In a standard YOLO backbone
channels typically double at each stride (64→128→256→512). This model
freezes at 128 after P3, halving the representational budget at P4 compared
to a conventional design.

---

### Claim 2: "Ghost convolution splits into intrinsic + cheap depthwise"

**CONFIRMED.** From `models/spike.py:448-478`:

```python
c_intrinsic = math.ceil(c2 / ratio)   # ratio=2 → half
c_ghost     = c2 - c_intrinsic

self.primary = SConv(c1, c_intrinsic, k, s, ...)           # standard conv
self.cheap   = SConv(c_intrinsic, c_ghost, dw_k=3, ...,
                     groups=c_intrinsic)                    # depthwise
```

For a 128→128 ghost conv: intrinsic=64, ghost=64.  The cheap branch sees
*only its own 64 channels* through depthwise ops — no cross-channel mixing.
This is the primary mechanism that suppresses feature diversity for
low-signal targets.

---

### Claim 3: "Detection head compressed to 64 channels per scale"

**CONFIRMED, and WORSE than stated.** From `models/spike.py:305-317`:

```python
c2 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4)
c3 = max((ch[0], min((self.nc * 2, 128))))
```

With `ch[0]=64` (all three scales) and `reg_max=8, nc=10`:

| Branch | Formula | Internal channels |
|--------|---------|-------------------|
| Box (cv2) | max(64//4, 8×4, 16) = 32 | **32** |
| Cls (cv3) | max(64, min(20, 128)) = 64 | 64 |

The box regression branch operates at **32 channels** — only 32 learned
features to regress 4 coordinates across `reg_max=8` distribution bins.
For dense small-object VisDrone scenes, this is extremely tight.

Each scale's head is: `SConv(64→32) → SConv(32→32, g=4) → SDConv(32→32, g=4)`.
The grouped convolution (`g=4`) further restricts cross-channel interaction
to 8-channel groups.

---

### Claim 4: "obj: 0.7 may not sufficiently compensate for weak positives"

**PARTIALLY INCORRECT — `obj` is a dead parameter.**

The actual loss function in `utils/loss_tal.py` is TAL-based and **never
references `h["obj"]`**. The loss gains are hardcoded (lines 213-215):

```python
loss[0] *= 7.5   # box
loss[1] *= 0.5   # cls
loss[2] *= 1.5   # dfl
```

`obj: 0.7` in `hyp.visdrone.yaml` is vestigial from the legacy anchor-based
`utils/loss.py`. **Tuning `obj` will have zero effect on training.**

Actionable gains instead:
- `cls: 0.5` → increasing this directly boosts classification signal
- `fl_gamma: 0.0` → focal loss is **disabled**; enabling (γ=1.5) would
  reweight hard positives
- TAL `topk` (env `YOLOM`, default 10) → increasing to 13-15 assigns
  more positive anchors per GT

---

### Claim 5: "mosaic: 0.5"

**CONFIRMED.** From `data/hyps/hyp.visdrone.yaml:38`:
```yaml
mosaic: 0.5
```

This is reasonable for dense scenes but may underexpose the model to
challenging multi-scale compositions. Sweeping 0.5→0.7 is worth testing.

---

## 2. Critical Issue: `time_step=1` Breaks Temporal SAVW

**The recommended training command is:**
```
python train.py ... --time-step 1
```

At `time_step=1`, the model processes a **single temporal frame**. This means:

- `μ_spike` = single activation map (no temporal mean)
- `σ_spike` = 0 (undefined with n=1)
- `H_event` = 0 (no temporal entropy)

**SAVW as described — routing based on temporal spike statistics — cannot
work at time_step=1.**

### Alternatives:

| Approach | Pros | Cons |
|----------|------|------|
| **A. Use time_step≥2** | Enables temporal stats; aligns with SNN nature | ~2× compute; breaks ZCU102 latency target |
| **B. Spatial spike statistics** | Works at T=1; uses activation sparsity per spatial region | Less novel; closer to spatial attention |
| **C. Channel activation statistics** | Works at T=1; uses per-channel firing rate as routing signal | Loses temporal claim; but still input-conditional |
| **D. Hybrid: train T=4, deploy T=1** | Full temporal stats during training; distill routing to T=1 at deploy | Training cost; distillation complexity |

**Recommendation:** Option C or D. Option C redefines the spike summary as:

```
s = [sparsity_ratio, channel_entropy, max_activation]
```

computed per-block over the spatial dimensions of intrinsic features.
This preserves the "spike-adaptive" framing while working at T=1.

---

## 3. Integration Points for SAVW

### 3.1 Where to inject

**Target:** `SGhostConv` in `models/spike.py:448-478`

Inject between `self.primary(x)` and `self.cheap(p)`:

```python
def forward(self, x):
    p = self.primary(x)          # list[T] of [B, c_intrinsic, H, W]
    if self.savw is not None:
        p = self.savw(p)         # ← SAVW remaps intrinsic features
    if self.cheap is None:
        return p
    g = self.cheap(p)
    return [torch.cat([p[i], g[i]], 1) for i in range(time_step)]
```

### 3.2 Which blocks to target first

From the YAML, the most impactful blocks for recall are:

1. **Layer 3** (backbone P4/16): `GhostBasicBlock1(128→128)` — deepest
   backbone stage, most abstract features, 128→64 intrinsic channels
2. **Layer 7** (head P3/8-medium): `GhostBasicBlock2(192→64)` — first
   FPN fusion, highest concat channel count
3. **Layer 10** (head P2/4-small): `GhostBasicBlock2(128→64)` — small
   object detection scale

Start with layer 3 only, then expand.

### 3.3 Parameter budget

Per SAVW-augmented SGhostConv (c_intrinsic=64, rank=4, K=3 experts):

```
Virtual remap bank: K × 2 × c × r = 3 × 2 × 64 × 4 =  1,536 params
Router MLP:         3 → 16 → K   = 3×16 + 16×3       =     96 params
                                              Total:     1,632 params
```

With ~10 SGhostConv blocks eligible: **~16K params total** (<3% overhead
on a ~0.5M parameter model).

### 3.4 YAML integration

Add to `models/yolo.py:708` alongside existing ghost modules:

```python
SSAVWGhostConv,    # ← new token
```

Example YAML change (layer 3 only):
```yaml
# Before
[-1, 1, GhostBasicBlock1, [128, 128, 64, 1]],  # 3-P4/16
# After
[-1, 1, SAVWGhostBasicBlock1, [128, 128, 64, 1]],  # 3-P4/16
```

### 3.5 Loss integration for PCL

In `utils/loss_tal.py`, after the TAL assigner returns `fg_mask` (line 194):

```python
# Positive Coverage Loss: penalize GTs with zero positive anchors
gt_has_pos = (mask_pos.sum(-1) > 0).float()  # (b, n_max_boxes)
coverage = (gt_has_pos * mask_gt.squeeze(-1)).sum() / mask_gt.sum().clamp(1)
loss_pcl = 1.0 - coverage
```

This is cheap to compute and directly targets recall.

---

## 4. Immediate Low-Hanging Fruit (No SAVW Required)

Before implementing SAVW, these code-level changes can improve recall:

### 4.1 Enable focal loss

```yaml
# hyp.visdrone.yaml
fl_gamma: 1.5  # was 0.0 (disabled)
```

This reweights hard positives (faint/small objects) in the BCE
classification loss.

### 4.2 Increase TAL topk

```bash
export YOLOM=15   # was 10
```

More positive anchor candidates per GT → higher recall ceiling.

### 4.3 Increase cls loss gain

```python
# loss_tal.py line 214
loss[1] *= 1.0   # was 0.5
```

Classification is under-weighted relative to box (7.5) and DFL (1.5).
Weak positives need stronger gradient signal.

### 4.4 Widen box head

In `SDDetect.__init__` (spike.py:314), the box branch c2 is:
```python
c2 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4)
```

For 64ch input: `max(16, 32, 16) = 32`. Changing to `ch[0] // 2` gives
c2=32 still, but with `ch[0] // 3` → `make_divisible(21, 4) = 24` which
is even worse. The real fix is increasing input to SDDetect beyond 64ch
(e.g., 96ch in head GhostBasicBlock2 outputs).

---

## 5. Summary Assessment

| Aspect | Status |
|--------|--------|
| Diagnosis of backbone bottleneck | Correct |
| Diagnosis of head compression | Correct (actually worse — 32ch box head) |
| Diagnosis of ghost channel suppression | Correct |
| `obj: 0.7` tuning suggestion | **Wrong** — `obj` is unused in TAL loss |
| SAVW concept (temporal routing) | Sound in theory, **broken at T=1** |
| SAVW with spatial stats (adapted) | Feasible, minimal parameter overhead |
| PCL loss addition | Straightforward, compatible with TAL |
| YAML/parser integration | Clean, follows existing patterns |
| Parameter budget | ~16K params, well within constraints |
| Novelty claim | Plausible but needs literature check on spatial-conditioned ghost routing |

### Verdict

The diagnosis is **mostly correct** with one significant error (`obj` is
dead) and one critical implementation gap (T=1 kills temporal statistics).

SAVW is worth pursuing if adapted to **spatial/channel activation
statistics** rather than temporal spike statistics. The integration path
is clean. Combine with the low-hanging fruit (focal loss, topk, cls gain)
first to establish a stronger baseline before ablating SAVW.

The strongest publishable angle is:
> "Input-adaptive virtual channel expansion for efficient SNN detection,
> conditioned on activation sparsity patterns in ghost convolution branches."

This avoids the temporal statistics dependency while preserving the core
novelty of parameter-tied dynamic width under fixed memory budget.
