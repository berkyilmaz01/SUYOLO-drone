# Paper Proposal: Membrane-Guided Ghost Convolutions for SNN Object Detection

## Target Venue: ECCV / CVPR / NeurIPS (Q1)

---

## Title Options

1. **"Membrane-Guided Ghost: Recovering Sub-Threshold Features for
   Efficient Spiking Object Detection"**
2. **"Ghost in the Membrane: Information-Theoretic Repair of Efficient
   SNN Architectures"**
3. **"Breaking the Recall Barrier: Membrane Potential Forwarding in
   Ghost-Compressed Spiking Detectors"**

---

## Abstract (Draft)

Efficient spiking neural network (SNN) object detectors increasingly
rely on ghost convolutions to reduce computational cost. However, we
identify a fundamental mismatch: ghost convolutions generate diverse
features through cheap depthwise transforms of rich continuous inputs,
but in SNNs these inputs are binary spikes — limiting the cheap branch
to at most k²+1 discrete output levels per spatial location. We prove
this "spike-ghost information collapse" loses O(log k²) bits versus
continuous networks and show it directly explains recall saturation in
dense-scene detection. We propose Membrane-Guided Ghost (MGG)
convolutions, which forward already-computed pre-threshold membrane
potentials to ghost cheap branches — restoring continuous feature
diversity at zero parameter cost. Combined with a recall-targeted
Positive Coverage Loss (PCL), MGG breaks through the recall ceiling on
VisDrone drone surveillance (+X% AR) while maintaining the energy
budget for neuromorphic deployment. To the best of our knowledge, this
is the first work to analyze and repair the interaction between ghost
compression and binary spike information loss.

---

## 1. The Core Insight (Why This Paper Exists)

### 1.1 The Problem Nobody Has Noticed

Ghost convolutions (Han et al., CVPR 2020) generate output features
via two branches:
- **Primary**: Standard convolution producing `c_intrinsic` channels
- **Cheap**: Depthwise convolution on primary output → `c_ghost` channels

In continuous networks, the cheap branch transforms rich float32
features → diverse outputs. **In SNNs, the primary branch outputs
binary spikes {0, 1}.**

A 3×3 depthwise convolution on binary inputs can produce **at most 10
distinct values** (sum of 9 binary pixels: 0,1,2,...,9). After BN and
IFNode re-thresholding, this collapses back to binary. The ghost cheap
branch in an SNN generates dramatically less feature diversity than in
a continuous network.

### 1.2 Quantified: Information Capacity per Feature Element

| Input type | Depthwise 3×3 output space | Effective bits |
|------------|---------------------------|----------------|
| Continuous (float32) | ℝ (infinite) | ~10-16 effective bits |
| Binary spike {0,1} | {0,1,...,9} | log₂(10) ≈ 3.32 bits |

**The ghost cheap branch in an SNN produces ~3-5× fewer effective bits
per spatial location than in a continuous network.**

### 1.3 Why This Causes Recall Saturation

In SU-YOLO's Mid-Ghost architecture, the information flow is:

```
Image → Encoder → [Binary] → Block1 → [Binary] → Block2 → [Binary]
   → Block3 → [Binary] → FPN → [Binary] → DetHead → [Continuous]
```

Faint objects produce near-threshold activations. At each IFNode
boundary, sub-threshold activations → 0 (indistinguishable from
"nothing here"). With N serial binary bottleneck points:

  P(object survives) = p₁ × p₂ × ... × pₙ

For a faint target with per-gate survival p=0.8 through N=12 gates:
P(survive) = 0.8¹² ≈ 0.069 — **only ~7% recall for marginal objects**.

Ghost compression makes this worse: fewer intrinsic channels → less
redundancy → lower per-gate survival probability.

### 1.4 Where Exactly In Our Code

Verified against SU-YOLO codebase (`models/spike.py`):

| Location | `act=` | Cheap branch input | Bottleneck? |
|----------|--------|-------------------|-------------|
| `SGhostEncoderLite.conv2` (stem, L514) | True | Binary spikes | **YES** |
| `GhostBasicBlock1.cvres/cv0/cv2` (L528-530) | False | Continuous (from own primary) | No |
| `GhostBasicBlock2.cv0/cv2` (L565-566) | False | Continuous (from own primary) | No |
| Head layer 4: standalone `SGhostConv` | True (default) | Binary spikes | **YES** |
| Head layer 11: standalone `SGhostConv` | True (default) | Binary spikes | **YES** |
| Head layer 14: standalone `SGhostConv` | True (default) | Binary spikes | **YES** |
| Block-level `act2/act3` IFNodes | — | — | **YES** (all blocks) |

The binary bottleneck hits at **4 standalone ghost convolution points**
(stem + 3 FPN junctions) AND at **2 IFNodes per block × 7 blocks = 14
block-boundary points**. Total: ~18 serial information gates.

---

## 2. Related Work & Novelty Gaps

### 2.1 Closest Related Work

| Paper | What they do | What they DON'T do |
|-------|-------------|-------------------|
| **SRIF** (ECCV 2022) — "Reducing Information Loss for SNNs" | Soft reset preserves residual membrane potential within same neuron | Does not forward membrane to other modules; no ghost analysis |
| **SpQuant-SNN** (Frontiers 2024) | Uses membrane potential prior for channel-skipping attention | Uses membrane for pruning decisions, not as feature input to ghost branches |
| **RMP-SNN** | Residual membrane potential for deeper networks | Highway connections for spikes, not membrane-to-ghost forwarding |
| **SpikeYOLO** (ECCV 2024 Best Paper Candidate) | I-LIF neuron with integer-valued training | No ghost convolution analysis; no recall-specific optimization |
| **MSD** (CVPR 2025) | Multi-scale spiking detection framework | No efficient architecture compression; no VisDrone/drone domain |
| **GhostNetV2** (NeurIPS 2022) | DFC attention for ghost modules in continuous networks | Continuous networks only; no SNN analysis |
| **BIF Neurons** | Bistable IF for information preservation | General neuron design; no ghost-specific or detection-specific |
| **IBRA-LIF** | Range-aligned activations for richer spike representation | Training trick; no architectural membrane forwarding |

### 2.2 What Has Never Been Done (Our Novelty)

1. **Nobody has analyzed ghost convolution information loss under
   binary spike constraints** — the ghost paradigm was always studied
   in continuous settings
2. **Nobody has proposed forwarding membrane potentials to ghost cheap
   branches** — membrane work focuses on soft reset (same neuron) or
   channel attention, not cross-module feature input
3. **Nobody has proposed a recall-targeted loss for SNN detection** —
   SNN detection papers optimize mAP; recall is the real bottleneck
   for dense small objects
4. **No SNN detection work targets VisDrone/drone surveillance** — an
   open application domain for neuromorphic efficiency

---

## 3. Proposed Method: Membrane-Guided Ghost (MGG)

### 3.1 Architecture: MGG Convolution

Replace `SGhostConv` at critical locations. Instead of:

```python
# Current SGhostConv (spike.py:473-478)
def forward(self, x):
    p = self.primary(x)        # SConv → binary spikes
    g = self.cheap(p)          # depthwise on BINARY → limited diversity
    return [cat(p[i], g[i]) for i in range(T)]
```

Proposed MGG:

```python
class MGGhostConv(nn.Module):
    """Membrane-Guided Ghost Convolution.

    Forwards pre-threshold membrane potentials to the cheap branch,
    restoring continuous feature diversity while maintaining binary
    spike output throughout.
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1,
                 ratio=2, dw_k=3):
        super().__init__()
        c_intrinsic = math.ceil(c2 / ratio)
        c_ghost = c2 - c_intrinsic

        # Primary: produces both spikes AND membrane potentials
        self.primary_conv = layer.Conv2d(c1, c_intrinsic, k, s,
                                         autopad(k, p), groups=g, bias=False)
        self.primary_bn = seBatchNorm(c_intrinsic, time_step)
        self.primary_spike = neuron.IFNode(
            surrogate_function=surrogate.ATan(),
            detach_reset=True, step_mode='s')

        # Cheap: receives MEMBRANE POTENTIALS (continuous), not spikes
        self.cheap = SConv(c_intrinsic, c_ghost, dw_k, 1, None,
                           c_intrinsic, 1, act=True)

    def forward(self, x):
        # Step 1: Compute primary convolution
        conv_out = [self.primary_conv(x[i]) for i in range(time_step)]

        # Step 2: Batch normalize → continuous membrane potentials
        membrane = self.primary_bn(conv_out)

        # Step 3: Primary output = binary spikes (standard path)
        spikes = [self.primary_spike(membrane[i]) for i in range(time_step)]

        # Step 4: Cheap branch receives MEMBRANE (continuous!)
        #         instead of spikes (binary) — THIS IS THE KEY CHANGE
        ghost = self.cheap(membrane)  # rich continuous input → diverse output

        # Step 5: Concatenate spike output + ghost output
        # Both outputs pass through IFNode → all binary at output
        return [torch.cat([spikes[i], ghost[i]], 1) for i in range(time_step)]
```

**Key properties:**
- **Output is still fully binary** (both branches pass through IFNode)
- **Zero extra parameters** (same conv kernels, same architecture)
- **Minimal extra compute** (membrane is already computed; just routed
  differently)
- **Backwards compatible** (drop-in replacement for SGhostConv)

### 3.2 Where to Deploy MGG

Replace SGhostConv with MGGhostConv at the 4 binary-bottleneck
locations identified in Section 1.4:

```yaml
# su-yolo-720p-mid-mgg.yaml (proposed)
backbone:
  [
   [-1, 1, MGGhostEncoderLite, [32, 3, 2]],  # 0-P1/2 — MGG in stem
   [-1, 1, GhostBasicBlock1, [64, 64, 32, 1]],    # 1-P2/4 (internal act=False, OK)
   [-1, 1, GhostBasicBlock1, [128, 128, 64, 1]],   # 2-P3/8
   [-1, 1, GhostBasicBlock1, [128, 128, 64, 1]],   # 3-P4/16
  ]
head:
  [
   [-1, 1, MGGhostConv, [64, 1, 1]],   # 4 — MGG at FPN junction
   # ... (rest unchanged) ...
   [-1, 1, MGGhostConv, [64, 3, 2]],   # 11 — MGG at downsample
   # ...
   [-1, 1, MGGhostConv, [64, 3, 2]],   # 14 — MGG at downsample
   # ...
  ]
```

### 3.3 Parameter & Compute Budget

| Component | SGhostConv | MGGhostConv | Delta |
|-----------|-----------|-------------|-------|
| Conv parameters | Same | Same | 0 |
| BN parameters | Same | Same | 0 |
| IFNode | Same | Same | 0 |
| **Total params** | — | — | **+0** |
| Cheap branch FLOPs | Same ops, binary input | Same ops, continuous input | ~0 |
| Memory | Spikes only | +membrane tensor at 4 points | +4 × (B,C,H,W) floats |

**Zero-parameter, near-zero-compute modification.**

The only overhead is temporarily storing membrane potentials at 4 network
points instead of discarding them immediately. At the mid-ghost model's
channel widths (32-64 channels), this is negligible.

---

## 4. Positive Coverage Loss (PCL)

### 4.1 Motivation

Standard TAL loss optimizes for classification accuracy (precision-oriented).
Recall drops when some GTs have NO assigned positive anchors — the network
never learns to detect those objects.

### 4.2 Formulation

After TAL assigner returns `mask_pos` (shape: `[B, n_max_boxes, H*W]`):

```python
# In ComputeLoss.__call__() (utils/loss_tal.py)

# Count how many positive anchors each GT has
gt_pos_count = mask_pos.sum(dim=-1)                    # (B, n_max_boxes)

# Binary: does each GT have at least one positive?
gt_has_positive = (gt_pos_count > 0).float()           # (B, n_max_boxes)

# Mask out padding GTs
valid_gt_mask = mask_gt.squeeze(-1)                    # (B, n_max_boxes)

# Coverage = fraction of valid GTs with at least one positive anchor
coverage = (gt_has_positive * valid_gt_mask).sum() / valid_gt_mask.sum().clamp(1)

# PCL: penalize missing coverage
loss_pcl = 1.0 - coverage

# Optional soft version: encourage MULTIPLE positives per GT
soft_coverage = (gt_pos_count.clamp(max=topk) / topk * valid_gt_mask).sum() \
                / valid_gt_mask.sum().clamp(1)
loss_pcl_soft = 1.0 - soft_coverage
```

### 4.3 Integration

```python
# loss_tal.py, after line 215
loss[0] *= 7.5    # box
loss[1] *= 0.5    # cls (consider increasing to 1.0)
loss[2] *= 1.5    # dfl
loss_pcl *= lambda_pcl  # e.g., 2.0

return (loss.sum() + loss_pcl) * batch_size, loss.detach()
```

---

## 5. Theoretical Analysis (Paper Section)

### 5.1 Information Capacity of Ghost Cheap Branch

**Proposition 1.** For a k×k depthwise convolution with binary input
x ∈ {0,1}^(k²), the output y = Σᵢ wᵢxᵢ has at most k²+1 distinct
values. The effective information capacity is:

  I_binary = log₂(k²+1) bits per spatial location

For continuous input x ∈ ℝ^(k²), the effective capacity is:

  I_continuous ≈ k² × B_eff bits (where B_eff ≈ 8-10 effective bits)

**Ratio:** I_continuous / I_binary ≈ k² × 8 / log₂(k²+1)

For k=3: 72 / 3.32 ≈ **21.7× information reduction**.

### 5.2 Recall Survival Probability

**Proposition 2.** For an SNN with N serial IFNode gates, each with
spike probability pᵢ (dependent on input signal strength), the recall
for objects of signal strength s is:

  R(s) = ∏ᵢ₌₁ᴺ pᵢ(s)

For marginal objects where pᵢ ≈ p ∈ (0.5, 0.9):

  R(s) = pᴺ

This decays exponentially with network depth N. Ghost compression
reduces redundancy, lowering p at each gate. MGG restores p by
providing the cheap branch continuous information independent of the
spike gate outcome.

### 5.3 MGG Information Recovery

**Proposition 3.** MGG convolution's cheap branch receives membrane
potentials m ∈ ℝ^(c_intrinsic × H × W) instead of spikes
s ∈ {0,1}^(c_intrinsic × H × W). This restores the cheap branch
information capacity from I_binary to I_continuous, a gain of
~21.7× per spatial location.

Crucially, the concatenated output [spikes; ghost(membrane)] is still
binary (ghost branch includes its own IFNode), preserving SNN
compatibility. The information gain manifests as **increased feature
diversity** in the ghost channels, not as continuous output values.

---

## 6. Experiment Design

### 6.1 Ablation Matrix

| # | Config | What it tests |
|---|--------|---------------|
| A | Baseline Mid-Ghost | Current state (recall ≈ 0.4) |
| B | A + focal loss (γ=1.5) | Low-hanging fruit baseline |
| C | A + topk=15 | Assigner improvement |
| D | A + cls gain 1.0 | Classification signal strength |
| E | B+C+D | Combined baseline improvements |
| F | E + MGG at 4 locations | Membrane-Guided Ghost only |
| G | E + PCL (λ=2.0) | Positive Coverage Loss only |
| H | E + MGG + PCL | **Full proposed method** |
| I | H + MGG at ALL ghost convs | Saturation test |

### 6.2 Metrics

- **Primary**: Recall@0.5, AR (Average Recall), AR_small, AR_medium
- **Secondary**: mAP@0.5, mAP@0.5:0.95
- **Efficiency**: GOPs, parameter count, energy (mJ on ZCU102)
- **Analysis**: Spike entropy per layer, firing rate distributions,
  membrane potential variance at MGG points

### 6.3 Datasets

| Dataset | Purpose | Characteristics |
|---------|---------|-----------------|
| **VisDrone-DET** | Primary evaluation | Dense small objects, 10 classes, drone perspective |
| **COCO val2017** | Generalization check | Standard benchmark, compare with SpikeYOLO/MSD |
| **VisDrone-MOT** | Temporal analysis | Multi-frame, test time_step>1 impact |

### 6.4 Comparison with SOTA

| Method | Venue | Type | mAP | Energy | Params |
|--------|-------|------|-----|--------|--------|
| SpikeYOLO | ECCV 2024 | SNN | 67.2 | 5.7× less | 23.1M |
| MSD | CVPR 2025 | SNN | 62.0 | 6.43 mJ | 7.8M |
| **Ours** | — | SNN+MGG | TBD | Target < MSD | ~0.5M |

Key advantage: our model is **~15× smaller** than MSD and **~46×
smaller** than SpikeYOLO. The comparison shows MGG enables extreme
compression without recall collapse.

---

## 7. Conference Fit Analysis

### ECCV 2025 (Deadline ~March 2025)
- **Fit**: Computer vision, detection, efficient architectures
- **Strengths**: Novel architecture + theory + application
- **Risk**: Tight timeline for experiments

### CVPR 2026 (Deadline ~November 2025)
- **Fit**: Perfect — detection, neuromorphic vision, efficiency
- **Strengths**: Can include ZCU102 hardware results
- **Best target if timeline allows**

### NeurIPS 2025 (Deadline ~May 2025)
- **Fit**: Learning theory (information bottleneck analysis) + architecture
- **Strengths**: Theoretical propositions + practical impact
- **Risk**: Less focused on "vision" — need strong theory section

### ICCV 2025 (Deadline ~March 2025)
- **Fit**: Good — vision, detection, neuromorphic
- **Similar to ECCV fit**

---

## 8. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Membrane forwarding creates training instability | Medium | Freeze MGG paths first 10 epochs; gradient clipping on membrane |
| Recall improvement is marginal (<2%) | Low | PCL + baseline fixes already significant; MGG is additive |
| Reviewers: "not truly zero cost" (memory overhead) | Medium | Quantify memory: ~0.3MB extra at 4 points; compare to model size |
| Reviewers: "SRIF already does membrane reuse" | Medium | SRIF does soft reset (same neuron); MGG forwards to different module |
| Hardware deployment issues | Low | SDConv already bridges spike→continuous; MGG is internal |

---

## 9. Paper Outline

1. **Introduction** (1 page)
   - SNN detection growing but recall lags
   - Ghost convolutions widely used but unanalyzed for SNNs
   - Preview: information collapse + MGG + PCL

2. **Related Work** (1 page)
   - SNN object detection (SpikeYOLO, MSD, Spiking-YOLO)
   - Efficient architectures (GhostNet, GhostNetV2)
   - Information preservation in SNNs (SRIF, RMP-SNN, SpQuant)

3. **Analysis: Spike-Ghost Information Collapse** (1.5 pages)
   - Theoretical propositions with proofs
   - Empirical measurement on trained SU-YOLO model
   - Visualization: cheap branch activation histograms

4. **Method** (2 pages)
   - 4.1 Membrane-Guided Ghost Convolution
   - 4.2 Positive Coverage Loss
   - 4.3 Integration with existing SNN detectors

5. **Experiments** (2.5 pages)
   - VisDrone ablation matrix
   - COCO comparison with SOTA
   - Efficiency analysis (GOPs, energy, latency)
   - Hardware deployment on ZCU102

6. **Analysis & Discussion** (1 page)
   - Spike entropy analysis
   - Per-class recall breakdown
   - Failure cases
   - Limitations

7. **Conclusion** (0.5 pages)

---

## 10. Implementation Roadmap in This Repo

### Phase 1: Baseline fixes (Week 1)
- Enable focal loss (γ=1.5) in `hyp.visdrone.yaml`
- Increase TAL topk via `YOLOM=15`
- Increase cls gain to 1.0 in `loss_tal.py`
- Train and measure recall improvement

### Phase 2: MGG module (Week 2-3)
- Implement `MGGhostConv` in `models/spike.py`
- Register in `models/yolo.py` parse_model()
- Create `su-yolo-720p-mid-mgg.yaml`
- Unit test: verify output shapes match SGhostConv

### Phase 3: PCL loss (Week 2)
- Add PCL computation to `utils/loss_tal.py`
- Add `lambda_pcl` to hyperparameter config
- Verify gradient flow

### Phase 4: Training & Ablation (Week 3-6)
- Run full ablation matrix (configs A through I)
- COCO evaluation for SOTA comparison
- Spike entropy and firing rate analysis

### Phase 5: Hardware & Writing (Week 6-8)
- ZCU102 deployment and energy measurement
- Write paper
- Prepare visualizations

---

## 11. Why This Can Win at a Top Conference

1. **Clean fundamental insight**: Ghost convolutions assume continuous
   features; SNNs have binary spikes. Nobody noticed this mismatch.

2. **Elegant solution**: Zero parameters, zero FLOPs, just route an
   already-computed value to a different module.

3. **Quantifiable theory**: Information capacity propositions with
   concrete numbers (21.7× reduction).

4. **Practical impact**: Breaks recall ceiling on a real drone
   detection task with an ultra-lightweight (~0.5M param) model.

5. **Broad applicability**: Any SNN using ghost/depthwise compression
   suffers this same problem. MGG is a general fix.

6. **Timely**: SNN detection is hot (SpikeYOLO ECCV 2024 Best Paper
   Candidate, MSD CVPR 2025). This adds a new dimension to the field.

7. **Reproducible**: Clean integration into existing YOLO codebase,
   single-file module change.
