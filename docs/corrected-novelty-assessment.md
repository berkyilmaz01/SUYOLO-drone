# SU-YOLO Drone: Corrected Novelty Assessment

> **Status**: Pre-training. All architecture estimates are design-time only.
> No mAP, FPS, or parameter counts have been measured from trained models.
>
> **Date**: 2026-02-18
> **Purpose**: Reviewer-proof novelty framing based on fact-checking against published literature.

---

## 1. Model Performance Summary

### Only Confirmed Result (Original Paper)

| Model | Dataset | mAP@0.5 | Params | Energy |
|-------|---------|---------|--------|--------|
| SU-YOLO (original) | URPC2019 | 78.8% | 6.97M | 2.98 mJ |

### Drone Variant Architectures (Design-Time Estimates, Not Measured)

All estimates at 1280x736, time_step=1, targeting Xilinx ZCU102 FPGA (3x B4096 DPU @ 281MHz).

| Model | Channels | Scales | Est. GOPS | Est. ZCU102 FPS | Est. Latency |
|-------|----------|--------|-----------|-----------------|--------------|
| su-yolo-720p-nano | 16→32→64 | P2+P3 | ~4.5 | ~130 (3-core) | ~7.8ms |
| su-yolo-720p-nano-v2 | 24→48→96 | P2+P3 | ~7.0 | ~90 (3-core) | ~11ms |
| su-yolo-720p-mid | 32→64→128 | P2+P3+P4 | ~15 | 45-65 | — |
| su-yolo-720p | 64→128→256→512 | P2+P3+P4 | ~40+ | — | — |

| Ghost Variant | Base GOPS | Ghost GOPS | Reduction |
|---------------|-----------|------------|-----------|
| su-yolo-720p-nano-ghost | ~4.5 | ~3.4 | ~24% |
| su-yolo-720p-nano-v2-ghost | ~7.0 | ~4.5 | ~36% |
| su-yolo-720p-mid-ghost | ~15 | ~9 | ~40% |

**Parameter counts must be measured** via `python tools/compute_gops.py --cfg <model.yaml>`.

---

## 2. Competitor Landscape (Updated)

### Sub-1M Detectors on VisDrone2019

| Model | Params | mAP50 | mAP50:95 | Input | Source |
|-------|--------|-------|----------|-------|--------|
| **ELNet** | **0.3M** | 28.4 | 15.5 | 640x640 | Remote Sensing 2025, MDPI |
| NanoDet-m | 0.95M | — | — | 320/416 | Official repo (COCO only) |
| PP-PicoDet-XS | 0.7M | — | — | — | PaddleDetection (COCO only) |
| LEAF-YOLO-N | 1.2M | 39.7 | 21.9 | 640x640 | VisDrone2019-val |

**Critical**: ELNet at 0.3M params on VisDrone2019 invalidates any "first sub-1M on drone benchmarks" claim.

### HazyDet Published Baselines (Top performers from Feng et al.)

| Model | Params | mAP (Synth Test) | mAP (RDDTS) | FPS |
|-------|--------|-------------------|--------------|-----|
| DeCoDet (SOTA) | 34.62M | 0.520 | 0.387 | 59.7 |
| Cascade RCNN | 69.15M | 0.516 | 0.372 | 42.9 |
| Deformable DETR | 40.01M | 0.515 | 0.369 | 50.9 |
| TOOD | 32.02M | 0.514 | 0.367 | 48.0 |
| DDOD | 32.20M | 0.507 | 0.371 | 47.6 |
| YOLOX | 8.94M | 0.423 | 0.354 | 71.2 |

**Note**: "RDDTS" = Real-hazy Drone Detection Testing Set. This IS the official HazyDet naming (confirmed from paper).

---

## 3. Corrected Novelty Claims

### Claim 1: "First Sub-1M Parameter Detector on Drone Benchmarks"

**Status: NOT DEFENSIBLE**

ELNet (2025) reports 0.3M params at 28.4 mAP50 on VisDrone2019. This directly refutes the claim.

**Corrected framing**: "To the best of our knowledge, we present the first *spiking* detector evaluated on VisDrone2019 and HazyDet at sub-1M parameters, offering a neuromorphic-hardware-friendly alternative to ANN ultralight detectors like ELNet (0.3M)."

### Claim 2: "Ghost Spiking Convolutions Improve Accuracy While Reducing Parameters"

**Status: Internal math consistent, but universal negative claim risky**

The claim "none of 30+ papers report simultaneous improvement on both axes" is an absolute that reviewers will challenge.

**Corrected framing**: "In our experiments, SGhostConv improves accuracy while reducing params/compute — an effect we did not observe reported in the Ghost-YOLO papers we surveyed [cite specific subset]."

### Claim 3: "First SNN-Based Detector on VisDrone and HazyDet"

**Status: Plausible but needs hedging**

No prior SNN detector clearly reports results on VisDrone or HazyDet, but absence of evidence is not evidence of absence.

**Corrected framing**: "To the best of our knowledge, no prior SNN detector reports results on VisDrone or HazyDet; existing SNN detectors are typically evaluated on COCO/VOC/event-camera datasets (e.g., EMS-YOLO, SA-YOLO)."

### Claim 4: "Native 1920x1080 Training at Ultra-Low Parameter Count"

**Status: Not safely defensible as absolute**

**Corrected framing**: "We train end-to-end at 1920x1080 without tiling (no SAHI) at sub-1M params, which is rare among lightweight detectors and (to our knowledge) not previously demonstrated for spiking YOLO-style models."

### Claim 5: "Ghost Convolutions Inside a Spiking Framework"

**Status: Strong if scoped correctly**

**Corrected framing**: "We integrate Ghost-style feature generation into a directly-trained spiking YOLO backbone/neck and show consistent gains. Both primary and cheap branches operate on spike trains (SGhostConv)."

### Claim 6: "Remarkable HazyDet Performance"

**Status: Comparison targets verified, naming confirmed**

The benchmark uses "Test-set" for synthetic hazy test and "RDDTS" for real-world. Numbers in `benchmark_hazydet.py` match published Table 3.

**Corrected framing**: Use official naming — "mAP on Synth Test-set" and "mAP on RDDTS" — and only cite numbers after actually running evaluation.

---

## 4. Corrected Algorithmic Novelty Claims

### SeBatchNorm2d (Variance-Scaled Separated BN)

**NOT novel as described**. Time-to-channel folding for BN is published as SeBN in SU-YOLO (2025). The variance scaling knob may be a new variant.

**Corrected framing**: "We build on SeBN (time-to-channel folding) and introduce a variance scaling factor that modulates normalization strength per path."

### Split-Gate CSP-ELAN blocks (BasicBlock1/2)

**Partially novel**. SU-YOLO already integrates CSPNet into spiking residual blocks.

**Corrected framing**: "We propose a spike-gated partial-channel residual block inspired by CSP-style split/concat, where IF neurons gate only the processed branch."

### Infinite-threshold temporal averaging (SDConv)

**NOT novel**. Infinite threshold to avoid firing is documented in SpikingJelly tutorials. Temporal averaging as rate decoding is standard.

**Corrected framing**: "A clean spike→rate decoding interface at the detection head that simplifies deployment."

### Morphological spike denoising (FindPoints)

**Potentially interesting but currently commented out**. Prior work exists on spike-based filtering.

**Corrected framing**: "A lightweight, handcrafted morphological filtering module for intermediate spike maps (optional), evaluated via ablations."

### Dual reset strategy

**NOT novel**. SpikingJelly requires reset after each batch. Pre-epoch reset is redundant.

**Corrected framing**: Do not claim as novelty.

### Ghost convolutions on binary spike streams (SGhostConv)

**Plausibly novel as a combination**. Not found as a standard pattern in core SNN detector literature.

**Corrected framing**: "We adapt Ghost-style feature generation to the spiking domain (SGhostConv), where both primary and cheap branches operate on spike trains."

### Selective Ghost placement strategy

**Engineering insight, not algorithmic novelty**. Valuable only with ablation data.

**Corrected framing**: "We find Ghost is most effective in stride-1 blocks and avoid it in stride-2/aggregation blocks; we validate this via ablations."

### Inner-IoU loss

**Not novel**. Published loss (arXiv:2311.02877). Only the application context is new.

### SDFL (Spiking Distribution Focal Loss)

**Likely not a new loss formulation**. Just a spiking-compatible DFL implementation.

**Corrected framing**: "A spiking-compatible implementation of DFL to keep the graph within the spiking framework."

### P2-only two-scale detection

**Not inherently novel**. Restricting to small-object scales is a known design choice.

**Corrected framing**: "For drone imagery dominated by small objects, we use P2/P3-only detection to reduce compute; ablations show improved efficiency with minimal accuracy loss."

---

## 5. Defensible Novelty Summary (What You CAN Claim)

1. **First spiking detector on VisDrone/HazyDet** (with "to the best of our knowledge" hedging)
2. **SGhostConv: Ghost modules operating on spike trains** (plausibly novel combination)
3. **Variance-scaled SeBN** (extension of existing SeBN, not entirely new)
4. **Spike-gated partial-channel residual blocks** (novel implementation variant of CSP in SNNs)
5. **Sub-1M spiking detector for drone imagery** (novel in SNN domain; ANNs like ELNet already achieve 0.3M)
6. **High-resolution spiking detection without tiling** (novel for spiking YOLO models specifically)
7. **Selective Ghost placement validated by ablations** (engineering contribution if ablated)

---

## 6. What Must Be Done Before Any Claims Are Defensible

- [ ] Run `compute_gops.py` on all 10 model variants to get exact parameter counts
- [ ] Train at least nano-ghost and nano-v2-ghost on VisDrone2019
- [ ] Train at least one variant on HazyDet (both synth test and RDDTS evaluation)
- [ ] Run ablation: ghost vs non-ghost (same model otherwise)
- [ ] Run ablation: variance scaling factor in SeBN
- [ ] Run ablation: P2+P3 vs P2+P3+P4 detection scales
- [ ] Measure actual FPS on ZCU102 or GPU
- [ ] Compare against ELNet (0.3M), LEAF-YOLO-N (1.2M) on VisDrone2019
