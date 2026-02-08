# SU-YOLO Ghost Code Review

## Verdict: Core pipeline works. Several dead-code and bug issues found.

### Critical Bugs

1. **`SDConv.bn` and `SDConv.act` never used in forward()** — `spike.py:131-140`
   - BatchNorm and IFNode allocated but forward() skips them. Wastes memory.

2. **`BasicBlock1.act1` never used** — `spike.py:240`
   - IFNode allocated, never called in forward(). Dead parameter.

3. **`BasicBlock2.act1` never used** — `spike.py:273`
   - Same issue as above.

4. **`FindPoints()` crashes on bfloat16** — `spike.py:365-440`
   - Only handles float32/float16. Any other dtype → `UnboundLocalError`.
   - Currently safe because Denoise is commented out.

5. **`detect.py` default source is empty** — `detect.py:247`
   - `--source` defaults to `ROOT / ''` which will fail.

### Dead Code

6. **`Denoise()`, `Denoise2()`, `Denoise3()`, `FindPoints()`** — `spike.py:344-440`
   - ~100 lines defined but never called anywhere.

7. **`TripleDDetect` computes d1/d2 during inference but discards them** — `yolo.py:401-404`
   - Only d3 returned at inference. ~66% wasted compute in detection head.

8. **`fuse()` does nothing** — `yolo.py:518-531`
   - Conv+BN fusion commented out. Inference misses 10-20% speedup.

### Design Issues

9. **EMA uses direct reference + separate model2** — `train.py:126,178`
   - Non-standard pattern. Works but doubles memory during init.

10. **Global mutable `time_step`** — `spike.py:24-31`
    - Not thread-safe. Breaks with multiple models in same process.

11. **Hardcoded `gs=32`** — `train.py:150`
    - Dynamic stride calculation commented out.

### What Works

- SNN temporal unfolding with IFNode neurons
- SeBatchNorm temporal dimension handling
- TAL loss (CIoU + DFL) computation
- All 5 YAML model configs (channel math verified)
- ByteTracker integration
- reset_net() per batch/frame
- Warmup, AMP, gradient clipping
