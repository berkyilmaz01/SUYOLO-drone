"""Dry-run test for knowledge distillation components.

Verifies shapes, loss computation, and gradient flow without needing
the full dataset or GPU. Run with: python test_kd_dry_run.py
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn


def test_distillation_loss():
    print('=== Test 1: DistillationLoss ===')
    from utils.loss_tal import DistillationLoss

    student_channels = [64, 64]
    teacher_channels = [256, 512]
    dl = DistillationLoss(student_channels, teacher_channels, temperature=4.0, device='cpu')
    print(f'  Adapters: {len(dl.adapters)}')
    for i, a in enumerate(dl.adapters):
        print(f'  Adapter {i}: {a.weight.shape}')
    assert dl.adapters[0].weight.shape == (256, 64, 1, 1)
    assert dl.adapters[1].weight.shape == (512, 64, 1, 1)

    # Logit KD
    s_cls = torch.randn(2, 3, 100)
    t_cls = torch.randn(2, 3, 100)
    logit_loss = dl.logit_kd_loss(s_cls, t_cls)
    print(f'  Logit KD loss: {logit_loss.item():.4f}')
    assert logit_loss.item() > 0

    # Feature KD
    s_feats = [torch.randn(2, 64, 135, 240), torch.randn(2, 64, 68, 120)]
    t_feats = [torch.randn(2, 256, 135, 240), torch.randn(2, 512, 68, 120)]
    feat_loss = dl.feature_kd_loss(s_feats, t_feats)
    print(f'  Feature KD loss: {feat_loss.item():.4f}')
    assert feat_loss.item() > 0

    # Combined
    kd_loss, kd_items = dl(s_cls, t_cls, s_feats, t_feats, alpha=1.0, beta=0.5)
    print(f'  Combined KD loss: {kd_loss.item():.4f}')
    print(f'  KD items: logit={kd_items[0].item():.4f}, feat={kd_items[1].item():.4f}')
    assert kd_loss.item() > 0
    print('  PASSED')


def test_gradient_flow():
    print('\n=== Test 2: Gradient flow ===')
    from utils.loss_tal import DistillationLoss

    dl = DistillationLoss([64, 64], [256, 512], temperature=4.0, device='cpu')
    s_cls = torch.randn(2, 3, 100, requires_grad=True)
    t_cls = torch.randn(2, 3, 100)
    s_feats = [torch.randn(2, 64, 8, 8, requires_grad=True),
               torch.randn(2, 64, 4, 4, requires_grad=True)]
    t_feats = [torch.randn(2, 256, 8, 8), torch.randn(2, 512, 4, 4)]

    kd_loss, _ = dl(s_cls, t_cls, s_feats, t_feats, alpha=1.0, beta=0.5)
    kd_loss.backward()

    for i, a in enumerate(dl.adapters):
        assert a.weight.grad is not None, f'Adapter {i} has no gradient'
        print(f'  Adapter {i} grad norm: {a.weight.grad.norm().item():.4f}')
    assert s_cls.grad is not None, 'Student cls logits have no gradient'
    print(f'  Student cls grad norm: {s_cls.grad.norm().item():.4f}')
    for i, sf in enumerate(s_feats):
        assert sf.grad is not None, f'Student feat {i} has no gradient'
        print(f'  Student feat {i} grad norm: {sf.grad.norm().item():.4f}')
    print('  PASSED')


def test_teacher_output_format():
    print('\n=== Test 3: Teacher output save/load ===')
    teacher_output = {
        'feat_0': torch.randn(256, 135, 240).half(),
        'feat_1': torch.randn(512, 68, 120).half(),
        'feat_2': torch.randn(512, 34, 60).half(),
        'feature_layers': [15, 18, 21],
        'det_logits_0': torch.randn(67, 135, 240).half(),
        'det_logits_1': torch.randn(67, 68, 120).half(),
        'det_logits_2': torch.randn(67, 34, 60).half(),
        'num_det_scales': 3,
    }
    tmp = os.path.join(tempfile.gettempdir(), 'test_teacher.pt')
    torch.save(teacher_output, tmp)
    loaded = torch.load(tmp, map_location='cpu', weights_only=False)
    assert len(loaded) == len(teacher_output)
    for k, v in loaded.items():
        if isinstance(v, torch.Tensor):
            print(f'  {k}: {v.shape} ({v.dtype})')
            assert v.dtype == torch.float16
        else:
            print(f'  {k}: {v}')
    os.remove(tmp)

    # Verify teacher cls logit extraction (reg_max=16, nc=3 -> 64+3=67)
    logit = loaded['det_logits_0']
    t_reg_max = 16
    t_cls = logit[t_reg_max * 4:, :, :]
    assert t_cls.shape[0] == 3, f'Expected nc=3, got {t_cls.shape[0]}'
    print(f'  Teacher cls logit extraction: {t_cls.shape}')
    print('  PASSED')


def test_per_scale_logit_kd():
    """Verify that per-scale logit KD correctly matches spatial dimensions."""
    print('\n=== Test 4: Per-scale logit KD (stride matching) ===')
    from utils.loss_tal import DistillationLoss

    dl = DistillationLoss([64, 64], [256, 512], temperature=4.0, device='cpu')

    # Simulate 1920x1080 input
    # Student: P2/4 (480x270), P3/8 (240x135), P4/16 (120x68)
    # Teacher: P3/8 (240x135), P4/16 (120x68), P5/32 (60x34)
    student_reg_max = 8
    teacher_reg_max = 16
    nc = 3
    s_no = student_reg_max * 4 + nc  # 35
    t_no = teacher_reg_max * 4 + nc  # 67

    student_feats = [
        torch.randn(2, s_no, 270, 480),  # P2/4
        torch.randn(2, s_no, 135, 240),  # P3/8
        torch.randn(2, s_no, 68, 120),   # P4/16
    ]
    teacher_logits = [
        torch.randn(t_no, 135, 240),  # P3/8
        torch.randn(t_no, 68, 120),   # P4/16
        torch.randn(t_no, 34, 60),    # P5/32
    ]

    # Matched scales: student[1]=P3/8 ↔ teacher[0], student[2]=P4/16 ↔ teacher[1]
    matched_s = [1, 2]
    matched_t = [0, 1]

    total_loss = torch.tensor(0.0)
    for ms, mt in zip(matched_s, matched_t):
        s_flat = student_feats[ms].view(2, s_no, -1)
        s_cls = s_flat[:, student_reg_max * 4:, :]  # (B, nc, H*W)

        t_logit = teacher_logits[mt]
        t_cls = t_logit[teacher_reg_max * 4:, :, :].view(nc, -1).unsqueeze(0).expand(2, -1, -1)

        assert s_cls.shape == t_cls.shape, f"Shape mismatch: {s_cls.shape} vs {t_cls.shape}"
        loss = dl.logit_kd_loss(s_cls, t_cls)
        print(f'  Scale match s[{ms}]↔t[{mt}]: s_cls {s_cls.shape}, loss={loss.item():.4f}')
        total_loss = total_loss + loss

    print(f'  Total per-scale logit loss: {total_loss.item():.4f}')
    assert total_loss.item() > 0
    print('  PASSED')


def test_feature_kd_gradient():
    """Verify feature KD loss carries gradient through adapters."""
    print('\n=== Test 5: Feature KD gradient through adapters ===')
    from utils.loss_tal import DistillationLoss

    dl = DistillationLoss([64, 64], [256, 512], temperature=4.0, device='cpu')
    s_feats = [torch.randn(2, 64, 8, 8, requires_grad=True),
               torch.randn(2, 64, 4, 4, requires_grad=True)]
    t_feats = [torch.randn(2, 256, 8, 8), torch.randn(2, 512, 4, 4)]

    # Call feature_kd_loss directly (as the fixed code does)
    feat_loss = dl.feature_kd_loss(s_feats, t_feats)
    assert feat_loss.requires_grad, "Feature loss must carry gradient!"
    feat_loss.backward()

    for i, a in enumerate(dl.adapters):
        assert a.weight.grad is not None, f'Adapter {i} has no gradient!'
        print(f'  Adapter {i} grad norm: {a.weight.grad.norm().item():.6f}')
    for i, sf in enumerate(s_feats):
        assert sf.grad is not None, f'Student feat {i} has no gradient!'
        print(f'  Student feat {i} grad norm: {sf.grad.norm().item():.6f}')
    print('  PASSED')


if __name__ == '__main__':
    test_distillation_loss()
    test_gradient_flow()
    test_teacher_output_format()
    test_per_scale_logit_kd()
    test_feature_kd_gradient()
    print('\n=== All tests passed ===')
