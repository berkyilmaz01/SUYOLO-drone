import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import xywh2xyxy
from utils.metrics import bbox_iou
from utils.tal.anchor_generator import dist2bbox, make_anchors, bbox2dist
from utils.tal.assigner import TaskAlignedAssigner
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class VarifocalLoss(nn.Module):
    # Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367
    def __init__(self):
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(),
                                                       reduction="none") * weight).sum()
        return loss


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl=False, inner_iou_ratio=None):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl
        self.inner_iou_ratio = inner_iou_ratio

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # iou loss
        bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)
        pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).view(-1, 4)
        target_bboxes_pos = torch.masked_select(target_bboxes, bbox_mask).view(-1, 4)
        bbox_weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)

        iou = bbox_iou(pred_bboxes_pos, target_bboxes_pos, xywh=False, CIoU=True, inner_iou_ratio=self.inner_iou_ratio)
        loss_iou = 1.0 - iou

        loss_iou *= bbox_weight
        loss_iou = loss_iou.sum() / target_scores_sum

        # dfl loss
        if self.use_dfl:
            dist_mask = fg_mask.unsqueeze(-1).repeat([1, 1, (self.reg_max + 1) * 4])
            pred_dist_pos = torch.masked_select(pred_dist, dist_mask).view(-1, 4, self.reg_max + 1)
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            target_ltrb_pos = torch.masked_select(target_ltrb, bbox_mask).view(-1, 4)
            loss_dfl = self._df_loss(pred_dist_pos, target_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl, iou

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction="none").view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_right.view(-1),
                                     reduction="none").view(target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)


class ComputeLoss:
    # Compute losses
    def __init__(self, model, use_dfl=True):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device), reduction='none')

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls = FocalLoss(BCEcls, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.BCEcls = BCEcls
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.assigner = TaskAlignedAssigner(topk=int(os.getenv('YOLOM', 10)),
                                            num_classes=self.nc,
                                            alpha=float(os.getenv('YOLOA', 0.5)),
                                            beta=float(os.getenv('YOLOB', 6.0)))
        inner_iou_ratio = h.get("inner_iou_ratio", None)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=use_dfl, inner_iou_ratio=inner_iou_ratio).to(device)
        self.proj = torch.arange(m.reg_max).float().to(device)  # / 120.0
        self.use_dfl = use_dfl

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, p, targets, img=None, epoch=0):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = p[1] if isinstance(p, tuple) else p
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size, grid_size = pred_scores.shape[:2]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.BCEcls(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[2], iou = self.bbox_loss(pred_distri,
                                                   pred_bboxes,
                                                   anchor_points,
                                                   target_bboxes,
                                                   target_scores,
                                                   target_scores_sum,
                                                   fg_mask)

        loss[0] *= 7.5  # box gain
        loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class DistillationLoss(nn.Module):
    """Knowledge distillation loss for offline teacher-student training.

    Combines:
      - Logit KD: KL divergence on classification logits with temperature scaling
        (Hinton et al. 2015, "Distilling the Knowledge in a Neural Network")
      - Feature KD: MSE between adapter-projected student features and teacher features
        at spatially aligned scales (P3/8 and P4/16)

    The adapter layers (1x1 convs) project student channels to match teacher channels
    so MSE can be computed. These adapters are learnable and included in the optimizer.
    """

    def __init__(self, student_channels, teacher_channels, temperature=4.0, device='cpu'):
        """
        Args:
            student_channels: list of student neck output channels at aligned scales
                              e.g. [64, 64] for mid-ghost P3 and P4
            teacher_channels: list of teacher neck output channels at aligned scales
                              e.g. [256, 512] for GELAN-C P3 and P4
            temperature: softening temperature for logit KD (higher = softer)
            device: torch device
        """
        super().__init__()
        self.temperature = temperature
        self.num_feat_scales = len(student_channels)

        # 1x1 conv adapters to project student features to teacher feature space
        self.adapters = nn.ModuleList()
        for s_ch, t_ch in zip(student_channels, teacher_channels):
            self.adapters.append(nn.Conv2d(s_ch, t_ch, 1, bias=False))

        self.to(device)

    def logit_kd_loss(self, student_logits, teacher_logits):
        """KL divergence loss on classification logits with temperature scaling.

        Both inputs are raw logits (before sigmoid). We compute the difference
        between the cross-entropy and the teacher's own entropy, so the loss
        is 0 when the student perfectly matches the teacher (proper KL).

        Args:
            student_logits: (B, C, N) student cls logits
            teacher_logits: (B, C, N) teacher cls logits
        """
        T = self.temperature

        with torch.cuda.amp.autocast(enabled=False):
            student_soft = student_logits.float() / T
            teacher_soft = teacher_logits.float().detach() / T
            teacher_prob = torch.sigmoid(teacher_soft)

            # BCE(student, teacher_prob): H(teacher) + KL(teacher || student)
            bce = F.binary_cross_entropy_with_logits(
                student_soft, teacher_prob, reduction='mean')

            # Subtract teacher's own entropy so loss = 0 at perfect match
            eps = 1e-7
            teacher_entropy = F.binary_cross_entropy(
                teacher_prob.clamp(eps, 1 - eps),
                teacher_prob.clamp(eps, 1 - eps),
                reduction='mean')

            loss = (bce - teacher_entropy) * (T * T)

        return loss

    def feature_kd_loss(self, student_feats, teacher_feats):
        """MSE loss between adapter-projected student features and teacher features.

        Args:
            student_feats: list of student feature tensors at aligned scales
            teacher_feats: list of teacher feature tensors at aligned scales
        """
        loss = torch.tensor(0.0, device=student_feats[0].device)
        n_scales = min(len(student_feats), len(teacher_feats), self.num_feat_scales)

        for i in range(n_scales):
            s_feat = student_feats[i]
            t_feat = teacher_feats[i].to(s_feat.device).detach()

            with torch.cuda.amp.autocast(enabled=False):
                adapted = self.adapters[i](s_feat.float())

                if adapted.shape[2:] != t_feat.shape[2:]:
                    adapted = F.interpolate(adapted, size=t_feat.shape[2:],
                                            mode='bilinear', align_corners=False)

                loss = loss + F.mse_loss(adapted, t_feat.float())

        return loss / max(n_scales, 1)

    def forward(self, student_cls_logits, teacher_cls_logits,
                student_feats, teacher_feats, alpha=1.0, beta=0.5):
        """
        Args:
            student_cls_logits: student classification logits (B, nc, total_anchors)
            teacher_cls_logits: teacher classification logits, or None to skip
            student_feats: list of student neck feature maps at aligned scales
            teacher_feats: list of teacher neck feature maps at aligned scales, or None
            alpha: weight for logit KD loss
            beta: weight for feature KD loss

        Returns:
            kd_loss: scalar distillation loss (carries gradients)
            kd_items: detached tensor [logit_kd, feat_kd] for logging
        """
        # Use separate variables to preserve the computation graph;
        # in-place assignment to a torch.zeros tensor severs gradients.
        logit_loss = torch.tensor(0.0, device=student_cls_logits.device)
        feat_loss = torch.tensor(0.0, device=student_cls_logits.device)

        if teacher_cls_logits is not None and alpha > 0:
            logit_loss = self.logit_kd_loss(student_cls_logits, teacher_cls_logits)

        if teacher_feats is not None and beta > 0 and student_feats is not None:
            feat_loss = self.feature_kd_loss(student_feats, teacher_feats)

        kd_loss = alpha * logit_loss + beta * feat_loss
        kd_items = torch.stack([logit_loss.detach(), feat_loss.detach()])
        return kd_loss, kd_items
