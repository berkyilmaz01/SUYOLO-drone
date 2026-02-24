"""Knowledge distillation training script.

Based on train.py with additions for offline teacher-student distillation:
  - Loads pre-generated teacher features and logits from disk
  - Computes combined task + KD loss (logit KL-div + feature MSE)
  - Adapter layers (1x1 conv) project student features to teacher space

Usage:
    python train_kd.py \
        --cfg models/detect/su-yolo-720p-mid-ghost.yaml \
        --data data/hazydet.yaml --hyp data/hyps/hyp.visdrone.yaml \
        --img 1920 --batch 2 --time-step 1 --epochs 200 --cos-lr \
        --teacher-outputs teacher_outputs \
        --kd-alpha 1.0 --kd-beta 0.5 --kd-temperature 4.0 \
        --name hazydet-student-kd
"""

import argparse
import math
import os
import random
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path

if sys.platform == 'darwin':
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import numpy as np
try:
    import torch
except Exception as e:
    if 'macOS' in str(e) and ('2602' in str(e) or '1602' in str(e) or 'required' in str(e)):
        sys.exit(
            'MPS backend requires a newer macOS than you have.\n'
            'Run training on CPU instead: python train_kd.py --device cpu\n\nOriginal error: ' + str(e)
        )
    raise
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

import val as validate
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader, create_kd_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_img_size,
                           check_suffix, check_yaml, colorstr, get_latest_run, increment_path, init_seeds,
                           intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods,
                           one_cycle, one_flat_cycle, print_args, print_mutation, strip_optimizer, yaml_save)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss_tal import ComputeLoss, DistillationLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP,
                               smart_optimizer, smart_resume, torch_distributed_zero_first)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = None
import models.spike
from models.spike import set_time_step
from spikingjelly.activation_based.functional import reset_net


def train(hyp, opt, device, callbacks):
    save_dir, time_step, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.time_step, opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    set_time_step(time_step)
    callbacks.run('on_pretrain_routine_start')

    # KD parameters
    teacher_dir = opt.teacher_outputs
    kd_alpha = opt.kd_alpha
    kd_beta = opt.kd_beta
    kd_temperature = opt.kd_temperature

    # Directories
    w = save_dir / 'weights'
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)
    last, best = w / 'last.pt', w / 'best.pt'
    last_striped, best_striped = w / 'last_striped.pt', w / 'best_striped.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    hyp['anchor_t'] = 5.0
    opt.hyp = hyp.copy()

    # Save run settings
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Loggers
    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)
        loggers.keys = [
            'train/box_loss',
            'train/cls_loss',
            'train/dfl_loss',
            'train/kd_logit_loss',
            'train/kd_feat_loss',
            'metrics/precision',
            'metrics/recall',
            'metrics/mAP_0.5',
            'metrics/mAP_0.5:0.95',
            'val/box_loss',
            'val/cls_loss',
            'val/dfl_loss',
            'x/lr0',
            'x/lr1',
            'x/lr2']
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))
        data_dict = loggers.remote_dataset
        if resume:
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # Config
    plots = not evolve and not opt.noplots
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']
    is_coco = isinstance(val_path, str) and val_path.endswith('val2017.txt')

    # Model
    check_suffix(weights, '.pt')
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)
        ckpt = torch.load(weights, map_location='cpu', weights_only=False)
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors'), imgsz=opt.imgsz).to(device)
        model2 = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors'), imgsz=opt.imgsz).to(device)
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []
        csd = ckpt['model'].float().state_dict()
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
        model.load_state_dict(csd, strict=False)
        model2.load_state_dict(csd, strict=False)
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors'), imgsz=opt.imgsz).to(device)
        model2 = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors'), imgsz=opt.imgsz).to(device)
    amp = check_amp(model)

    # Freeze
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]
    for k, v in model.named_parameters():
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    # Image size
    gs = 32
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)

    # Batch size
    if RANK == -1 and batch_size == -1:
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})

    # Determine student neck channels for distillation adapter layers
    # Read from a sample teacher file to get teacher channel info
    teacher_sample = None
    if teacher_dir:
        teacher_path = Path(teacher_dir)
        sample_files = list(teacher_path.glob('*.pt'))
        if sample_files:
            teacher_sample = torch.load(sample_files[0], map_location='cpu', weights_only=False)
            LOGGER.info(f"Loaded sample teacher output from {sample_files[0]}")

    # Build distillation loss with adapter layers
    distill_loss = None
    if teacher_sample is not None:
        num_teacher_scales = teacher_sample.get('num_det_scales', 0)
        teacher_feature_layers = teacher_sample.get('feature_layers', [])
        teacher_reg_max = teacher_sample.get('teacher_reg_max', 16)
        teacher_nc = teacher_sample.get('teacher_nc', None)

        # Validate that teacher and student use the same number of classes
        if teacher_nc is not None and teacher_nc != nc:
            raise ValueError(f"Teacher nc={teacher_nc} != student nc={nc}. "
                             "Teacher and student must be trained on the same dataset.")
        LOGGER.info(f"Teacher: reg_max={teacher_reg_max}, nc={teacher_nc or 'unknown'}, "
                     f"det_scales={num_teacher_scales}")

        # Get teacher feature channels from saved tensors
        teacher_channels = []
        for k in range(len(teacher_feature_layers)):
            key = f'feat_{k}'
            if key in teacher_sample:
                teacher_channels.append(teacher_sample[key].shape[0])

        # Student detection head tells us which layers it reads from
        student_detect = de_parallel(model).model[-1]
        student_from = student_detect.f if isinstance(student_detect.f, list) else [student_detect.f]

        # Match scales: find overlapping spatial strides between teacher and student
        # Teacher GELAN-C: P3/8, P4/16, P5/32 (layers 15, 18, 21)
        # Student mid-ghost: P2/4, P3/8, P4/16
        # Matching: teacher[0]=P3/8 <-> student[1]=P3/8, teacher[1]=P4/16 <-> student[2]=P4/16
        student_strides = student_detect.stride.tolist()
        teacher_strides_map = {8: 0, 16: 1, 32: 2}  # GELAN-C stride -> teacher feat index

        matched_student_indices = []
        matched_teacher_indices = []
        matched_student_channels = []
        matched_teacher_channels = []

        for s_idx, s_stride in enumerate(student_strides):
            t_idx = teacher_strides_map.get(int(s_stride))
            if t_idx is not None and t_idx < len(teacher_channels):
                matched_student_indices.append(s_idx)
                matched_teacher_indices.append(t_idx)

                matched_student_channels.append(student_detect.cv3[s_idx][0].conv.in_channels)
                matched_teacher_channels.append(teacher_channels[t_idx])

        LOGGER.info(f"KD scale matching: student strides {student_strides} -> "
                     f"matched {len(matched_student_indices)} scales")
        LOGGER.info(f"  Student channels: {matched_student_channels}")
        LOGGER.info(f"  Teacher channels: {matched_teacher_channels}")

        if matched_student_channels and matched_teacher_channels:
            distill_loss = DistillationLoss(
                student_channels=matched_student_channels,
                teacher_channels=matched_teacher_channels,
                temperature=kd_temperature,
                device=device)
            LOGGER.info(f"KD loss initialized: alpha={kd_alpha}, beta={kd_beta}, T={kd_temperature}")
        else:
            LOGGER.warning("No matching scales found between teacher and student, skipping feature KD")

        # Store matching info for training loop
        opt._kd_matched_student_indices = matched_student_indices
        opt._kd_matched_teacher_indices = matched_teacher_indices
        opt._kd_teacher_reg_max = teacher_reg_max
    else:
        LOGGER.warning("No teacher outputs found, training without distillation")

    # Optimizer — include adapter parameters if distillation is active
    nbs = 64
    accumulate = max(round(nbs / batch_size), 1)
    hyp['weight_decay'] *= batch_size * accumulate / nbs
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    if distill_loss is not None:
        adapter_params = list(distill_loss.adapters.parameters())
        if adapter_params:
            optimizer.add_param_group({
                'params': adapter_params,
                'lr': hyp['lr0'],
                'weight_decay': hyp['weight_decay']
            })
            LOGGER.info(f"Added {sum(p.numel() for p in adapter_params)} adapter parameters to optimizer")

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)
    elif opt.flat_cos_lr:
        lf = one_flat_cycle(1, hyp['lrf'], epochs)
    elif opt.fixed_lr:
        lf = lambda x: 1.0
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    ema = ModelEMA(model2) if RANK in {-1, 0} else None

    # Resume
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
            if distill_loss is not None and 'distill_loss' in ckpt:
                distill_loss.load_state_dict(ckpt['distill_loss'])
                LOGGER.info("Restored distillation adapter weights from checkpoint")
        del ckpt, csd

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # Trainloader — uses KD dataloader that also loads teacher .pt files
    train_loader, dataset = create_kd_dataloader(train_path,
                                                  imgsz,
                                                  batch_size // WORLD_SIZE,
                                                  gs,
                                                  single_cls,
                                                  hyp=hyp,
                                                  augment=True,
                                                  cache=None if opt.cache == 'val' else opt.cache,
                                                  rect=opt.rect,
                                                  rank=LOCAL_RANK,
                                                  workers=workers,
                                                  image_weights=opt.image_weights,
                                                  close_mosaic=opt.close_mosaic != 0,
                                                  quad=opt.quad,
                                                  prefix=colorstr('train: '),
                                                  shuffle=True,
                                                  min_items=opt.min_items,
                                                  teacher_dir=teacher_dir)
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in {-1, 0}:
        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]
        if not resume:
            model.half().float()
        callbacks.run('on_pretrain_routine_end', labels, names)

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Model attributes
    nl = de_parallel(model).model[-1].nl
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc
    model.hyp = hyp
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    model.names = names

    # Start training
    t0 = time.time()
    nb = len(train_loader)
    nw = max(round(hyp['warmup_epochs'] * nb), 100)
    last_opt_step = -1
    maps = np.zeros(nc)
    results = (0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch - 1
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model)
    callbacks.run('on_train_start')
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting KD training for {epochs} epochs...')

    # Pre-compute which student layers to hook for feature extraction
    student_detect = de_parallel(model).model[-1]
    student_from = student_detect.f if isinstance(student_detect.f, list) else [student_detect.f]
    matched_s_indices = getattr(opt, '_kd_matched_student_indices', [])
    matched_t_indices = getattr(opt, '_kd_matched_teacher_indices', [])
    t_reg_max = getattr(opt, '_kd_teacher_reg_max', 16)
    hook_layers = [student_from[si] for si in matched_s_indices] if matched_s_indices else []

    torch.use_deterministic_algorithms(False)

    for epoch in range(start_epoch, epochs):
        callbacks.run('on_train_epoch_start')
        model.train()

        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)
        if epoch >= (epochs - opt.close_mosaic):
            if dataset.mosaic:
                LOGGER.info("Closing dataloader mosaic")
            dataset.mosaic = False

        # KD only when mosaic is off: teacher outputs are per-image, but
        # mosaic composites 4 images, making teacher logits/features invalid
        use_logit_kd = not dataset.mosaic and kd_alpha > 0 and distill_loss is not None
        use_feat_kd = not dataset.mosaic and kd_beta > 0 and distill_loss is not None

        mloss = torch.zeros(3, device=device)
        mkd = torch.zeros(2, device=device)  # mean KD losses [logit, feat]
        kd_logit_fail_count = 0
        kd_feat_fail_count = 0
        KD_FAIL_THRESHOLD = 10
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%11s' * 9) % ('Epoch', 'GPU_mem', 'box_loss', 'cls_loss', 'dfl_loss',
                                            'kd_logit', 'kd_feat', 'Instances', 'Size'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)
        optimizer.zero_grad()
        reset_net(model)

        for i, (imgs, targets, paths, _, teacher_data) in pbar:
            callbacks.run('on_train_batch_start')
            ni = i + nb * epoch

            imgs = imgs.to(device, non_blocking=True).float() / 255

            # Warmup
            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5 + gs)) // gs * gs
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward with feature hooks for KD
            try:
                with torch.cuda.amp.autocast(amp):
                    # Set up hooks to capture student neck features
                    student_features = {}
                    hooks = []
                    if use_feat_kd and hook_layers:
                        for layer_idx in hook_layers:
                            def make_hook(idx):
                                def hook_fn(module, input, output):
                                    if isinstance(output, (list, tuple)):
                                        student_features[idx] = output[0]
                                    else:
                                        student_features[idx] = output
                                return hook_fn
                            h = de_parallel(model).model[layer_idx].register_forward_hook(make_hook(layer_idx))
                            hooks.append(h)

                    pred = model(imgs)

                    for h in hooks:
                        h.remove()

                    loss, loss_items = compute_loss(pred, targets.to(device))

                    # --- Knowledge Distillation Loss ---
                    kd_loss_val = torch.tensor(0.0, device=device)
                    kd_items = torch.zeros(2, device=device)

                    if distill_loss is not None and teacher_data:
                        # Check if any teacher data was loaded for this batch
                        has_teacher = any(len(td) > 0 for td in teacher_data)

                        if has_teacher and use_logit_kd:
                            # Verify all batch items have teacher logits
                            all_have_logits = all(
                                td and 'det_logits_0' in td for td in teacher_data)

                            if all_have_logits:
                                feats = pred[1] if isinstance(pred, tuple) else pred
                                student_nc = de_parallel(model).model[-1].nc
                                student_reg_max = de_parallel(model).model[-1].reg_max
                                student_no = de_parallel(model).model[-1].no

                                # Per-scale logit KD at matched strides only
                                # (student P3/8 ↔ teacher P3/8, student P4/16 ↔ teacher P4/16)
                                try:
                                    per_scale_logit_loss = torch.tensor(0.0, device=device)
                                    n_matched = 0
                                    for ms_idx, mt_idx in zip(matched_s_indices, matched_t_indices):
                                        # Student cls logits at this scale
                                        s_scale = feats[ms_idx]  # (B, no, H, W)
                                        B = s_scale.shape[0]
                                        s_flat = s_scale.view(B, student_no, -1)
                                        s_cls_scale = s_flat[:, student_reg_max * 4:, :]  # (B, nc, H*W)

                                        # Teacher cls logits at the matched scale
                                        t_logit_key = f'det_logits_{mt_idx}'
                                        t_cls_batch = []
                                        t_valid = True
                                        t_bbox_channels = t_reg_max * 4
                                        for td in teacher_data:
                                            if td and t_logit_key in td:
                                                t_logit = td[t_logit_key].to(device)  # (no, H, W)
                                                t_cls_single = t_logit[t_bbox_channels:, :, :].view(student_nc, -1)  # (nc, H*W)
                                                t_cls_batch.append(t_cls_single)
                                            else:
                                                t_valid = False
                                                break

                                        if not t_valid or len(t_cls_batch) != B:
                                            continue

                                        t_cls_scale = torch.stack(t_cls_batch, dim=0)  # (B, nc, H*W)

                                        # Spatial dims must match (same stride → same H×W)
                                        min_hw = min(s_cls_scale.shape[2], t_cls_scale.shape[2])
                                        scale_loss = distill_loss.logit_kd_loss(
                                            s_cls_scale[:, :, :min_hw],
                                            t_cls_scale[:, :, :min_hw])
                                        per_scale_logit_loss = per_scale_logit_loss + scale_loss
                                        n_matched += 1

                                    if n_matched > 0:
                                        kd_loss_val = kd_alpha * per_scale_logit_loss / n_matched
                                        kd_items[0] = (per_scale_logit_loss / n_matched).detach()

                                except Exception as e:
                                    kd_logit_fail_count += 1
                                    LOGGER.warning(f"KD logit loss failed ({kd_logit_fail_count}x): {e}\n{traceback.format_exc()}")
                                    if kd_logit_fail_count >= KD_FAIL_THRESHOLD:
                                        raise RuntimeError(
                                            f"KD logit loss failed {KD_FAIL_THRESHOLD} times in epoch {epoch}. "
                                            f"Last error: {e}") from e

                        # Feature KD (only when mosaic is off — spatial alignment required)
                        if use_feat_kd and has_teacher and student_features:
                            try:
                                s_feats = [student_features[hook_layers[k]]
                                           for k in range(len(matched_s_indices))]
                                t_feats = []
                                for b_idx in range(imgs.shape[0]):
                                    td = teacher_data[b_idx]
                                    if td:
                                        b_feats = []
                                        for ti in matched_t_indices:
                                            key = f'feat_{ti}'
                                            if key in td:
                                                b_feats.append(td[key].to(device))
                                        if len(b_feats) == len(matched_t_indices):
                                            t_feats.append(b_feats)

                                if len(t_feats) == imgs.shape[0]:
                                    t_feats_stacked = []
                                    for scale_idx in range(len(matched_t_indices)):
                                        t_feats_stacked.append(
                                            torch.stack([t_feats[b][scale_idx] for b in range(len(t_feats))], dim=0))

                                    # Use feature_kd_loss directly to preserve gradient
                                    feat_loss = distill_loss.feature_kd_loss(s_feats, t_feats_stacked)
                                    kd_loss_val = kd_loss_val + kd_beta * feat_loss
                                    kd_items[1] = feat_loss.detach()
                            except Exception as e:
                                kd_feat_fail_count += 1
                                LOGGER.warning(f"KD feature loss failed ({kd_feat_fail_count}x): {e}\n{traceback.format_exc()}")
                                if kd_feat_fail_count >= KD_FAIL_THRESHOLD:
                                    raise RuntimeError(
                                        f"KD feature loss failed {KD_FAIL_THRESHOLD} times in epoch {epoch}. "
                                        f"Last error: {e}") from e

                    loss = loss + kd_loss_val * batch_size

                    if RANK != -1:
                        loss *= WORLD_SIZE
                    if opt.quad:
                        loss *= 4.

                # Backward
                scaler.scale(loss).backward()
            except torch.cuda.OutOfMemoryError:
                LOGGER.warning(f'WARNING: OOM in batch {i}, skipping...')
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                if distill_loss is not None:
                    torch.nn.utils.clip_grad_norm_(distill_loss.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)
                mkd = (mkd * i + kd_items) / (i + 1)
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                pbar.set_description(('%11s' * 2 + '%11.4g' * 7) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, *mkd,
                                      targets.shape[0], imgs.shape[-1]))
                if callbacks.stop_training:
                    return

            reset_net(model)

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()

        if RANK in {-1, 0}:
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:
                results, maps, _ = validate.run(data_dict,
                                                time_step=time_step,
                                                batch_size=batch_size // WORLD_SIZE * 2,
                                                imgsz=imgsz,
                                                half=amp,
                                                model=ema.ema,
                                                single_cls=single_cls,
                                                dataloader=val_loader,
                                                save_dir=save_dir,
                                                plots=False,
                                                callbacks=callbacks,
                                                compute_loss=compute_loss)

            fi = fitness(np.array(results).reshape(1, -1))
            stop = stopper(epoch=epoch, fitness=fi)
            if fi > best_fitness:
                best_fitness = fi
            log_vals = [x.item() if hasattr(x, 'item') else x for x in
                        list(mloss) + list(mkd) + list(results) + lr]
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            if (not nosave) or (final_epoch and not evolve):
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'git': GIT_INFO,
                    'date': datetime.now().isoformat()}
                if distill_loss is not None:
                    ckpt['distill_loss'] = distill_loss.state_dict()

                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        if RANK != -1:
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break

    if RANK in {-1, 0}:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                if f is last:
                    strip_optimizer(f, last_striped)
                else:
                    strip_optimizer(f, best_striped)
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validate.run(
                        data_dict,
                        time_step=time_step,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss)
                    if is_coco:
                        coco_vals = [x.item() if hasattr(x, 'item') else x for x in
                                     list(mloss) + list(mkd) + list(results) + lr]
                        callbacks.run('on_fit_epoch_end', coco_vals, epoch, best_fitness, fi)

        callbacks.run('on_train_end', last, best, epoch, results)

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/detect/su-yolo.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/visdrone.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-high.yaml', help='hyperparameters path')
    parser.add_argument('--time-step', type=int, default=4, help='total time step')
    parser.add_argument('--epochs', type=int, default=300, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=736, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW', 'LION'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=4, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--flat-cos-lr', action='store_true', help='flat cosine LR scheduler')
    parser.add_argument('--fixed-lr', action='store_true', help='fixed LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument('--min-items', type=int, default=0, help='Experimental')
    parser.add_argument('--close-mosaic', type=int, default=100, help='Epoch count before end to disable mosaic (KD needs mosaic off; higher = more KD epochs)')

    # KD-specific arguments
    parser.add_argument('--teacher-outputs', type=str, default=None, help='path to teacher output .pt files directory')
    parser.add_argument('--kd-alpha', type=float, default=1.0, help='logit KD loss weight')
    parser.add_argument('--kd-beta', type=float, default=0.5, help='feature KD loss weight')
    parser.add_argument('--kd-temperature', type=float, default=4.0, help='KD temperature for logit softening')

    # Logger arguments
    parser.add_argument('--entity', default=None, help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    if RANK in {-1, 0}:
        print_args(vars(opt))

    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'
        opt_data = opt.data
        if opt_yaml.is_file():
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location='cpu', weights_only=False)['opt']
        opt = argparse.Namespace(**d)
        opt.cfg, opt.weights, opt.resume = '', str(last), True
        if is_url(opt_data):
            opt.data = check_file(opt_data)
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLO Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    train(opt.hyp, opt, device, callbacks)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
