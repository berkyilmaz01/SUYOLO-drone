"""
Generate a 30 FPS detection demo video from test-set images.

Usage:
    python make_demo_video.py \
        --weights runs/train/hazydet/weights/best.pt \
        --source ../HazyDet/test/images \
        --data data/hazydet.yaml \
        --imgsz 1920 1920 \
        --conf-thres 0.25 \
        --output runs/demo/hazydet_demo.mp4

The script runs SUYOLO inference on every image in --source (sorted by
filename), draws bounding boxes with class labels and confidence, overlays
a compact HUD (frame counter, detection stats, model info), and writes
all frames into a single MP4 video at 30 FPS.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS
from utils.general import (
    LOGGER,
    check_img_size,
    letterbox,
    non_max_suppression,
    scale_boxes,
)
from utils.plots import Colors
from utils.torch_utils import select_device

from spikingjelly.activation_based.functional import reset_net
from models.spike import set_time_step

colors = Colors()

# ── class-specific colours (BGR) ──────────────────────────────────────
CLASS_COLORS = {
    0: (0, 200, 255),   # car   – amber
    1: (0, 255, 100),   # truck – green
    2: (255, 100, 0),   # bus   – blue
}
CLASS_NAMES = {0: "car", 1: "truck", 2: "bus"}

# ── HUD styling ───────────────────────────────────────────────────────
HUD_BG = (30, 30, 30)
HUD_TEXT = (240, 240, 240)
HUD_ACCENT = (0, 200, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_box(img, box, cls, conf, lw=2):
    """Draw a single detection box with label."""
    x1, y1, x2, y2 = map(int, box)
    color = CLASS_COLORS.get(int(cls), (200, 200, 200))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, lw, cv2.LINE_AA)

    label = f"{CLASS_NAMES.get(int(cls), '?')} {conf:.2f}"
    tf = max(lw - 1, 1)
    fs = lw / 3
    (tw, th), _ = cv2.getTextSize(label, FONT, fs, tf)
    # label background
    outside = y1 - th - 4 >= 0
    lbl_y = y1 - th - 4 if outside else y1
    cv2.rectangle(img, (x1, lbl_y), (x1 + tw + 2, lbl_y + th + 4), color, -1, cv2.LINE_AA)
    cv2.putText(img, label, (x1 + 1, lbl_y + th + 1), FONT, fs, (255, 255, 255), tf, cv2.LINE_AA)


def draw_hud(img, frame_idx, total_frames, n_det, cls_counts, fps_model, elapsed):
    """Overlay a compact heads-up display on the frame."""
    h, w = img.shape[:2]
    pad = 12
    line_h = 28

    # ── top-left: frame counter + model speed ─────────────────────────
    lines_tl = [
        f"Frame {frame_idx + 1}/{total_frames}",
        f"Detections: {n_det}",
        f"Inference: {fps_model:.1f} ms",
    ]
    box_h = pad * 2 + line_h * len(lines_tl)
    box_w = 320
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), HUD_BG, -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    for i, txt in enumerate(lines_tl):
        cv2.putText(img, txt, (pad, pad + line_h * (i + 1) - 6), FONT, 0.65, HUD_TEXT, 1, cv2.LINE_AA)

    # ── top-right: per-class counts ───────────────────────────────────
    lines_tr = [f"{CLASS_NAMES[c]}: {cnt}" for c, cnt in sorted(cls_counts.items())]
    if lines_tr:
        tr_box_w = 200
        tr_box_h = pad * 2 + line_h * len(lines_tr)
        overlay2 = img.copy()
        cv2.rectangle(overlay2, (w - tr_box_w, 0), (w, tr_box_h), HUD_BG, -1)
        cv2.addWeighted(overlay2, 0.65, img, 0.35, 0, img)
        for i, txt in enumerate(lines_tr):
            cls_id = sorted(cls_counts.keys())[i]
            clr = CLASS_COLORS.get(cls_id, HUD_TEXT)
            cv2.putText(img, txt, (w - tr_box_w + pad, pad + line_h * (i + 1) - 6),
                        FONT, 0.65, clr, 1, cv2.LINE_AA)

    # ── bottom-left: model tag ────────────────────────────────────────
    tag = "SUYOLO  |  HazyDet Benchmark  |  SNN 0.5M params"
    (tw, th), _ = cv2.getTextSize(tag, FONT, 0.55, 1)
    tag_h = th + pad * 2
    overlay3 = img.copy()
    cv2.rectangle(overlay3, (0, h - tag_h), (tw + pad * 2, h), HUD_BG, -1)
    cv2.addWeighted(overlay3, 0.65, img, 0.35, 0, img)
    cv2.putText(img, tag, (pad, h - pad), FONT, 0.55, HUD_ACCENT, 1, cv2.LINE_AA)

    # ── bottom-right: progress bar ────────────────────────────────────
    bar_w = 300
    bar_h = 8
    bx = w - bar_w - pad
    by = h - pad - bar_h
    progress = (frame_idx + 1) / max(total_frames, 1)
    cv2.rectangle(img, (bx, by), (bx + bar_w, by + bar_h), (80, 80, 80), -1)
    cv2.rectangle(img, (bx, by), (bx + int(bar_w * progress), by + bar_h), HUD_ACCENT, -1)


def collect_images(source_dir):
    """Collect and sort image paths from a directory."""
    exts = set(IMG_FORMATS)
    imgs = []
    src = Path(source_dir)
    for f in sorted(src.iterdir()):
        if f.suffix.lower().lstrip('.') in exts:
            imgs.append(f)
    LOGGER.info(f"Found {len(imgs)} images in {source_dir}")
    return imgs


@torch.no_grad()
def run(opt):
    set_time_step(opt.time_step)

    # ── resolve paths ─────────────────────────────────────────────────
    source = Path(opt.source)
    output = Path(opt.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # ── load model ────────────────────────────────────────────────────
    device = select_device(opt.device)
    model = DetectMultiBackend(opt.weights, device=device, data=opt.data, fp16=opt.half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(opt.imgsz, s=stride)

    # warmup
    model.warmup(imgsz=(1, 3, *imgsz))

    # ── collect images ────────────────────────────────────────────────
    image_paths = collect_images(source)
    if not image_paths:
        LOGGER.error("No images found — aborting.")
        return
    total = len(image_paths)

    # ── determine output resolution from first image ──────────────────
    sample = cv2.imread(str(image_paths[0]))
    out_h, out_w = sample.shape[:2]

    # if images are very large, scale down for a reasonable video size
    max_dim = opt.max_video_dim
    if max(out_h, out_w) > max_dim:
        scale = max_dim / max(out_h, out_w)
        out_w = int(out_w * scale)
        out_h = int(out_h * scale)

    # ── video writer ──────────────────────────────────────────────────
    fps = opt.fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        LOGGER.error(f"Cannot open video writer for {output}")
        return
    LOGGER.info(f"Writing {total} frames → {output}  ({out_w}x{out_h} @ {fps} FPS)")

    # ── inference loop ────────────────────────────────────────────────
    t_total = time.time()
    for idx, img_path in enumerate(image_paths):
        # read original image
        im0 = cv2.imread(str(img_path))
        if im0 is None:
            LOGGER.warning(f"Skipping unreadable image: {img_path}")
            continue

        # preprocess
        img = letterbox(im0, imgsz, stride=stride, auto=pt)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC→CHW, BGR→RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if model.fp16 else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # inference
        t1 = time.time()
        reset_net(model)
        pred = model(img)
        pred = pred[0][1] if isinstance(pred[0], list) else pred[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, max_det=opt.max_det)
        t_inf = (time.time() - t1) * 1000  # ms

        # process detections
        det = pred[0]
        cls_counts = {}
        n_det = 0
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
            n_det = len(det)
            for *xyxy, conf, cls in det:
                c = int(cls)
                cls_counts[c] = cls_counts.get(c, 0) + 1
                draw_box(im0, xyxy, c, float(conf), lw=2)

        # resize if needed
        if (im0.shape[1], im0.shape[0]) != (out_w, out_h):
            im0 = cv2.resize(im0, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        # overlay HUD
        draw_hud(im0, idx, total, n_det, cls_counts, t_inf, time.time() - t_total)

        # write frame
        writer.write(im0)

        if (idx + 1) % 100 == 0 or idx == total - 1:
            LOGGER.info(f"[{idx + 1}/{total}]  {img_path.name}  {n_det} dets  {t_inf:.1f}ms")

    writer.release()
    elapsed = time.time() - t_total
    LOGGER.info(f"Done. {total} frames in {elapsed:.1f}s  →  {output}  "
                f"({os.path.getsize(output) / 1e6:.1f} MB)")


def parse_opt():
    p = argparse.ArgumentParser(description="SUYOLO detection demo video generator")
    p.add_argument("--weights", type=str, required=True, help="Path to model weights (.pt)")
    p.add_argument("--source", type=str, required=True, help="Directory of test images")
    p.add_argument("--data", type=str, default="data/hazydet.yaml", help="Dataset YAML")
    p.add_argument("--imgsz", nargs="+", type=int, default=[1920, 1920], help="Inference size (h w)")
    p.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--max-det", type=int, default=1000, help="Max detections per image")
    p.add_argument("--device", default="0", help="CUDA device or cpu")
    p.add_argument("--half", action="store_true", help="FP16 inference")
    p.add_argument("--output", type=str, default="runs/demo/hazydet_demo.mp4", help="Output video path")
    p.add_argument("--fps", type=int, default=30, help="Video frame rate")
    p.add_argument("--max-video-dim", type=int, default=1920, help="Cap output video resolution")
    p.add_argument("--time-step", type=int, default=4, help="SNN time steps")
    opt = p.parse_args()
    if len(opt.imgsz) == 1:
        opt.imgsz *= 2
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
