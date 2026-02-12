#!/bin/bash
# Evaluate a model on all HazyDet splits in one go
# Usage: bash tools/eval_all_splits.sh <weights> <imgsz> <time-step>
#
# Example:
#   bash tools/eval_all_splits.sh results/hazydet/hazydet5_1920_ghost/weights/best.pt 1920 1

WEIGHTS=${1:?Usage: bash tools/eval_all_splits.sh <weights> <imgsz> <time-step>}
IMGSZ=${2:-1920}
TIMESTEP=${3:-1}

echo "============================================================"
echo "  SU-YOLO — Evaluate All HazyDet Splits"
echo "============================================================"
echo "  Weights:    $WEIGHTS"
echo "  Image size: $IMGSZ"
echo "  Time step:  $TIMESTEP"
echo "============================================================"
echo ""

echo ">>> [1/3] Hazy Test Set (synthetic haze)"
echo "------------------------------------------------------------"
python val.py --weights "$WEIGHTS" --data data/hazydet.yaml \
    --imgsz "$IMGSZ" --time-step "$TIMESTEP" --task test
echo ""

echo ">>> [2/3] Real-World Test Set (RDDTS)"
echo "------------------------------------------------------------"
python val.py --weights "$WEIGHTS" --data data/hazydet-real.yaml \
    --imgsz "$IMGSZ" --time-step "$TIMESTEP" --task test
echo ""

echo ">>> [3/3] Clean Test Set (no haze)"
echo "------------------------------------------------------------"
python val.py --weights "$WEIGHTS" --data data/hazydet-clean.yaml \
    --imgsz "$IMGSZ" --time-step "$TIMESTEP" --task test
echo ""

echo "============================================================"
echo "  Done — all 3 splits evaluated"
echo "============================================================"
