"""Lightweight ByteTrack-style multi-object tracker for drone deployment.

Designed for ZCU102 ARM CPU — runs in <2ms per frame with typical drone scenes.
No deep features required; uses IoU-based association + Kalman filter.

Usage:
    from utils.tracker import ByteTracker
    tracker = ByteTracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=60)
    # det: numpy array (N, 6) with [x1, y1, x2, y2, conf, cls]
    tracks = tracker.update(det)
    # tracks: numpy array (M, 7) with [x1, y1, x2, y2, track_id, cls, conf]
"""

import numpy as np
from collections import deque


class KalmanFilter:
    """Lightweight 2D Kalman filter for bounding box tracking.

    State: [cx, cy, w, h, vx, vy, vw, vh] (center + size + velocities)
    Measurement: [cx, cy, w, h]
    """

    def __init__(self):
        # State transition (constant velocity model)
        self.F = np.eye(8, dtype=np.float32)
        self.F[:4, 4:] = np.eye(4, dtype=np.float32)

        # Measurement matrix
        self.H = np.eye(4, 8, dtype=np.float32)

        # Process noise
        self.Q = np.eye(8, dtype=np.float32)
        self.Q[:4, :4] *= 1.0
        self.Q[4:, 4:] *= 0.01

        # Measurement noise
        self.R = np.eye(4, dtype=np.float32) * 1.0

    def init(self, measurement):
        """Initialize state from first measurement [cx, cy, w, h]."""
        x = np.zeros(8, dtype=np.float32)
        x[:4] = measurement
        P = np.eye(8, dtype=np.float32) * 10.0
        P[4:, 4:] *= 100.0
        return x, P

    def predict(self, x, P):
        """Predict next state."""
        x = self.F @ x
        P = self.F @ P @ self.F.T + self.Q
        return x, P

    def update(self, x, P, measurement):
        """Update state with measurement."""
        y = measurement - self.H @ x
        S = self.H @ P @ self.H.T + self.R
        K = P @ self.H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(8, dtype=np.float32) - K @ self.H) @ P
        return x, P


def xyxy_to_cxcywh(bbox):
    """Convert [x1, y1, x2, y2] to [cx, cy, w, h]."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return np.array([cx, cy, w, h], dtype=np.float32)


def cxcywh_to_xyxy(cxcywh):
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = cxcywh
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


def iou_batch(bb_a, bb_b):
    """Compute IoU between two sets of boxes.

    Args:
        bb_a: (N, 4) array [x1, y1, x2, y2]
        bb_b: (M, 4) array [x1, y1, x2, y2]

    Returns:
        (N, M) IoU matrix
    """
    if len(bb_a) == 0 or len(bb_b) == 0:
        return np.empty((len(bb_a), len(bb_b)), dtype=np.float32)

    bb_a = np.asarray(bb_a, dtype=np.float32)
    bb_b = np.asarray(bb_b, dtype=np.float32)

    # Intersection
    x1 = np.maximum(bb_a[:, 0:1], bb_b[:, 0:1].T)
    y1 = np.maximum(bb_a[:, 1:2], bb_b[:, 1:2].T)
    x2 = np.minimum(bb_a[:, 2:3], bb_b[:, 2:3].T)
    y2 = np.minimum(bb_a[:, 3:4], bb_b[:, 3:4].T)

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Union
    area_a = (bb_a[:, 2] - bb_a[:, 0]) * (bb_a[:, 3] - bb_a[:, 1])
    area_b = (bb_b[:, 2] - bb_b[:, 0]) * (bb_b[:, 3] - bb_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    return inter / (union + 1e-7)


def linear_assignment(cost_matrix):
    """Simple Hungarian-style assignment using scipy if available, else greedy.

    For drone scenes with <100 objects, greedy is sufficient and faster on ARM.
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int)

    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return np.column_stack((row_ind, col_ind))
    except ImportError:
        # Greedy fallback for deployment without scipy
        matches = []
        rows, cols = cost_matrix.shape
        used_cols = set()
        # Sort by lowest cost
        flat_indices = np.argsort(cost_matrix.ravel())
        for idx in flat_indices:
            r, c = divmod(idx, cols)
            if r not in {m[0] for m in matches} and c not in used_cols:
                matches.append((r, c))
                used_cols.add(c)
                if len(matches) == min(rows, cols):
                    break
        return np.array(matches, dtype=int) if matches else np.empty((0, 2), dtype=int)


class Track:
    """Single object track with Kalman filter state."""

    _next_id = 1

    def __init__(self, bbox, score, cls, kf):
        self.kf = kf
        measurement = xyxy_to_cxcywh(bbox)
        self.x, self.P = kf.init(measurement)
        self.score = score
        self.cls = int(cls)
        self.track_id = Track._next_id
        Track._next_id += 1
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.is_activated = False

    def predict(self):
        self.x, self.P = self.kf.predict(self.x, self.P)
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox, score, cls):
        measurement = xyxy_to_cxcywh(bbox)
        self.x, self.P = self.kf.update(self.x, self.P, measurement)
        self.score = score
        self.cls = int(cls)
        self.hits += 1
        self.time_since_update = 0
        self.is_activated = True

    @property
    def bbox(self):
        return cxcywh_to_xyxy(self.x[:4])

    @property
    def predicted_bbox(self):
        return cxcywh_to_xyxy(self.x[:4])


class ByteTracker:
    """ByteTrack-style tracker optimized for drone surveillance on ZCU102.

    Two-stage association:
    1. High-confidence detections matched to existing tracks via IoU
    2. Low-confidence detections matched to remaining tracks (recovers occluded objects)

    Args:
        track_thresh: Confidence threshold separating high/low detections (default 0.5)
        match_thresh: IoU threshold for matching (default 0.8)
        track_buffer: Frames to keep lost tracks alive (default 30 = 0.5s at 60fps)
        frame_rate: Expected frame rate for buffer scaling (default 60)
    """

    def __init__(self, track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=60):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.low_thresh = 0.1  # minimum confidence to consider
        self.track_buffer = int(track_buffer * frame_rate / 30.0)  # scale buffer to fps
        self.max_time_lost = self.track_buffer

        self.kf = KalmanFilter()
        self.active_tracks = []
        self.lost_tracks = []
        self.frame_id = 0

    def update(self, detections):
        """Update tracker with new detections.

        Args:
            detections: numpy array (N, 6) with [x1, y1, x2, y2, conf, cls]
                        or empty array if no detections

        Returns:
            numpy array (M, 7) with [x1, y1, x2, y2, track_id, cls, conf]
        """
        self.frame_id += 1

        if detections is None or len(detections) == 0:
            # Predict all tracks forward, move stale ones to lost
            for track in self.active_tracks:
                track.predict()
            self._expire_tracks()
            return np.empty((0, 7), dtype=np.float32)

        detections = np.asarray(detections, dtype=np.float32)
        scores = detections[:, 4]

        # Split into high and low confidence
        high_mask = scores >= self.track_thresh
        low_mask = (scores >= self.low_thresh) & (~high_mask)

        det_high = detections[high_mask]
        det_low = detections[low_mask]

        # Predict existing tracks
        for track in self.active_tracks + self.lost_tracks:
            track.predict()

        # --- Stage 1: Match high-conf detections to active tracks ---
        matched_a, unmatched_tracks_a, unmatched_dets_a = self._associate(
            self.active_tracks, det_high, thresh=self.match_thresh
        )

        # Update matched tracks
        for track_idx, det_idx in matched_a:
            self.active_tracks[track_idx].update(
                det_high[det_idx, :4], det_high[det_idx, 4], det_high[det_idx, 5]
            )

        # --- Stage 2: Match low-conf detections to remaining active tracks ---
        remaining_tracks = [self.active_tracks[i] for i in unmatched_tracks_a]
        matched_b, unmatched_tracks_b, _ = self._associate(
            remaining_tracks, det_low, thresh=0.5
        )

        for track_idx, det_idx in matched_b:
            remaining_tracks[track_idx].update(
                det_low[det_idx, :4], det_low[det_idx, 4], det_low[det_idx, 5]
            )

        # Tracks that weren't matched in either stage → lost
        still_unmatched = [remaining_tracks[i] for i in unmatched_tracks_b]

        # --- Stage 3: Match unmatched high-conf dets to lost tracks ---
        unmatched_high_dets = det_high[unmatched_dets_a]
        matched_c, unmatched_lost, unmatched_dets_c = self._associate(
            self.lost_tracks, unmatched_high_dets, thresh=0.7
        )

        # Re-activate matched lost tracks
        for track_idx, det_idx in matched_c:
            self.lost_tracks[track_idx].update(
                unmatched_high_dets[det_idx, :4],
                unmatched_high_dets[det_idx, 4],
                unmatched_high_dets[det_idx, 5],
            )
            self.active_tracks.append(self.lost_tracks[track_idx])

        # Remaining unmatched lost tracks stay lost
        remaining_lost = [self.lost_tracks[i] for i in unmatched_lost]

        # --- Create new tracks for unmatched high-conf detections ---
        for det_idx in unmatched_dets_c:
            det = unmatched_high_dets[det_idx]
            if det[4] >= self.track_thresh:
                new_track = Track(det[:4], det[4], det[5], self.kf)
                new_track.is_activated = True
                self.active_tracks.append(new_track)

        # Move unmatched active tracks to lost
        self.lost_tracks = remaining_lost + still_unmatched
        self.active_tracks = [t for t in self.active_tracks if t.time_since_update == 0]

        # Expire old lost tracks
        self._expire_tracks()

        # Build output
        outputs = []
        for track in self.active_tracks:
            if track.is_activated:
                bbox = track.bbox
                outputs.append([
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    float(track.track_id), float(track.cls), track.score
                ])

        return np.array(outputs, dtype=np.float32) if outputs else np.empty((0, 7), dtype=np.float32)

    def _associate(self, tracks, detections, thresh):
        """Associate tracks with detections using IoU.

        Returns:
            matches: list of (track_idx, det_idx) pairs
            unmatched_tracks: list of unmatched track indices
            unmatched_dets: list of unmatched detection indices
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Get predicted bboxes for tracks
        track_bboxes = np.array([t.predicted_bbox for t in tracks])
        det_bboxes = detections[:, :4]

        # Compute IoU cost matrix
        iou_matrix = iou_batch(track_bboxes, det_bboxes)
        cost_matrix = 1.0 - iou_matrix

        # Mask out pairs below threshold
        cost_matrix[iou_matrix < (1.0 - thresh)] = 1.0 + 1e-5

        # Solve assignment
        assignments = linear_assignment(cost_matrix)

        matches = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))

        for row, col in assignments:
            if cost_matrix[row, col] <= 1.0:
                matches.append((row, col))
                unmatched_tracks.discard(row)
                unmatched_dets.discard(col)

        return matches, list(unmatched_tracks), list(unmatched_dets)

    def _expire_tracks(self):
        """Remove tracks that have been lost too long."""
        self.lost_tracks = [
            t for t in self.lost_tracks if t.time_since_update <= self.max_time_lost
        ]

    def reset(self):
        """Reset tracker state (e.g., for new video)."""
        self.active_tracks = []
        self.lost_tracks = []
        self.frame_id = 0
        Track._next_id = 1
