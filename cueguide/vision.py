#!/usr/bin/env python3
"""
pool_cue_guide.py

Vision-based pool cue guide (Eight-ball) inspired by Weatherford's algorithm.
Features:
- Homographic rectification to a top-down view
- Table, rail, pocket, and ball detection
- Cue ball, 8-ball identification; stripes vs. solids via cue-based white balance
- Cue guides: draw aim lines; evaluate and suggest "best shot" (direct, with simple one-bank fallback)
- Outputs annotated images and a JSON summary

Dependencies:
    pip install opencv-python numpy

Usage:
    python pool_cue_guide.py --image input.jpg --outdir out

Notes:
- This is an engineering reference implementation; parameters may require tuning per table/lighting.
- Bank shot evaluation is simplified (single-rail reflection) and heuristic.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display on a server

# -----------------------------
# Geometry helpers
# -----------------------------

def order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 corner points as [top-left, top-right, bottom-right, bottom-left].
    pts: (4,2) float32
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # top-left: smallest sum
    rect[2] = pts[np.argmax(s)]      # bottom-right: largest sum
    rect[1] = pts[np.argmin(diff)]   # top-right: smallest diff
    rect[3] = pts[np.argmax(diff)]   # bottom-left: largest diff
    return rect

def line_intersection(p1, p2, p3, p4):
    """Return intersection point of lines p1-p2 and p3-p4 (each [x,y]) or None if parallel."""
    x1,y1 = p1; x2,y2 = p2; x3,y3 = p3; x4,y4 = p4
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
    return np.array([px, py], dtype=np.float32)

def reflect_point_across_line(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Reflect point pt across infinite line through a-b.
    """
    ap = pt - a
    ab = b - a
    ab_norm = ab / (np.linalg.norm(ab) + 1e-9)
    proj = a + np.dot(ap, ab_norm) * ab_norm
    mirrored = proj + (proj - pt)
    return mirrored

def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

# -----------------------------
# Vision helpers
# -----------------------------

def largest_contour(mask: np.ndarray):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    return cnt

def mask_from_center_color(img_bgr: np.ndarray, ksize: int = 0) -> np.ndarray:
    """
    Estimate tabletop color from a central box; create mask of pixels 'close' to that color (HSV distance).
    """
    h, w = img_bgr.shape[:2]
    cx0, cy0 = int(w*0.3), int(h*0.3)
    cx1, cy1 = int(w*0.7), int(h*0.7)
    center = img_bgr[cy0:cy1, cx0:cx1]
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    med = np.median(hsv.reshape(-1, 3), axis=0)

    full_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Distance in HSV with emphasis on hue closeness; thresholds tuned empirically.
    dh = np.minimum(np.abs(full_hsv[:,:,0] - med[0]), 180 - np.abs(full_hsv[:,:,0] - med[0]))
    ds = np.abs(full_hsv[:,:,1] - med[1])
    dv = np.abs(full_hsv[:,:,2] - med[2])
    # Weight hue most, then saturation; allow value variation due to lighting
    dist_hsv = 2.0*dh + 1.0*(ds/2.0) + 0.3*(dv/3.0)
    mask = (dist_hsv < 30).astype(np.uint8) * 255

    # Morphological clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def find_table_corners(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Returns 4x2 float32 array of table corners in original image coordinates.
    """
    mask = mask_from_center_color(img_bgr)
    cnt = largest_contour(mask)
    if cnt is None or cv2.contourArea(cnt) < 0.1 * img_bgr.shape[0] * img_bgr.shape[1]:
        return None

    eps = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True)
    if len(approx) < 4:
        # try convex hull then approx
        hull = cv2.convexHull(cnt)
        approx = cv2.approxPolyDP(hull, eps, True)

    # If >4 points, choose the four that form the largest quadrilateral by area
    if len(approx) != 4:
        # fallback: minAreaRect box
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        corners = box.astype(np.float32)
    else:
        corners = approx.reshape(-1, 2).astype(np.float32)

    return order_corners(corners)

def warp_to_topdown(img_bgr: np.ndarray, corners: np.ndarray, target_short: int = 900) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warps the image so the table becomes top-down.
    Returns rectified image and the 3x3 homography H.
    We assume pool table length ~ 2x width; we infer output size from corner distances.
    """
    (tl, tr, br, bl) = corners
    widthA = dist(br, bl)
    widthB = dist(tr, tl)
    heightA = dist(tr, br)
    heightB = dist(tl, bl)
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))

    # Constrain to ratio ~ 2:1 (length:width). Use longer side as width.
    if maxW < maxH:
        maxW, maxH = maxH, maxW
    # Normalize output scale
    scale = target_short / float(maxH)
    maxW = int(maxW * scale)
    maxH = int(maxH * scale)

    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype="float32")

    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    rectified = cv2.warpPerspective(img_bgr, H, (maxW, maxH))
    return rectified, H

# -----------------------------
# Ball & pocket detection
# -----------------------------

@dataclass(eq=False)  # identity equality: fields hold numpy arrays
class Ball:
    center: np.ndarray   # (x,y)
    radius: float
    mean_bgr: Tuple[float, float, float]
    label: str = "unknown"  # 'cue','eight','solid','stripe','unknown'
    id: str = ""            # stable user-facing label (B1, B2, ...)

def detect_balls(rect_bgr: np.ndarray, table_mask: Optional[np.ndarray] = None) -> List[Ball]:
    """
    Detect balls in rectified image using HoughCircles; return Ball list.
    """
    img = rect_bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    h, w = gray.shape
    # Ball radius heuristic: based on table height; typical diameter ~= 1/16 - 1/20 of table width in 2D
    min_r = max(6, int(0.015 * w))
    max_r = int(0.035 * w)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_r*2,
                               param1=120, param2=20, minRadius=min_r, maxRadius=max_r)

    balls: List[Ball] = []
    if circles is not None:
        for c in np.round(circles[0, :]).astype("int"):
            x, y, r = c
            if x-r < 0 or y-r < 0 or x+r >= w or y+r >= h:
                continue
            if table_mask is not None and table_mask[y, x] == 0:
                continue
            patch = img[max(0, y-r):min(h, y+r), max(0, x-r):min(w, x+r)]
            mean_bgr = tuple(np.mean(patch.reshape(-1, 3), axis=0).tolist())
            balls.append(Ball(center=np.array([x, y], dtype=np.float32), radius=float(r), mean_bgr=mean_bgr))

    return balls

def detect_table_mask(rect_bgr: np.ndarray) -> np.ndarray:
    """Mask of the tabletop in rectified view."""
    mask = mask_from_center_color(rect_bgr)
    # Keep the largest contiguous region as table
    cnt = largest_contour(mask)
    table_mask = np.zeros_like(mask)
    if cnt is not None:
        cv2.drawContours(table_mask, [cnt], -1, 255, thickness=cv2.FILLED)
    return table_mask

def detect_pockets(rect_bgr: np.ndarray, table_mask: np.ndarray) -> List[np.ndarray]:
    """
    Approximate 6 pocket centers: 4 corners + midpoints on long edges.
    Refines by searching for dark circular regions near these anchor points.
    """
    h, w = table_mask.shape
    # Find table bounding box
    ys, xs = np.where(table_mask > 0)
    if len(xs) == 0:
        return []
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    # Anchor points
    anchors = [
        np.array([x0, y0], dtype=np.float32),  # TL
        np.array([x1, y0], dtype=np.float32),  # TR
        np.array([x1, y1], dtype=np.float32),  # BR
        np.array([x0, y1], dtype=np.float32),  # BL
        np.array([(x0+x1)//2, y0], dtype=np.float32),  # top middle
        np.array([(x0+x1)//2, y1], dtype=np.float32)   # bottom middle
    ]

    # Search small window around anchors for darkest circle center
    pockets = []
    gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
    for a in anchors:
        ax, ay = int(a[0]), int(a[1])
        wx, wy = int(0.08*w), int(0.08*h)
        xA, xB = max(0, ax-wx), min(w, ax+wx)
        yA, yB = max(0, ay-wy), min(h, ay+wy)
        roi = gray[yA:yB, xA:xB]
        if roi.size == 0:
            pockets.append(a)
            continue
        # find min-intensity pixel (darkest)
        minVal, _, minLoc, _ = cv2.minMaxLoc(roi)
        px = xA + minLoc[0]
        py = yA + minLoc[1]
        pockets.append(np.array([px, py], dtype=np.float32))

    # Deduplicate (just in case): keep 6 by KMeans-like grouping
    # Simple approach: sort by x,y and take first 6
    pockets = pockets[:6]
    return pockets

# -----------------------------
# Classification
# -----------------------------

def classify_balls(rect_bgr: np.ndarray, balls: List[Ball]) -> None:
    """
    Assign 'cue', 'eight', 'stripe', 'solid' labels.
    - Cue ball: highest brightness (value) and near-neutral color
    - Eight ball: lowest brightness
    - Stripes vs solids: cue-based white-balance -> percentage of near-white pixels within ball ROI
    """
    if not balls:
        return

    # Heuristic for cue/eight by intensity
    vals = []
    for b in balls:
        bgr = np.array(b.mean_bgr, dtype=np.float32)
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV).astype(np.float32)
        v = float(hsv[0,0,2])
        vals.append(v)
    # Identify cue and eight
    cue_idx = int(np.argmax(vals))
    eight_idx = int(np.argmin(vals))
    for i, b in enumerate(balls):
        if i == cue_idx:
            b.label = 'cue'
        elif i == eight_idx:
            b.label = 'eight'

    # White-balance using cue ball
    cue = balls[cue_idx]
    # Gain to map cue to white (255,255,255) on average
    cue_bgr = np.array(cue.mean_bgr, dtype=np.float32) + 1e-6
    gains = 255.0 / cue_bgr

    h, w = rect_bgr.shape[:2]
    for b in balls:
        if b.label in ('cue', 'eight'):
            continue
        x, y, r = int(b.center[0]), int(b.center[1]), int(b.radius)
        xA, xB = max(0, x-r), min(w, x+r)
        yA, yB = max(0, y-r), min(h, y+r)
        patch = rect_bgr[yA:yB, xA:xB].astype(np.float32)
        # Apply gains
        wb = np.clip(patch * gains, 0, 255).astype(np.uint8)
        # Compute whiteness ratio
        hsv = cv2.cvtColor(wb, cv2.COLOR_BGR2HSV)
        # define "white": low saturation & high value
        white_mask = ((hsv[:,:,1] < 40) & (hsv[:,:,2] > 180)).astype(np.uint8)
        # mask to circle only
        yy, xx = np.ogrid[0:yB-yA, 0:xB-xA]
        cy, cx = (y - yA), (x - xA)
        circle_mask = ((xx - cx)**2 + (yy - cy)**2 <= r*r).astype(np.uint8)
        wm = cv2.bitwise_and(white_mask, white_mask, mask=circle_mask)
        total = float(np.sum(circle_mask))
        white_ratio = (float(np.sum(wm)) / total) if total > 1.0 else 0.0
        b.label = 'stripe' if white_ratio > 0.20 else 'solid'

# -----------------------------
# Shot calculation
# -----------------------------

@dataclass
class Shot:
    target_label: str
    ball_center: Tuple[float, float]
    pocket_center: Tuple[float, float]
    ghost_point: Tuple[float, float]
    cue_to_ghost_clear: bool
    obj_to_pocket_clear: bool
    path_score: float  # higher is better
    is_combo: bool = False
    combo_mid_ball_center: Optional[Tuple[float, float]] = None

def segment_clear(a: np.ndarray, b: np.ndarray, balls: List[Ball], ignore_indices: List[int], clearance: float) -> bool:
    """
    Check if segment a-b is clear of collisions with balls not in ignore_indices.
    clearance: minimal distance from segment to ball center (in pixels).
    """
    for idx, ball in enumerate(balls):
        if idx in ignore_indices:
            continue
        p = ball.center
        # distance from point p to segment ab
        ab = b - a
        t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9)
        t = clamp01(float(t))
        closest = a + t * ab
        if dist(p, closest) < clearance + ball.radius:
            return False
    return True


def compute_combo_shot(rect_bgr: np.ndarray, balls: List[Ball], cue_idx: int, b1_idx: int, b2_idx: int, pocket: np.ndarray, R: float):
    # \"\"\"Return a Shot for a simple two-ball combo cue->b1->b2->pocket if paths are clear, else None.
    # Geometry:
    #   - Desired b2 path is toward pocket: u = unit(pocket - b2)
    #   - At impact, centers are colinear along u; b1 impact-center (on b2) is ic = b2 - u*(2R)
    #   - b1 must arrive with center at ic; cue aims at b1 with ghost point g_cue = b1 - v*(2R), v = unit(ic - b1)
    # Clearance checks:
    #   - b2 -> pocket clear
    #   - b1 -> ic clear
    #   - cue -> g_cue clear
    # \"\"\"
    h, w = rect_bgr.shape[:2]
    cue = balls[cue_idx]
    b1 = balls[b1_idx]
    b2 = balls[b2_idx]

    b2c = b2.center.astype(np.float32)
    p = pocket.astype(np.float32)
    u = p - b2c
    nu = u / (np.linalg.norm(u) + 1e-9)
    impact_center = b2c - nu * (2.0 * R)

    # Ghost for cue to strike b1
    v = impact_center - b1.center
    nv = v / (np.linalg.norm(v) + 1e-9)
    ghost_cue = b1.center - nv * (2.0 * R)

    # Clearance checks (ignore cue, b1, b2 as needed)
    if not segment_clear(b2.center, p, balls, ignore_indices=[cue_idx, b1_idx, b2_idx], clearance=R*0.5):
        return None
    if not segment_clear(b1.center, impact_center, balls, ignore_indices=[cue_idx, b1_idx, b2_idx], clearance=R*0.6):
        return None
    if not segment_clear(cue.center, ghost_cue, balls, ignore_indices=[cue_idx, b1_idx, b2_idx], clearance=R*0.9):
        return None

    # Score: penalize combos vs direct; prefer shorter segments and reasonable angles
    d_cue = dist(cue.center, ghost_cue)
    d_b1 = dist(b1.center, impact_center)
    d_b2 = dist(b2.center, p)

    # Angle coherence: cue->b1 vs b2->pocket
    a1 = impact_center - cue.center
    a2 = p - b2.center
    denom = (np.linalg.norm(a1) * np.linalg.norm(a2)) + 1e-9
    cosang = float(np.dot(a1, a2) / denom)
    angle_score = (cosang + 1.0) / 2.0  # 0..1

    score = 0.65 * (1.2*angle_score + 0.5/(d_cue+1e-3) + 0.3/(d_b1+1e-3) + 0.2/(d_b2+1e-3))

    return Shot(
        target_label=balls[b2_idx].label,
        ball_center=(float(b2.center[0]), float(b2.center[1])),
        pocket_center=(float(p[0]), float(p[1])),
        ghost_point=(float(ghost_cue[0]), float(ghost_cue[1])),
        cue_to_ghost_clear=True,
        obj_to_pocket_clear=True,
        path_score=score,
        is_combo=True,
        combo_mid_ball_center=(float(b1.center[0]), float(b1.center[1]))
    )


def compute_shots(rect_bgr: np.ndarray, balls: List[Ball], pockets: List[np.ndarray]) -> List[Shot]:
    """
    Compute direct shots (and simple one-bank) for all non-cue/non-eight balls to all pockets.
    """
    # Find cue index
    cue_idx = next((i for i,b in enumerate(balls) if b.label == 'cue'), None)
    if cue_idx is None:
        return []
    cue = balls[cue_idx]
    R = float(np.median([b.radius for b in balls])) if balls else 10.0

    candidates: List[Shot] = []
    for bi, b in enumerate(balls):
        if b.label in ('cue', 'unknown'):
            continue
        # You may wish to ignore 8-ball unless it's the last ball; here we include but penalize
        for p in pockets:
            p = p.astype(np.float32)
            # Ghost point for potting: center-to-center line passes through pocket
            v = b.center - p
            nv = v / (np.linalg.norm(v) + 1e-9)
            ghost = b.center + nv * (2.0 * R)  # ~2R center-offset
            # Check paths
            cue_clear = segment_clear(cue.center, ghost, balls, ignore_indices=[bi, cue_idx], clearance=R*0.9)
            obj_clear = segment_clear(b.center, p, balls, ignore_indices=[bi, cue_idx], clearance=R*0.5)
            score = 0.0
            if cue_clear and obj_clear:
                # Shorter, straighter paths score higher
                d1 = dist(cue.center, ghost)
                d2 = dist(b.center, p)
                # angle between vectors cue->ghost and ball->pocket (prefer small)
                a1 = ghost - cue.center
                a2 = p - b.center
                denom = (np.linalg.norm(a1) * np.linalg.norm(a2)) + 1e-9
                cosang = float(np.dot(a1, a2) / denom)
                angle_score = (cosang + 1.0) / 2.0  # [0..1]
                # Penalize 8-ball a bit (heuristic)
                eight_pen = 0.5 if b.label == 'eight' else 1.0
                score = eight_pen * (1.5*angle_score + 0.7/(d1+1e-3) + 0.3/(d2+1e-3))
                candidates.append(Shot(
                    target_label=b.label,
                    ball_center=(float(b.center[0]), float(b.center[1])),
                    pocket_center=(float(p[0]), float(p[1])),
                    ghost_point=(float(ghost[0]), float(ghost[1])),
                    cue_to_ghost_clear=True,
                    obj_to_pocket_clear=True,
                    path_score=score
                ))
            else:
                # Try one simple bank: reflect pocket across nearest rail (top/bottom/left/right)
                h, w = rect_bgr.shape[:2]
                rails = [
                    (np.array([0,0], np.float32), np.array([w,0], np.float32)),   # top y=0
                    (np.array([0,h], np.float32), np.array([w,h], np.float32)),   # bottom y=h
                    (np.array([0,0], np.float32), np.array([0,h], np.float32)),   # left x=0
                    (np.array([w,0], np.float32), np.array([w,h], np.float32)),   # right x=w
                ]
                best_bank = None
                best_bank_score = 0.0
                for a,b_ in rails:
                    p_ref = reflect_point_across_line(p, a, b_)
                    # Line ball -> p_ref intersects the rail near bank point; we don't enforce exact hit point here
                    v2 = b.center - p_ref
                    nv2 = v2 / (np.linalg.norm(v2) + 1e-9)
                    ghost2 = b.center + nv2 * (2.0 * R)
                    cue_clear2 = segment_clear(cue.center, ghost2, balls, ignore_indices=[bi, cue_idx], clearance=R*0.9)
                    # The actual ball path is ball->bank (implicit) -> pocket; approximate clearance along ball->pocket
                    obj_clear2 = segment_clear(b.center, p, balls, ignore_indices=[bi, cue_idx], clearance=R*0.5)
                    if cue_clear2 and obj_clear2:
                        d1 = dist(cue.center, ghost2)
                        d2 = dist(b.center, p)
                        score2 = 0.8*(1.0/(d1+1e-3) + 0.5/(d2+1e-3))  # weaker than direct
                        if score2 > best_bank_score:
                            best_bank_score = score2
                            best_bank = Shot(
                                target_label=balls[bi].label,
                                ball_center=(float(b.center[0]), float(b.center[1])),
                                pocket_center=(float(p[0]), float(p[1])),
                                ghost_point=(float(ghost2[0]), float(ghost2[1])),
                                cue_to_ghost_clear=True,
                                obj_to_pocket_clear=True,
                                path_score=score2
                            )
                if best_bank is not None:
                    candidates.append(best_bank)

    # Sort by score descending
    # Try simple 2-ball combos (cue->b1->b2->pocket) if direct is blocked or lower score
    for b1i, b1 in enumerate(balls):
        if b1i == cue_idx or b1.label in ('cue', 'unknown'):
            continue
        for b2i, b2 in enumerate(balls):
            if b2i == cue_idx or b2i == b1i or b2.label in ('cue', 'unknown'):
                continue
            for p in pockets:
                shot = compute_combo_shot(rect_bgr, balls, cue_idx, b1i, b2i, p, R)
                if shot is not None:
                    candidates.append(shot)

    candidates.sort(key=lambda s: (s.is_combo, -s.path_score))  # prefer non-combo when scores tie
    return candidates

# -----------------------------
# Drawing
# -----------------------------

def draw_overlay(rect_bgr: np.ndarray,
                 table_mask: np.ndarray,
                 pockets: List[np.ndarray],
                 balls: List[Ball],
                 best_shot: Optional[Shot] = None,
                 out_path: Optional[str] = None) -> np.ndarray:
    img = rect_bgr.copy()

    # Draw pockets
    for p in pockets:
        p = p.astype(int)
        cv2.rectangle(img, (p[0]-8, p[1]-8), (p[0]+8, p[1]+8), (0,255,255), 2)

    # Draw balls
    for b in balls:
        color = (0,255,0)  # default
        thickness = 2
        if b.label == 'cue':
            color = (255, 128, 0)  # blue-ish in BGR? (OpenCV uses BGR; use (255,128,0) ~ cyan/orange mix)
        elif b.label == 'eight':
            color = (0,0,255)  # red
        elif b.label == 'stripe':
            color = (0,255,255)  # yellow
        elif b.label == 'solid':
            color = (0,255,0)  # green
        center = tuple(np.round(b.center).astype(int))
        cv2.circle(img, center, int(b.radius), color, thickness=2)
        if b.label == 'stripe':
            # dashed ring: draw small arcs
            for a in range(0, 360, 30):
                a1 = math.radians(a)
                a2 = math.radians(a+10)
                p1 = (int(center[0] + b.radius*math.cos(a1)), int(center[1] + b.radius*math.sin(a1)))
                p2 = (int(center[0] + b.radius*math.cos(a2)), int(center[1] + b.radius*math.sin(a2)))
                cv2.line(img, p1, p2, color, 2)
        # label
        cv2.putText(img, b.label[:1], (center[0]-6, center[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

    # Best shot lines
    if best_shot is not None:
        cue_center = next((b.center for b in balls if b.label == 'cue'), None)
        if cue_center is not None:
            c = tuple(np.round(cue_center).astype(int))
            g = tuple((int(best_shot.ghost_point[0]), int(best_shot.ghost_point[1])))
            bcen = tuple((int(best_shot.ball_center[0]), int(best_shot.ball_center[1])))
            p = tuple((int(best_shot.pocket_center[0]), int(best_shot.pocket_center[1])))
            # cue -> ghost
            cv2.line(img, c, g, (255,255,255), 3)
            # ghost -> ball
            cv2.line(img, g, bcen, (255,0,0), 2)
            # ball -> pocket
            cv2.line(img, bcen, p, (0,0,255), 2)
            cv2.circle(img, g, 5, (255,255,255), -1)
            cv2.circle(img, p, 6, (0,0,255), -1)
            cv2.putText(img, "BEST SHOT", (min(c[0], p[0])+10, min(c[1], p[1])+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,255,50), 2, cv2.LINE_AA)

    if out_path:
        cv2.imwrite(out_path, img)
    return img


# -----------------------------
# XY Plot (matplotlib)
# -----------------------------
def plot_xy(rect_bgr: np.ndarray,
            pockets: List[np.ndarray],
            balls: List[Ball],
            best_shot: Optional[Shot],
            out_path: str) -> None:
    # \"\"\"Render a single XY chart of the table with balls, pockets, and best-shot vectors.
    # Notes (per plotting constraints):
    #   - Uses matplotlib (no seaborn)
    #   - Single plot (no subplots)
    #   - Does not set any explicit colors or styles
    # \"\"\"
    import matplotlib.pyplot as plt

    h, w = rect_bgr.shape[:2]

    # Create figure
    plt.figure(figsize=(8, 4))

    # Draw table boundary
    xs = [0, w, w, 0, 0]
    ys = [0, 0, h, h, 0]
    plt.plot(xs, ys)

    # Draw pockets
    for p in pockets:
        px, py = float(p[0]), float(p[1])
        plt.scatter([px], [py], marker='s')
        plt.text(px, py, "P", fontsize=8)

    # Draw balls
    for b in balls:
        x, y, r = float(b.center[0]), float(b.center[1]), float(b.radius)
        # circle
        circ = plt.Circle((x, y), r, fill=False)
        plt.gca().add_patch(circ)
        label_text = b.label
        plt.text(x + r*0.1, y + r*0.1, label_text, fontsize=8)

    # Draw best shot vectors
    if best_shot is not None:
        cx, cy = next((float(b.center[0]), float(b.center[1])) for b in balls if b.label == 'cue')
        gx, gy = best_shot.ghost_point
        bx, by = best_shot.ball_center
        px, py = best_shot.pocket_center
        # cue -> ghost
        plt.arrow(cx, cy, gx - cx, gy - cy, length_includes_head=True, head_width=8, head_length=12)
        # ghost -> ball
        plt.plot([gx, bx], [gy, by])
        # ball -> pocket
        plt.plot([bx, px], [by, py])
        plt.text((cx+px)/2.0, (cy+py)/2.0, "BEST SHOT", fontsize=10)

    # Axes formatting
    plt.title("Pool Table XY (pixels)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(0, w)
    plt.ylim(h, 0)  # invert Y so origin appears top-left like image coords
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -----------------------------
# Physics-ish heuristics & technique summary
# -----------------------------

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-9
    return v / n

def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    a_u = unit(a); b_u = unit(b)
    cosang = float(np.clip(np.dot(a_u, b_u), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))

def side_cut_direction(a_to_obj: np.ndarray, obj_to_pocket: np.ndarray) -> str:
    # \"\"\"Return 'left' or 'right' for the cut direction from the shooter's POV in image coordinates.
    # Uses sign of z-component of 2D cross (a × b). Positive -> left cut.\"\"\"
    ax, ay = a_to_obj[0], a_to_obj[1]
    bx, by = obj_to_pocket[0], obj_to_pocket[1]
    z = ax*by - ay*bx
    return 'left' if z > 0 else 'right'

def advise_force(d1: float, d2: float, table_w: float) -> dict:
    L = d1 + d2
    # Normalize by table width
    ratio = L / (table_w + 1e-9)
    if ratio < 0.6:
        cat = 'low'
        v = 1.5  # m/s approx
    elif ratio < 1.2:
        cat = 'medium'
        v = 2.5
    else:
        cat = 'high'
        v = 3.5
    return {'category': cat, 'linear_speed_mps': v, 'path_length_px': L, 'norm_to_table_width': ratio}

def advise_english(cut_deg: float, cut_dir: str) -> dict:
    # Outside english counteracts throw; amount increases with cut
    if cut_deg >= 40:
        tips = 0.75
    elif cut_deg >= 20:
        tips = 0.5
    elif cut_deg >= 8:
        tips = 0.25
    else:
        tips = 0.0
    return {'type': f'outside-{cut_dir}', 'tips': tips}

def advise_draw_follow(cut_deg: float, d1: float, obj_radius: float) -> dict:
    # Heuristic: short cue->object and small cut -> gentle follow; big cut -> stun or slight draw to hold line
    if d1 <= 8*obj_radius and cut_deg <= 12:
        return {'type': 'follow', 'tips': 0.25}
    if cut_deg >= 35:
        return {'type': 'stun/delayed-draw', 'tips': 0.25}
    # neutral default
    return {'type': 'stun', 'tips': 0.0}

def estimate_spin_rpm(force: dict, draw_follow: dict, english: dict) -> dict:
    # Very rough spin model: omega ~ k * offset_tips * v / R
    R = 0.028575  # m, cue ball radius (2.25in diameter)
    v = float(force['linear_speed_mps'])
    k = 18.0  # lumped constant absorbing tip compliance, contact time (~.001s), etc.
    # vertical (top/bottom) spin from draw/follow
    ty = float(draw_follow.get('tips', 0.0))
    omega_top_bottom = k * ty * v / (R + 1e-9)   # rad/s
    # side spin from english
    tx = float(english.get('tips', 0.0))
    omega_side = k * tx * v / (R + 1e-9)         # rad/s

    rpm_tb = omega_top_bottom * 60.0 / (2.0*np.pi)
    rpm_side = omega_side * 60.0 / (2.0*np.pi)
    # object ball picks up slight topspin from friction; scale small with cut
    obj_rpm = max(0.0, 0.15 * rpm_tb)

    return {
        'cue_ball_rpm_top_bottom': float(rpm_tb),
        'cue_ball_rpm_side': float(rpm_side),
        'object_ball_rpm_estimate': float(obj_rpm)
    }

def build_physics_summary(rect_bgr: np.ndarray,
                          balls: List[Ball],
                          best: Optional[Shot]) -> Optional[dict]:
    if best is None:
        return None
    h, w = rect_bgr.shape[:2]
    cue = next((b for b in balls if b.label == 'cue'), None)
    if cue is None:
        return None

    c = cue.center
    b = np.array(best.ball_center, dtype=np.float32)
    p = np.array(best.pocket_center, dtype=np.float32)
    g = np.array(best.ghost_point, dtype=np.float32)

    a_vec = b - c       # cue -> object
    o_vec = p - b       # object -> pocket

    cut_deg = angle_between_deg(a_vec, o_vec)
    cut_dir = side_cut_direction(a_vec, o_vec)

    d1 = dist(c, g)
    d2 = dist(b, p)
    force = advise_force(d1, d2, float(w))

    english = advise_english(cut_deg, cut_dir)
    df = advise_draw_follow(cut_deg, d1, obj_radius=float(np.median([x.radius for x in balls]) if balls else 10.0))

    spin = estimate_spin_rpm(force, df, english)

    return {
        'cut_angle_degrees': float(cut_deg),
        'cut_direction': cut_dir,
        'force': force,
        'draw_follow': df,
        'english': english,
        'spin_estimates': spin,
        'assumptions': {
            'model': 'heuristic; single-collision; one-rail bank considered; no detailed cloth/ball friction modeling',
            'cue_ball_radius_m': 0.028575
        }
    }

# -----------------------------
# Main pipeline
# -----------------------------

def process_image(in_path: str, outdir: str):
    
# Make timestamped output directory inside outdir
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(outdir, ts)
    os.makedirs(outdir, exist_ok=True)

    src = cv2.imread(in_path)
    if src is None:
        raise FileNotFoundError(f"Could not read image: {in_path}")

    # 1) Find table corners in original view
    corners = find_table_corners(src)
    if corners is None:
        raise RuntimeError("Failed to find table corners. Adjust lighting/contrast or tune thresholds.")

    # 2) Warp to top-down
    rect, H = warp_to_topdown(src, corners, target_short=900)
    cv2.imwrite(os.path.join(outdir, "rectified.png"), rect)

    # 3) Table mask (rectified), rails ~ borders
    table_mask = detect_table_mask(rect)
    cv2.imwrite(os.path.join(outdir, "table_mask.png"), table_mask)

    # 4) Pockets
    pockets = detect_pockets(rect, table_mask)

    # 5) Balls
    balls = detect_balls(rect, table_mask)
    # 6) Classification
    classify_balls(rect, balls)

    # 7) Shots
    shots = compute_shots(rect, balls, pockets)
    best = shots[0] if shots else None

    # 8) Draw overlays

    # Physics/technique summary
    physics_summary = build_physics_summary(rect, balls, best)

    out_img = draw_overlay(rect, table_mask, pockets, balls, best_shot=best,
                           out_path=os.path.join(outdir, "overlay.png"))

    # 9) XY plot
    plot_xy(rect, pockets, balls, best, os.path.join(outdir, "xy_best_shot.png"))

    # 10) Save JSON summary
    summary = {
        "image": in_path,
        "rectified": os.path.join(outdir, "rectified.png"),
        "overlay": os.path.join(outdir, "overlay.png"),
        "xy_plot": os.path.join(outdir, "xy_best_shot.png"),
        "num_balls": len(balls),
        "balls": [
            {
                "label": b.label,
                "center_xy": [float(b.center[0]), float(b.center[1])],
                "radius": float(b.radius),
                "mean_bgr": [float(x) for x in b.mean_bgr],
            } for b in balls
        ],
        "pockets": [[float(p[0]), float(p[1])] for p in pockets],
        "physics_summary": physics_summary,
        "best_shot": ({
            "is_combo": best.is_combo,
            "combo_mid_ball_center": list(best.combo_mid_ball_center) if (hasattr(best, 'combo_mid_ball_center') and best.combo_mid_ball_center is not None) else None,
            "target_label": best.target_label,
            "ball_center": list(best.ball_center),
            "pocket_center": list(best.pocket_center),
            "ghost_point": list(best.ghost_point),
            "path_score": best.path_score,
        } if best is not None else None
    )}
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Write a concise text summary
    if physics_summary is not None:
        lines = []
        lines.append(f"CUT ANGLE: {physics_summary['cut_angle_degrees']:.1f} deg ({physics_summary['cut_direction']} cut)")
        lines.append(f"FORCE: {physics_summary['force']['category']} (v≈{physics_summary['force']['linear_speed_mps']:.1f} m/s)")
        lines.append(f"DRAW/FOLLOW: {physics_summary['draw_follow']['type']} ({physics_summary['draw_follow']['tips']} tips)")
        lines.append(f"ENGLISH: {physics_summary['english']['type']} ({physics_summary['english']['tips']} tips)")
        lines.append(f"CUE SPIN: TB≈{physics_summary['spin_estimates']['cue_ball_rpm_top_bottom']:.0f} rpm, SIDE≈{physics_summary['spin_estimates']['cue_ball_rpm_side']:.0f} rpm")
        lines.append(f"OBJECT SPIN: ≈{physics_summary['spin_estimates']['object_ball_rpm_estimate']:.0f} rpm")
        with open(os.path.join(outdir, "physics_summary.txt"), "w") as tf:
            tf.write("\n".join(lines))

    return summary

def main():
    ap = argparse.ArgumentParser(description="Pool cue guide pipeline")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--outdir", default="out", help="Output directory")
    args = ap.parse_args()

    summary = process_image(args.image, args.outdir)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
