"""Review loop + shot picking on top of the vision pipeline (vision.py).

Flow: analyze(image) -> user reviews labelled balls -> refine(improve/remove) -> ... -> approve -> top 3 shots.
All coordinates are in the rectified (top-down) image, so the transparent overlay lines up with rectified.png.
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from . import vision as v

MAX_SIDE = 1600  # phone photos are downscaled before analysis


@dataclass
class Session:
    id: str
    rect: Optional[np.ndarray] = None
    table_mask: Optional[np.ndarray] = None
    pockets: list = field(default_factory=list)
    balls: list = field(default_factory=list)
    felt_hsv: Optional[np.ndarray] = None
    suppressed: list = field(default_factory=list)  # (x, y, r) zones the user removed
    round: int = 0
    next_id: int = 1
    created: float = field(default_factory=time.time)
    history: list = field(default_factory=list)
    mode: str = "auto"          # 'markers' | 'auto' | 'manual' | 'none' (photo only, waiting for taps)
    pocket_r: float = 0.0       # pocket radius in rectified pixels (markers/manual)
    src_jpg: bytes = b""        # the (downscaled) original, kept as JPEG so sessions stay small
    src_size: tuple = (0, 0)    # (w, h) of that original: coordinates the tap editor uses
    H: Optional[np.ndarray] = None   # original -> rectified homography
    message: str = ""


class AnalysisError(Exception):
    pass


# ---------------------------------------------------------------- table finding

def _felt_dist(hsv: np.ndarray, felt) -> np.ndarray:
    """Distance from the felt colour. Hue and saturation dominate; brightness barely counts,
    so a lighting vignette (bright middle, dark edges) doesn't look like a different surface."""
    dh = np.minimum(np.abs(hsv[:, :, 0] - felt[0]), 180 - np.abs(hsv[:, :, 0] - felt[0]))
    return 2.0 * dh + np.abs(hsv[:, :, 1] - felt[1]) / 2 + 0.15 * np.abs(hsv[:, :, 2] - felt[2])


def _coverage_crop(m: np.ndarray, keep: float = 0.8) -> np.ndarray:
    """Zero out columns/rows that are mostly not felt. Rails with blue patterns survive a colour test but
    not this: the felt spans the full height/width, a strip of tiles doesn't."""
    h, w = m.shape
    col = (m > 0).mean(axis=0)
    row = (m > 0).mean(axis=1)
    cbase = float(np.median(col[w // 4: 3 * w // 4]))
    rbase = float(np.median(row[h // 4: 3 * h // 4]))
    cx = np.where(col >= keep * cbase)[0]
    ry = np.where(row >= keep * rbase)[0]
    if len(cx) == 0 or len(ry) == 0:
        return m
    out = np.zeros_like(m)
    out[ry[0]:ry[-1] + 1, cx[0]:cx[-1] + 1] = m[ry[0]:ry[-1] + 1, cx[0]:cx[-1] + 1]
    return out


def _felt_mask(img: np.ndarray, ref=None):
    """Mask of the playing surface by hue (not brightness), with holes (balls, markings) filled in.
    Returns (mask, ref) where ref is the felt's HSV reference taken from the middle of the photo."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    if ref is None:
        mid = hsv[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)].reshape(-1, 3)
        ref = np.median(mid, axis=0)
    dh = np.minimum(np.abs(hsv[:, :, 0] - ref[0]), 180 - np.abs(hsv[:, :, 0] - ref[0]))
    ok = (dh <= 14) & (hsv[:, :, 1] >= max(40.0, 0.45 * ref[1])) & (hsv[:, :, 2] >= 25)
    m = ok.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    big = max(15, int(0.09 * min(h, w))) | 1   # sheds rail patterns and other strips narrower than ~9% of the photo
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (big, big)))
    m = _coverage_crop(m)
    cnt = v.largest_contour(m)
    out = np.zeros_like(m)
    if cnt is not None:
        cv2.drawContours(out, [cv2.convexHull(cnt)], -1, 255, thickness=cv2.FILLED)
    return out, ref


def _four_corners(mask: np.ndarray):
    """Corners as the hull points furthest along each diagonal (robust to chamfered pocket corners)."""
    cnt = v.largest_contour(mask)
    if cnt is None:
        return None
    hull = cv2.convexHull(cnt)
    pts = hull.reshape(-1, 2).astype(np.float32)
    rect = cv2.minAreaRect(hull)
    if cv2.contourArea(hull) >= 0.94 * rect[1][0] * rect[1][1] > 0:
        # Basically a rectangle (top-down photo or screenshot): rounded corners would skew a diagonal pick.
        quad = v.order_corners(cv2.boxPoints(rect).astype(np.float32))
    else:
        quad = v.order_corners(pts)  # angled photo: TL, TR, BR, BL via sum/diff extremes over ALL hull points
    top, right = np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[1])
    if top < right:              # portrait: make the long side the "top" so the warp isn't stretched
        quad = np.roll(quad, -1, axis=0)
    return quad


PAD = 0.07  # grow the outline so the cushions and pockets are inside the rectified image


def _find_table(src: np.ndarray):
    mask, ref = _felt_mask(src)
    if (mask > 0).mean() < 0.12:
        return None, None, None
    quad = _four_corners(mask)
    if quad is None:
        return None, None, None
    c = quad.mean(axis=0)
    return (c + (quad - c) * (1 + PAD)).astype(np.float32), ref, quad


def _pockets(rect: np.ndarray, mask: np.ndarray) -> list:
    """Six pockets: the felt outline's corners and long-side midpoints, snapped to the nearest dark blob."""
    h, w = mask.shape
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []
    x0, x1 = float(np.percentile(xs, 0.5)), float(np.percentile(xs, 99.5))
    y0, y1 = float(np.percentile(ys, 0.5)), float(np.percentile(ys, 99.5))
    push = 0.012 * w
    anchors = [(x0 - push, y0 - push), (x1 + push, y0 - push), (x1 + push, y1 + push), (x0 - push, y1 + push),
               ((x0 + x1) / 2, y0 - push), ((x0 + x1) / 2, y1 + push)]
    dark = (cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)[:, :, 2] < 70).astype(np.uint8)
    out = []
    for ax, ay in anchors:
        wx, wy = int(0.07 * w), int(0.12 * h)
        xa, xb = int(max(0, ax - wx)), int(min(w, ax + wx))
        ya, yb = int(max(0, ay - wy)), int(min(h, ay + wy))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(dark[ya:yb, xa:xb])
        best = None
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 0.0004 * w * h:
                continue
            cx, cy = cent[i][0] + xa, cent[i][1] + ya
            d = math.hypot(cx - ax, cy - ay)
            if best is None or d < best[0]:
                best = (d, cx, cy)
        out.append(np.array([best[1], best[2]] if best else [ax, ay], dtype=np.float32))
    return out


# ---------------------------------------------------------------- green pocket markers

MARKER_HUE = (40, 72)   # OpenCV hue of the neon green (about 53); felt is usually blue or a darker/duller green


def _find_markers(src: np.ndarray) -> list:
    """Bright-green discs painted/placed over the pockets. Returns [(x, y, r)], biggest first."""
    h, w = src.shape[:2]
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 0] >= MARKER_HUE[0]) & (hsv[:, :, 0] <= MARKER_HUE[1])
         & (hsv[:, :, 1] >= 130) & (hsv[:, :, 2] >= 190)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if not (0.0012 * h * w <= area <= 0.04 * h * w):
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        bx, by, bw, bh = cv2.boundingRect(c)
        edge = bx <= 1 or by <= 1 or bx + bw >= w - 1 or by + bh >= h - 1
        if not edge and area < 0.6 * math.pi * r * r:
            continue  # not disc-shaped
        out.append((float(x), float(y), float(r)))
    out.sort(key=lambda t: -t[2])
    return out[:6] if len(out) in (4, 5, 6) or len(out) > 6 else out


def _layout_from_points(pts: np.ndarray, rm: Optional[float] = None, allow_partial: bool = False):
    """Pocket points (any numbering) -> (corner quad TL,TR,BR,BL, all six pockets, pocket radius).
    Corners are the extremes along the diagonals; missing middle pockets are the midpoints of the long sides."""
    n = len(pts)
    if n < 4 or n > 6 or (not allow_partial and n not in (4, 6)):
        return None
    xs, ys = pts[:, 0], pts[:, 1]
    idx = [int(np.argmin(xs + ys)), int(np.argmax(xs - ys)), int(np.argmax(xs + ys)), int(np.argmin(xs - ys))]
    if len(set(idx)) != 4:
        return None
    quad = pts[idx].astype(np.float32)                 # TL, TR, BR, BL
    top, right = np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[1])
    if top < right:                                    # portrait: make a long side the "top"
        quad = np.roll(quad, -1, axis=0)
        top, right = right, top
    mids = [pts[i] for i in range(n) if i not in idx]
    for a_, b_ in ((quad[0], quad[1]), (quad[3], quad[2])):   # the two long sides
        mid = (a_ + b_) / 2
        if not any(np.linalg.norm(m - mid) < 0.25 * top for m in mids):
            mids.append(mid)
    if len(mids) != 2:
        return None
    if rm is None:
        rm = 0.03 * float(max(top, right))
    return quad, [*quad, *mids], float(rm)


def _marker_layout(mk: list):
    """6 markers = all pockets; 4 = corner pockets only. Anything else is ambiguous (automatic detection instead)."""
    if len(mk) not in (4, 6):
        return None
    pts = np.array([[x, y] for x, y, _ in mk], np.float32)
    return _layout_from_points(pts, rm=float(np.mean([r for _, _, r in mk])))


def _grow(quad: np.ndarray, d: float) -> np.ndarray:
    """Move every corner outward (d>0) or inward (d<0) along both of its edges."""
    out = []
    for i in range(4):
        c, nx, pv = quad[i], quad[(i + 1) % 4], quad[i - 1]
        out.append(c - _unit(nx - c) * d - _unit(pv - c) * d)
    return np.array(out, np.float32)


def _hough_balls(rect: np.ndarray, mask: np.ndarray) -> list:
    """Circle search sized for balls on a table that fills the picture (about 1-2.5% of its width)."""
    gray = cv2.medianBlur(cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY), 5)
    h, w = gray.shape
    lo, hi = max(6, int(0.009 * w)), int(0.025 * w)
    cs = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=lo * 2, param1=120, param2=20,
                          minRadius=lo, maxRadius=hi)
    out = []
    if cs is not None:
        for x, y, r in np.round(cs[0]).astype(int):
            if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h or mask[y, x] == 0:
                continue
            out.append(v.Ball(center=np.array([x, y], dtype=np.float32), radius=float(r),
                              mean_bgr=_patch_mean(rect, int(x), int(y), int(r))))
    return out


# ---------------------------------------------------------------- analyze

def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise AnalysisError("That file isn't an image we can read (try JPG or PNG).")
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def _felt_hsv(rect: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
    px = hsv[mask > 0]
    return np.median(px, axis=0) if len(px) else np.array([60, 120, 120], dtype=np.float32)


def _assign_ids(sess: Session, balls: list) -> None:
    for b in sorted(balls, key=lambda b: (round(float(b.center[0]) / 40), float(b.center[1]))):
        if not b.id:
            b.id = f"B{sess.next_id}"
            sess.next_id += 1


def _blob_balls(rect, mask, felt, pockets, known, R, thr: float = 38, max_add: int = 99) -> list:
    """Colour-blobs that differ from the felt (hue/saturation, not brightness): finds balls Hough misses."""
    h, w = rect.shape[:2]
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV).astype(np.float32)
    d = _felt_dist(hsv, felt)
    blob = ((d > thr) & (mask > 0)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, k, iterations=2)
    cands = []
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if not (0.6 * math.pi * R * R <= area <= 1.4 * math.pi * R * R):
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if area < 0.78 * math.pi * r * r:
            continue  # not round: a line, a smudge, or a merged pair
        bw, bh = cv2.boundingRect(c)[2:]
        if max(bw, bh) > 1.35 * min(bw, bh):
            continue  # thin or stretched
        if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
            continue
        if any(math.hypot(x - p[0], y - p[1]) < 1.6 * R for p in pockets):
            continue
        if any(math.hypot(x - b.center[0], y - b.center[1]) < 1.2 * R for b in known):
            continue
        cands.append((area / (math.pi * r * r), x, y, r))
    cands.sort(reverse=True)  # roundest first
    return [v.Ball(center=np.array([x, y], dtype=np.float32), radius=float(r),
                   mean_bgr=_patch_mean(rect, int(x), int(y), int(r))) for _, x, y, r in cands[:max_add]]


def _on_felt(rect, felt, balls: list, need: float = 0.45) -> list:
    """Keep balls with felt around them. A real ball sits on cloth (or a cushion, which is cloth-coloured);
    rail ornaments and tiles don't."""
    if not balls:
        return balls
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV).astype(np.float32)
    felt_px = (_felt_dist(hsv, felt) < 34).astype(np.uint8)
    h, w = felt_px.shape
    keep = []
    for b in balls:
        x, y, r = int(b.center[0]), int(b.center[1]), float(b.radius)
        ring = np.zeros((h, w), np.uint8)
        cv2.circle(ring, (x, y), int(r * 1.7), 1, -1)
        cv2.circle(ring, (x, y), int(r * 1.15), 0, -1)
        n = int(ring.sum())
        if n and float((felt_px & ring).sum()) / n >= need:
            keep.append(b)
    return keep


def _inside_play(mask: np.ndarray, R: float, balls: list) -> list:
    """A ball centre is at least one radius inside the table outline."""
    k = max(3, int(R)) * 2 + 1
    play = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    h, w = play.shape
    return [b for b in balls if 0 <= int(b.center[1]) < h and 0 <= int(b.center[0]) < w
            and play[int(b.center[1]), int(b.center[0])] > 0]


def _build_markers(src: np.ndarray, layout):
    quad, six, rm = layout
    padded = _grow(quad, 1.15 * rm)    # pockets sit on the corners; grow so they are fully in the picture
    inner = _grow(quad, -0.5 * rm)     # playing area: just inside the cushions
    rect, H = v.warp_to_topdown(src, padded, target_short=900)
    tf = lambda pts: cv2.perspectiveTransform(np.array(pts, np.float32)[None], H)[0]
    mask = np.zeros(rect.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, tf(inner).astype(np.int32), 255)
    pockets = [p.astype(np.float32) for p in tf(six)]
    width_src = 0.5 * (np.linalg.norm(padded[1] - padded[0]) + np.linalg.norm(padded[2] - padded[3]))
    return rect, mask, pockets, rm * rect.shape[1] / max(1.0, float(width_src)), H


def _build_auto(src: np.ndarray):
    corners, ref, inner_quad = _find_table(src)
    if corners is None:
        raise AnalysisError(
            "Couldn't find the table. Take the photo from above with the whole felt in view and even lighting, "
            "or mark the six pockets bright green."
        )
    rect, H = v.warp_to_topdown(src, corners, target_short=900)
    mask, _ = _felt_mask(rect, ref)
    # Balls live on the felt and cushions, never on the rails: keep only the un-padded table outline.
    inner = np.zeros_like(mask)
    cv2.fillConvexPoly(inner, cv2.perspectiveTransform(inner_quad[None].astype(np.float32), H)[0].astype(np.int32), 255)
    mask = cv2.bitwise_and(mask, inner)
    return rect, mask, _pockets(rect, mask), 0.0, H


def _jpg(src: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", src, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def src_of(sess: Session) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(sess.src_jpg, np.uint8), cv2.IMREAD_COLOR)


def photo_session(src: np.ndarray, message: str) -> Session:
    """Automatic detection failed: keep the photo so the user can tap the pockets and balls instead."""
    h, w = src.shape[:2]
    return Session(id=uuid.uuid4().hex[:12], mode="none", message=message, src_jpg=_jpg(src), src_size=(w, h))


def analyze(data: bytes) -> Session:
    return analyze_src(decode_image(data))


def analyze_src(src: np.ndarray) -> Session:
    layout = _marker_layout(_find_markers(src))
    mode = "markers" if layout else "auto"
    rect, mask, pockets, pocket_r, H = _build_markers(src, layout) if layout else _build_auto(src)

    balls = _hough_balls(rect, mask)
    if len(balls) >= 4:
        med = float(np.median([b.radius for b in balls]))
        balls = [b for b in balls if 0.75 * med <= b.radius <= 1.3 * med]
    balls = [b for b in balls if not any(math.hypot(b.center[0] - p[0], b.center[1] - p[1]) < 1.6 * b.radius for p in pockets)]
    felt = _felt_hsv(rect, mask)
    if felt[1] < 35:
        raise AnalysisError("That doesn't look like a pool table (no felt colour found). Try another photo.")
    R0 = float(np.median([b.radius for b in balls])) if len(balls) >= 3 else 0.013 * rect.shape[1]
    balls += _blob_balls(rect, mask, felt, pockets, balls, R0)
    balls = _on_felt(rect, felt, balls)
    balls = _inside_play(mask, R0, balls)
    v.classify_balls(rect, balls)
    if len(balls) < 2:
        raise AnalysisError("Couldn't find the balls. Use a sharper, brighter photo from above.")
    sess = Session(id=uuid.uuid4().hex[:12], rect=rect, table_mask=mask, pockets=pockets, balls=balls,
                   felt_hsv=felt, mode=mode, pocket_r=pocket_r, H=H,
                   src_jpg=_jpg(src), src_size=(src.shape[1], src.shape[0]))
    _assign_ids(sess, sess.balls)
    sess.history.append({"round": 0, "action": f"initial detection ({mode})", "balls": len(balls)})
    return sess


# ---------------------------------------------------------------- manual (tap to place)

def parse_kind(kind: str):
    """'pocket3' -> ('pocket', 3); 'cue'; 'ball11' -> ('ball', 11); 'solid?' / 'stripe?' (number unknown)."""
    k = (kind or "").strip().lower()
    if k in ("cue", "solid?", "stripe?"):
        return k, None
    for pre, lo, hi in (("pocket", 1, 6), ("ball", 1, 15)):
        if k.startswith(pre) and k[len(pre):].isdigit() and lo <= int(k[len(pre):]) <= hi:
            return pre, int(k[len(pre):])
    return None


def _ball_from_kind(kind: str, n):
    if kind == "cue":
        return "cue", "CUE"
    if kind == "ball":
        return ("eight" if n == 8 else "solid" if n < 8 else "stripe"), f"#{n}"
    return ("solid" if kind == "solid?" else "stripe"), None


def build_manual(sess: Session, items: list[dict]) -> dict:
    """Make the table and balls from the user's taps (x, y in the original photo, plus what each point is)."""
    parsed, seen = [], set()
    for it in items:
        pk = parse_kind(it.get("kind", ""))
        if pk is None:
            raise AnalysisError(f"A point has no label yet. Pick what it is from its dropdown.")
        if pk[1] is not None and pk in seen:
            raise AnalysisError(f"{it['kind'].replace('ball', 'ball ').replace('pocket', 'pocket ')} is used twice.")
        seen.add(pk)
        parsed.append((it, pk))
    pockets_in = [(it, pk) for it, pk in parsed if pk[0] == "pocket"]
    if len(pockets_in) < 4:
        raise AnalysisError("Tap at least the four corner pockets (any numbers) so the table can be straightened.")
    layout = _layout_from_points(np.array([[it["x"], it["y"]] for it, _ in pockets_in], np.float32), allow_partial=True)
    if layout is None:
        raise AnalysisError("Those pockets don't make a table shape. Check that the four corners are spread out.")
    src = src_of(sess)
    h, w = src.shape[:2]
    if cv2.contourArea(layout[0]) < 0.04 * h * w:
        raise AnalysisError("The pockets are too close together. Tap the pocket centres, not the middle of the table.")
    rect, mask, pockets, pocket_r, H = _build_markers(src, layout)

    bi = [(it, pk) for it, pk in parsed if pk[0] != "pocket"]
    if bi:
        pr = cv2.perspectiveTransform(np.array([[it["x"], it["y"]] for it, _ in bi], np.float32)[None], H)[0]
    Rg = 0.0125 * rect.shape[1]
    balls = []
    for (it, pk), (x, y) in zip(bi, pr if bi else []):
        label, bid = _ball_from_kind(*pk)
        b = v.Ball(center=np.array([x, y], dtype=np.float32), radius=Rg, mean_bgr=(0.0, 0.0, 0.0), label=label)
        b.id = bid or ""
        balls.append(b)
    sess.rect, sess.table_mask, sess.pockets, sess.pocket_r, sess.H = rect, mask, pockets, pocket_r, H
    sess.balls, sess.mode, sess.message = balls, "manual", ""
    sess.felt_hsv, sess.suppressed, sess.round = _felt_hsv(rect, mask), [], 0

    # Snap each tap to the real ball centre (a tap is a few pixels off); try a few ball sizes.
    snapped = {}
    for scale in (1.0, 0.8, 1.25, 1.55):
        for b in balls:
            b.radius = Rg * scale
        for b in balls:
            if id(b) in snapped:
                continue
            before = b.center.copy()
            if _local_redetect(sess, b):
                snapped[id(b)] = (b.center.copy(), b.radius)
            else:
                b.center = before
    med = float(np.median([r for _, r in snapped.values()])) if len(snapped) >= 2 else Rg
    for b in balls:
        if id(b) in snapped:
            b.center, b.radius = snapped[id(b)][0], snapped[id(b)][1]
        else:
            b.radius = med
            b.mean_bgr = _patch_mean(rect, int(b.center[0]), int(b.center[1]), int(med))
    _assign_ids(sess, balls)
    sess.history.append({"round": 0, "action": "built from taps", "balls": len(balls)})
    return {"pockets": len(pockets), "balls": len(balls), "snapped": len(snapped)}


def prefill_items(sess: Session) -> list[dict]:
    """The current detection as tap points in the original photo, so the user can start from it and fix it."""
    if sess.H is None or sess.rect is None:
        return []
    Hi = np.linalg.inv(sess.H)
    tf = lambda pts: cv2.perspectiveTransform(np.array(pts, np.float32)[None], Hi)[0]
    out = []
    if sess.pockets:
        c0 = np.mean(sess.pockets, axis=0)
        ring = sorted(sess.pockets, key=lambda p: math.atan2(p[1] - c0[1], p[0] - c0[0]))
        for i, (x, y) in enumerate(tf(ring), 1):
            out.append({"x": round(float(x), 1), "y": round(float(y), 1), "kind": f"pocket{i}"})
    if sess.balls:
        for b, (x, y) in zip(sess.balls, tf([b.center for b in sess.balls])):
            kind = {"cue": "cue", "eight": "ball8", "solid": "solid?", "stripe": "stripe?"}.get(b.label, "solid?")
            out.append({"x": round(float(x), 1), "y": round(float(y), 1), "kind": kind})
    return out


def _relabel(sess: Session) -> None:
    """After a refine round. Automatic modes re-run the cue/8/stripe/solid guess; manual mode keeps the
    user's labels and only guesses solid/stripe for balls found by 'look for missing'."""
    if sess.mode != "manual":
        for b in sess.balls:
            b.label = "unknown"
        v.classify_balls(sess.rect, sess.balls)
        return
    new = [b for b in sess.balls if b.label == "unknown"]
    if not new:
        return
    cue = next((b for b in sess.balls if b.label == "cue"), None)
    eight = next((b for b in sess.balls if b.label == "eight"), None)
    if cue and eight:
        v.classify_balls(sess.rect, [cue, eight, *new])
        cue.label, eight.label = "cue", "eight"
    for b in new:
        if b.label in ("unknown", "cue", "eight"):
            b.label = "solid"


# ---------------------------------------------------------------- refine

def _ball_by_id(sess: Session, bid: str):
    return next((b for b in sess.balls if b.id.upper() == bid.upper()), None)


def _patch_mean(rect, x, y, r):
    h, w = rect.shape[:2]
    patch = rect[max(0, y - r):min(h, y + r), max(0, x - r):min(w, x + r)]
    return tuple(np.mean(patch.reshape(-1, 3), axis=0).tolist()) if patch.size else (0.0, 0.0, 0.0)


def _local_redetect(sess: Session, b) -> bool:
    """Re-fit one ball: Hough on a small window with progressively looser thresholds, then a colour-blob fallback."""
    rect = sess.rect
    h, w = rect.shape[:2]
    R = float(np.median([x.radius for x in sess.balls])) if sess.balls else b.radius
    cx, cy = int(b.center[0]), int(b.center[1])
    pad = int(3.0 * R)
    x0, x1, y0, y1 = max(0, cx - pad), min(w, cx + pad), max(0, cy - pad), min(h, cy + pad)
    roi = cv2.medianBlur(cv2.cvtColor(rect[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), 5)
    others = [o for o in sess.balls if o is not b]

    def free(x, y, r):
        return all(math.hypot(x - o.center[0], y - o.center[1]) > 1.2 * (r + o.radius) * 0.9 for o in others)

    for p2 in (24, 18, 14, 10):
        cs = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(R), param1=100, param2=p2,
                              minRadius=max(4, int(R * 0.8)), maxRadius=int(R * 1.25))
        if cs is None:
            continue
        cands = [(float(c[0]) + x0, float(c[1]) + y0, float(c[2])) for c in cs[0]]
        cands = [c for c in cands if free(*c) and sess.table_mask[min(h - 1, int(c[1])), min(w - 1, int(c[0]))] > 0]
        if cands:
            x, y, r = min(cands, key=lambda c: math.hypot(c[0] - cx, c[1] - cy))
            b.center = np.array([x, y], dtype=np.float32)
            b.radius = r
            b.mean_bgr = _patch_mean(rect, int(x), int(y), int(r))
            return True

    # Fallback: the non-felt blob nearest to the old centre.
    hsv = cv2.cvtColor(rect[y0:y1, x0:x1], cv2.COLOR_BGR2HSV).astype(np.float32)
    d = _felt_dist(hsv, sess.felt_hsv)
    blob = (d > 40).astype(np.uint8) * 255
    blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        area = cv2.contourArea(c)
        if not (0.4 * math.pi * R * R <= area <= 1.8 * math.pi * R * R):
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        x, y = x + x0, y + y0
        if not free(x, y, r):
            continue
        dd = math.hypot(x - cx, y - cy)
        if best is None or dd < best[0]:
            best = (dd, x, y, r)
    if best:
        _, x, y, r = best
        b.center = np.array([x, y], dtype=np.float32)
        b.radius = float(r)
        b.mean_bgr = _patch_mean(rect, int(x), int(y), int(r))
        return True
    return False


def _rescan(sess: Session) -> int:
    """Looser colour-blob pass that only adds round, ball-sized objects not already known or removed.
    Capped per pass so a pattern on the cloth can't flood the list."""
    R = float(np.median([b.radius for b in sess.balls])) if sess.balls else 0.025 * sess.rect.shape[1]
    thr = max(26, 34 - 3 * sess.round)
    found = _blob_balls(sess.rect, sess.table_mask, sess.felt_hsv, sess.pockets, sess.balls, R, thr=thr, max_add=3)
    found = _inside_play(sess.table_mask, R, _on_felt(sess.rect, sess.felt_hsv, found))
    found = [b for b in found
             if not any(math.hypot(b.center[0] - sx, b.center[1] - sy) < sr for sx, sy, sr in sess.suppressed)]
    sess.balls.extend(found)
    return len(found)


def refine(sess: Session, improve: list[str], remove: list[str], find_missing: bool = False) -> dict:
    """Apply the user's review. Returns what happened per entry so the UI can say it."""
    report = {"removed": [], "improved": [], "unchanged": [], "unknown": [], "added": 0}
    for bid in remove:
        b = _ball_by_id(sess, bid)
        if not b:
            report["unknown"].append(bid)
            continue
        sess.suppressed.append((float(b.center[0]), float(b.center[1]), float(b.radius) * 1.3))
        sess.balls.remove(b)
        report["removed"].append(b.id)
    for bid in improve:
        if bid in remove:
            continue
        b = _ball_by_id(sess, bid)
        if not b:
            report["unknown"].append(bid)
            continue
        (report["improved"] if _local_redetect(sess, b) else report["unchanged"]).append(b.id)
    if find_missing:
        report["added"] = _rescan(sess)
    # Re-label cue/eight/stripe/solid with the updated set; ids stay put, new balls get new ids.
    _relabel(sess)
    _assign_ids(sess, sess.balls)
    sess.round += 1
    sess.history.append({"round": sess.round, **{k: report[k] for k in ("removed", "improved", "unchanged", "added")}})
    return report


# ---------------------------------------------------------------- overlay

_COL = {  # BGRA
    "cue": (255, 200, 60, 255), "eight": (60, 60, 255, 255), "stripe": (40, 220, 255, 255),
    "solid": (80, 230, 80, 255), "unknown": (200, 200, 200, 255),
}


def overlay_png(sess: Session, option: Optional[dict] = None, pick: Optional[list] = None) -> bytes:
    """Transparent RGBA drawing of pockets, labelled balls and (optionally) one shot, same size as the rectified image."""
    h, w = sess.rect.shape[:2]
    img = np.zeros((h, w, 4), np.uint8)
    GREEN = (90, 255, 100, 255)
    if len(sess.pockets) >= 3:
        c0 = np.mean(sess.pockets, axis=0)
        ring = sorted(sess.pockets, key=lambda p: math.atan2(p[1] - c0[1], p[0] - c0[0]))
        cv2.polylines(img, [np.array([_pt(p) for p in ring], np.int32)], True, GREEN, 2, cv2.LINE_AA)  # table outline
    pr = int(sess.pocket_r) if sess.pocket_r else 20
    for p in sess.pockets:
        cv2.circle(img, _pt(p), pr, GREEN, 3, cv2.LINE_AA)
    for b in sess.balls:
        c = (int(round(b.center[0])), int(round(b.center[1])))
        col = _COL.get(b.label, _COL["unknown"])
        r = int(round(b.radius))
        mark = pick and b.id in pick
        cv2.circle(img, c, r + (3 if mark else 0), (255, 0, 255, 255) if mark else col, 3 if mark else 2, cv2.LINE_AA)
        if b.label == "stripe":
            cv2.circle(img, c, max(2, r - 5), col, 1, cv2.LINE_AA)
        txt = b.id
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        tx, ty = c[0] - tw // 2, c[1] - r - 6
        cv2.rectangle(img, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 4), (0, 0, 0, 200), -1)
        cv2.putText(img, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, b.label[:1].upper(), (c[0] - 5, c[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255, 255), 1, cv2.LINE_AA)
    if option:
        _draw_option(img, sess, option)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def _pt(p):
    return (int(round(p[0])), int(round(p[1])))


def _dotted(img, pts, col, step=11.0, rad=2):
    """Dotted polyline: a dot every `step` px along the path (dark halo so it reads on any felt/photo)."""
    if len(pts) < 2:
        return
    P = np.array(pts, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    for d in np.arange(0, cum[-1] + 1e-6, step):
        i = min(int(np.searchsorted(cum, d, side="right")) - 1, len(seg) - 1)
        t = (d - cum[i]) / (seg[i] or 1.0)
        q = P[i] + (P[i + 1] - P[i]) * min(1.0, t)
        cv2.circle(img, _pt(q), rad + 1, (0, 0, 0, 230), -1, cv2.LINE_AA)
        cv2.circle(img, _pt(q), rad, col, -1, cv2.LINE_AA)


def _dotted_circle(img, c, r, col, n=18, rad=2):
    for i in range(n):
        a = 2 * math.pi * i / n
        q = (c[0] + r * math.cos(a), c[1] + r * math.sin(a))
        cv2.circle(img, _pt(q), rad + 1, (0, 0, 0, 230), -1, cv2.LINE_AA)
        cv2.circle(img, _pt(q), rad, col, -1, cv2.LINE_AA)


def _draw_option(img, sess, o):
    """Every ball the shot moves: dotted path + dotted ball where it stops (or goes down)."""
    sim = o.get("sim")
    if not sim:
        return
    R = float(np.median([b.radius for b in sess.balls])) if sess.balls else 14.0
    for m in sim["moves"]:
        col = _COL.get(m["label"], _COL["unknown"])
        _dotted(img, m["path"], col)
        end = m["end"]
        if m["pocketed"]:
            _dotted_circle(img, end, R * 0.8, col)
            cv2.putText(img, "IN", (int(end[0]) - 9, int(end[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255, 255), 2, cv2.LINE_AA)
        else:
            _dotted_circle(img, end, R, col)
            tx = f"{m['id']}"
            (tw, th), _ = cv2.getTextSize(tx, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.putText(img, tx, (int(end[0]) - tw // 2, int(end[1]) + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255, 255), 1, cv2.LINE_AA)
    g = sim.get("ghost")
    if g:  # where the cue ball is when it touches the object ball
        _dotted_circle(img, g, R, (255, 255, 255, 255), n=14, rad=1)



# ---------------------------------------------------------------- shot picking

def _unit(a):
    return a / (np.linalg.norm(a) + 1e-9)


def _legal(b, group: str, only_eight: bool) -> bool:
    if b.label in ("cue", "unknown"):
        return False
    if only_eight:
        return b.label == "eight"
    if b.label == "eight":
        return False
    return group == "open" or b.label == group


def _group_cleared(sess, group):
    return group in ("solid", "stripe") and not any(b.label == group for b in sess.balls)


def _offense(sess: Session, group: str) -> list[dict]:
    only8 = _group_cleared(sess, group)
    shots = v.compute_shots(sess.rect, sess.balls, sess.pockets)
    out = []
    for s in shots:
        if s.is_combo:
            continue  # combos are in the engine but rarely the right pick; keep the list readable
        tgt = _nearest_ball(sess, s.ball_center)
        if tgt is None or not _legal(tgt, group, only8):
            continue
        phys = v.build_physics_summary(sess.rect, sess.balls, s)
        out.append({"kind": "offense", "raw": float(s.path_score), "target": tgt.id, "target_label": tgt.label,
                    "ball": s.ball_center, "pocket": s.pocket_center, "ghost": s.ghost_point, "physics": phys})
    best = {}
    for o in out:  # best per (ball, pocket)
        k = (o["target"], tuple(np.round(o["pocket"])))
        if k not in best or o["raw"] > best[k]["raw"]:
            best[k] = o
    return sorted(best.values(), key=lambda o: -o["raw"])


def _nearest_ball(sess, pt):
    if not sess.balls:
        return None
    return min(sess.balls, key=lambda b: math.hypot(b.center[0] - pt[0], b.center[1] - pt[1]))


def _defense(sess: Session, group: str) -> list[dict]:
    """Safeties: touch a legal ball thin, send the cue ball where the opponent's balls can't be hit directly."""
    cue = next((b for b in sess.balls if b.label == "cue"), None)
    if cue is None:
        return []
    h, w = sess.rect.shape[:2]
    R = float(np.median([b.radius for b in sess.balls]))
    only8 = _group_cleared(sess, group)
    legal = [b for b in sess.balls if _legal(b, group, only8)]
    opp = [b for b in sess.balls if b.label not in ("cue", "unknown") and b not in legal and not (b.label == "eight" and not only8 and False)]
    if group == "open":
        opp = [b for b in sess.balls if b.label not in ("cue", "unknown", "eight")]
    diag = math.hypot(w, h)
    out = []
    for t in legal:
        best = None
        base = _unit(t.center - cue.center)
        for deg in (-55, -40, -25, 25, 40, 55):
            a = math.radians(deg)
            rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], np.float32)
            n = rot @ base                     # direction the object ball will travel
            ghost = t.center - n * (2 * R)     # where the cue ball sits at contact
            if not (R <= ghost[0] <= w - R and R <= ghost[1] <= h - R):
                continue
            if not v.segment_clear(cue.center, ghost, sess.balls, [sess.balls.index(t), sess.balls.index(cue)], R * 0.9):
                continue
            vd = _unit(ghost - cue.center)
            tang = vd - np.dot(vd, n) * n      # cue ball's leftover direction after the hit
            if np.linalg.norm(tang) < 1e-3:
                continue
            sin_t = float(np.linalg.norm(tang))
            tang = _unit(tang)
            travel = sin_t * 0.45 * w
            cue_end = np.clip(ghost + tang * travel, [R, R], [w - R, h - R])
            obj_end = np.clip(t.center + n * (0.25 * w), [R, R], [w - R, h - R])
            if not v.segment_clear(ghost, cue_end, sess.balls, [sess.balls.index(t), sess.balls.index(cue)], R * 0.9):
                continue  # the cue ball would hit something else on the way
            if any(math.hypot(*(cue_end - p)) < 2.2 * R for p in sess.pockets):
                continue  # don't scratch
            if any(math.hypot(*(obj_end - p)) < 1.5 * R for p in sess.pockets):
                continue  # that would pot it (an offense shot, not a safety)
            others = [x for x in opp if x is not t]
            if others:
                idx_t = sess.balls.index(t)
                blocked = sum(1 for o in others
                              if not v.segment_clear(cue_end, o.center, sess.balls, [sess.balls.index(o), sess.balls.index(cue), idx_t], R * 0.95))
                blocked_frac = blocked / len(others)
                dist_n = float(np.mean([math.hypot(*(cue_end - o.center)) for o in others])) / diag
            else:
                blocked_frac, dist_n = 0.0, 0.5
            rail = min(obj_end[0], w - obj_end[0], obj_end[1], h - obj_end[1])
            rail_bonus = 1.0 if rail < 3 * R else 0.0
            score = 0.6 * blocked_frac + 0.3 * min(1.0, dist_n * 2.0) + 0.1 * rail_bonus
            cand = {"kind": "defense", "raw": float(score), "target": t.id, "target_label": t.label,
                    "ball": (float(t.center[0]), float(t.center[1])), "ghost": (float(ghost[0]), float(ghost[1])),
                    "cue_end": (float(cue_end[0]), float(cue_end[1])), "object_end": (float(obj_end[0]), float(obj_end[1])),
                    "blocked": f"{int(round(blocked_frac * 100))}% of opponent lines blocked", "cut_deg": abs(deg)}
            if best is None or cand["raw"] > best["raw"]:
                best = cand
        if best:
            out.append(best)
    return sorted(out, key=lambda o: -o["raw"])


# ---------------------------------------------------------------- simulation

A_ROLL = 0.06     # rolling deceleration, px/step^2
E_BALL = 0.96     # ball-ball restitution
E_RAIL = 0.78     # cushion restitution
V_MAX = 22.0


def _bounds(sess, R):
    P = np.array(sess.pockets, float)
    return (P[:, 0].min() + R, P[:, 1].min() + R, P[:, 0].max() - R, P[:, 1].max() - R)


def simulate(sess: Session, cue_dir, v0: float, spin: float = 0.0, max_steps: int = 6000) -> dict:
    """Replay one stroke. Positions in rectified px. spin: >0 follow, 0 stun, <0 draw (cue ball only)."""
    R = float(np.median([b.radius for b in sess.balls]))
    x0, y0, x1, y1 = _bounds(sess, R)
    cap = 1.5 * max(sess.pocket_r or 0, 1.5 * R)
    P = [np.array(p, float) for p in sess.pockets]
    balls = sess.balls
    n = len(balls)
    pos = np.array([b.center for b in balls], float)
    vel = np.zeros((n, 2))
    ci = next(i for i, b in enumerate(balls) if b.label == "cue")
    vel[ci] = _unit(np.array(cue_dir, float)) * v0
    alive = [True] * n
    pocket_of = {}
    paths = [[pos[i].copy()] for i in range(n)]
    last_rec = pos.copy()
    contacts, rails = [], [0] * n
    touched = set()
    spun = False
    sub = max(1, int(math.ceil(v0 / 5.0)))
    dt = 1.0 / sub
    for step in range(max_steps * sub):
        mov = [i for i in range(n) if alive[i] and np.linalg.norm(vel[i]) > 1e-3]
        if not mov:
            break
        for i in mov:
            sp = np.linalg.norm(vel[i])
            vel[i] *= max(0.0, sp - A_ROLL * dt) / sp
            pos[i] += vel[i] * dt
        for i in mov:
            if not alive[i]:
                continue
            for k, pk in enumerate(P):  # pocket capture
                if np.linalg.norm(pos[i] - pk) < cap:
                    alive[i] = False
                    pocket_of[i] = k
                    pos[i] = pk
                    vel[i] = 0
                    paths[i].append(pk.copy())
                    break
            if not alive[i]:
                continue
            for ax, lo, hi in ((0, x0, x1), (1, y0, y1)):  # cushions
                if pos[i][ax] < lo:
                    pos[i][ax] = lo + (lo - pos[i][ax]); vel[i][ax] = abs(vel[i][ax]) * E_RAIL; rails[i] += 1
                    paths[i].append(pos[i].copy()); last_rec[i] = pos[i]
                elif pos[i][ax] > hi:
                    pos[i][ax] = hi - (pos[i][ax] - hi); vel[i][ax] = -abs(vel[i][ax]) * E_RAIL; rails[i] += 1
                    paths[i].append(pos[i].copy()); last_rec[i] = pos[i]
        for i in mov:  # ball-ball
            if not alive[i]:
                continue
            for j in range(n):
                if j == i or not alive[j]:
                    continue
                d = pos[j] - pos[i]
                dist = float(np.linalg.norm(d))
                if dist >= 2 * R or dist < 1e-6:
                    continue
                nrm = d / dist
                rel = float(np.dot(vel[i] - vel[j], nrm))
                if rel <= 0:
                    continue
                if j in mov and j < i:
                    continue  # pair handled once
                imp = (1 + E_BALL) / 2 * rel
                vi_before = vel[i].copy()
                vel[i] -= imp * nrm
                vel[j] += imp * nrm
                overlap = 2 * R - dist
                pos[i] -= nrm * overlap / 2
                pos[j] += nrm * overlap / 2
                if ci in (i, j) and not spun:
                    c, o = (i, j) if i == ci else (j, i)
                    contacts.append({"a": balls[c].id, "b": balls[o].id, "at": pos[c].tolist()})
                    spun = True
                    if i == ci:
                        vel[ci] += spin * abs(float(np.dot(vi_before, nrm))) * nrm
                elif (balls[i].id, balls[j].id) not in touched and (balls[j].id, balls[i].id) not in touched:
                    contacts.append({"a": balls[i].id, "b": balls[j].id, "at": pos[i].tolist()})
                touched.add((balls[i].id, balls[j].id))
                paths[i].append(pos[i].copy()); paths[j].append(pos[j].copy())
                last_rec[i] = pos[i]; last_rec[j] = pos[j]
        for i in mov:
            if alive[i] and np.linalg.norm(pos[i] - last_rec[i]) >= 6:
                paths[i].append(pos[i].copy()); last_rec[i] = pos[i].copy()
    moves = []
    for i in range(n):
        if np.linalg.norm(pos[i] - np.array(balls[i].center)) < 0.5 * R and i not in pocket_of:
            continue
        pth = paths[i] + ([pos[i].copy()] if i not in pocket_of else [])
        moves.append({"id": balls[i].id, "label": balls[i].label, "path": [[float(a), float(b)] for a, b in pth],
                      "start": [float(balls[i].center[0]), float(balls[i].center[1])],
                      "end": [float(pos[i][0]), float(pos[i][1])], "pocketed": i in pocket_of,
                      "pocket": pocket_of.get(i), "rails": rails[i]})
    moves.sort(key=lambda m: m["label"] != "cue")
    cm = next((m for m in moves if m["label"] == "cue"), None)
    return {"moves": moves, "contacts": contacts, "cue_scratch": bool(cm and cm["pocketed"]),
            "first": contacts[0]["b"] if contacts and contacts[0]["a"] == balls[ci].id else None,
            "ghost": contacts[0]["at"] if contacts and contacts[0]["a"] == balls[ci].id else None}


def _pocket_name(sess, k):
    P = np.array(sess.pockets, float)
    c = P.mean(axis=0)
    p = P[k]
    dx, dy = p[0] - c[0], p[1] - c[1]
    span_x = (P[:, 0].max() - P[:, 0].min()) or 1
    side = abs(dx) < 0.2 * span_x
    v_ = "top" if dy < 0 else "bottom"
    if side:
        return f"{v_} side pocket"
    return f"{v_}-{'left' if dx < 0 else 'right'} corner"


def _zone(sess, p, R):
    x0, y0, x1, y1 = _bounds(sess, R)
    fx, fy = (p[0] - x0) / max(1, x1 - x0), (p[1] - y0) / max(1, y1 - y0)
    h_ = "left" if fx < 0.33 else "right" if fx > 0.67 else "centre"
    v_ = "top" if fy < 0.33 else "bottom" if fy > 0.67 else "middle"
    name = "centre of the table" if (h_, v_) == ("centre", "middle") else (
        f"{v_} {h_}" if v_ != "middle" and h_ != "centre" else (f"{v_} centre" if h_ == "centre" else f"{h_} side"))
    rail = min(p[0] - x0, x1 - p[0], p[1] - y0, y1 - p[1]) < 1.6 * R
    return name + (" (on the rail)" if rail else "")


def _effects(sess, sim, R) -> list:
    out = []
    first = sim.get("first")
    for m in sim["moves"]:
        if m["label"] == "cue":
            if m["pocketed"]:
                out.append("Cue ball goes in the pocket: scratch.")
            else:
                r = f", after {m['rails']} rail{'s' if m['rails'] != 1 else ''}" if m["rails"] else ""
                out.append(f"Cue ball finishes {_zone(sess, m['end'], R)}{r}.")
        elif m["pocketed"]:
            out.append(f"{m['id']} ({m['label']}) is pocketed in the {_pocket_name(sess, m['pocket'])}.")
        else:
            verb = "is hit first and finishes" if m["id"] == first else "gets knocked to"
            out.append(f"{m['id']} ({m['label']}) {verb} {_zone(sess, m['end'], R)}.")
    return out


def _best_speed_offense(sess, o, R, cue):
    ghost = np.array(o["ghost"], float)
    ball = np.array(o["ball"], float)
    pk = np.array(o["pocket"], float)
    cue_c = np.array(cue.center, float)
    d1 = float(np.linalg.norm(ghost - cue_c))
    d2 = float(np.linalg.norm(pk - ball))
    cos = max(0.25, float(np.dot(_unit(ghost - cue_c), _unit(pk - ball))))
    vo = math.sqrt(2 * A_ROLL * d2 * 1.35)
    vc = vo / (0.98 * cos)
    return min(V_MAX, math.sqrt(vc * vc + 2 * A_ROLL * d1))


def _pos_score(sess, end, R):
    """Higher = cue ball ends somewhere comfortable: off the rail, away from pockets."""
    x0, y0, x1, y1 = _bounds(sess, R)
    rail = min(end[0] - x0, x1 - end[0], end[1] - y0, y1 - end[1])
    pk = min(math.hypot(end[0] - p[0], end[1] - p[1]) for p in sess.pockets)
    return 0.5 * min(1.0, rail / (4 * R)) + 0.5 * min(1.0, pk / (6 * R))


def _sim_offense(sess, o, R, cue, group, only8):
    v0 = _best_speed_offense(sess, o, R, cue)
    best = None
    for name, spin in (("follow", 0.35), ("stun", 0.0), ("draw", -0.3)):
        s = simulate(sess, np.array(o["ghost"]) - np.array(cue.center), v0, spin)
        if s["cue_scratch"] or s["first"] != o["target"]:
            continue
        pot = [m for m in s["moves"] if m["pocketed"] and m["label"] != "cue"]
        if not any(m["id"] == o["target"] for m in pot):
            continue
        bad = [m for m in pot if not _legal_pot(m["label"], group, only8)]
        cm = next(m for m in s["moves"] if m["label"] == "cue")
        score = _pos_score(sess, cm["end"], R) - (0.6 if bad else 0.0) + 0.1 * (len(pot) - 1 - len(bad) > 0)
        if best is None or score > best[0]:
            best = (score, name, s, v0)
    return best


def _legal_pot(label, group, only8):
    if only8:
        return label == "eight"
    return label != "eight" and (group == "open" or label == group)


def _score_defense(sess, s, target, group, only8, R):
    cm = next(m for m in s["moves"] if m["label"] == "cue")
    opp = [b for b in sess.balls if b.label not in ("cue", "unknown") and not _legal(b, group, only8) and b.id != target
           and not any(m["id"] == b.id and m["pocketed"] for m in s["moves"])]
    if group == "open":
        opp = [b for b in opp if b.label != "eight"]
    final = []
    for b in sess.balls:
        mv = next((m for m in s["moves"] if m["id"] == b.id), None)
        if mv and mv["pocketed"]:
            continue
        c = np.array(mv["end"] if mv else b.center, float)
        final.append((b, c))
    cue_end = np.array(cm["end"], float)
    blocked = 0
    for b in opp:
        c = np.array(next(m["end"] for m in s["moves"] if m["id"] == b.id) if any(m["id"] == b.id for m in s["moves"]) else b.center, float)
        clear = True
        d = c - cue_end
        L = float(np.linalg.norm(d)) or 1.0
        u = d / L
        for bb, cc in final:
            if bb.id in (b.id, "CUE") or bb.label == "cue":
                continue
            t = float(np.dot(cc - cue_end, u))
            if 0 < t < L and float(np.linalg.norm(cue_end + u * t - cc)) < 1.9 * R:
                clear = False
                break
        blocked += 0 if clear else 1
    frac = blocked / len(opp) if opp else 0.0
    x0, y0, x1, y1 = _bounds(sess, R)
    diag = math.hypot(x1 - x0, y1 - y0)
    dist_n = float(np.mean([math.hypot(*(cue_end - np.array(b.center))) for b in opp])) / diag if opp else 0.5
    oe = next((np.array(m["end"]) for m in s["moves"] if m["id"] == target), np.array([x0, y0]))
    rail = min(oe[0] - x0, x1 - oe[0], oe[1] - y0, y1 - oe[1]) < 3 * R
    return 0.6 * frac + 0.3 * min(1.0, dist_n * 2.0) + 0.1 * (1.0 if rail else 0.0), frac


def _sim_defense(sess, o, R, cue, group, only8):
    ghost = np.array(o["ghost"], float)
    ball = np.array(o["ball"], float)
    oe = np.array(o["object_end"], float)
    cue_c = np.array(cue.center, float)
    d1 = float(np.linalg.norm(ghost - cue_c))
    d2 = float(np.linalg.norm(oe - ball))
    nrm = _unit(oe - ball)
    cos = max(0.25, float(np.dot(_unit(ghost - cue_c), nrm)))
    vo = math.sqrt(2 * A_ROLL * d2)
    v0 = min(V_MAX, math.sqrt((vo / (0.98 * cos)) ** 2 + 2 * A_ROLL * d1))
    s = simulate(sess, ghost - cue_c, v0, 0.0)
    if s["cue_scratch"] or s["first"] != o["target"]:
        return None
    if any(m["pocketed"] and not _legal_pot(m["label"], group, only8) for m in s["moves"] if m["label"] != "cue"):
        return None
    score, frac = _score_defense(sess, s, o["target"], group, only8, R)
    return score, frac, s


def _describe(o: dict) -> str:
    if o["kind"] == "offense":
        ph = o.get("physics") or {}
        cut = ph.get("cut_angle_degrees")
        bits = [f"Pot {o['target']} ({o['target_label']})"]
        if cut is not None:
            bits.append(f"about {cut:.0f}° cut {ph.get('cut_direction', '')}")
        if o.get("spin"):
            bits.append(f"{o['spin']} on the cue ball")
        if ph.get("force"):
            bits.append(f"{ph['force']['category']} speed")
        return ", ".join(bits)
    return (f"Safety: touch {o['target']} ({o['target_label']}) thin, leaving the cue ball where "
            f"{o['blocked']}; no pot intended")


def top_shots(sess: Session, group: str = "open", n: int = 3) -> list[dict]:
    """Best n options overall (each replayed by the simulator), always including a safety when one exists."""
    group = group if group in ("open", "solid", "stripe") else "open"
    only8 = _group_cleared(sess, group)
    cue = next((b for b in sess.balls if b.label == "cue"), None)
    if cue is None or len(sess.pockets) < 3:
        return []
    R = float(np.median([b.radius for b in sess.balls]))
    off, de = [], []
    for o in _offense(sess, group)[:8]:
        r = _sim_offense(sess, o, R, cue, group, only8)
        if not r:
            continue
        score, spin, s, v0 = r
        o["sim"], o["spin"], o["raw"] = s, {"follow": "follow", "stun": "", "draw": "draw"}[spin], o["raw"] * (0.7 + 0.3 * score)
        o["speed"] = round(v0, 1)
        off.append(o)
    for o in _defense(sess, group)[:6]:
        r = _sim_defense(sess, o, R, cue, group, only8)
        if not r:
            continue
        score, frac, s = r
        o["sim"], o["raw"] = s, score
        o["blocked"] = f"{int(round(frac * 100))}% of opponent lines blocked"
        de.append(o)
    omax = max([o["raw"] for o in off], default=1.0) or 1.0
    dmax = max([o["raw"] for o in de], default=1.0) or 1.0
    for o in off:
        o["rating"] = round(float(100 * o["raw"] / omax * 0.95), 1)
    for o in de:
        o["rating"] = round(100 * min(1.0, o["raw"] / max(dmax, 0.5)) * 0.8, 1)
    pool = sorted(off + de, key=lambda o: -o["rating"])
    chosen = pool[:n]
    if de and not any(o["kind"] == "defense" for o in chosen):
        chosen = chosen[:n - 1] + [sorted(de, key=lambda o: -o["rating"])[0]]
    chosen.sort(key=lambda o: -o["rating"])
    for i, o in enumerate(chosen, 1):
        o["rank"] = i
        o["summary"] = _describe(o)
        o["effects"] = _effects(sess, o["sim"], R)
        o["legend"] = [{"id": m["id"], "label": m["label"], "pocketed": m["pocketed"]} for m in o["sim"]["moves"]]
        o.pop("raw", None)
    return chosen

