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
    rect: np.ndarray
    table_mask: np.ndarray
    pockets: list
    balls: list
    felt_hsv: np.ndarray
    suppressed: list = field(default_factory=list)  # (x, y, r) zones the user removed
    round: int = 0
    next_id: int = 1
    created: float = field(default_factory=time.time)
    history: list = field(default_factory=list)


class AnalysisError(Exception):
    pass


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


def _blob_balls(rect, mask, felt, pockets, known, R, thr: float = 38) -> list:
    """Colour-distance-from-felt blobs: finds green/red/blue balls that grayscale Hough misses."""
    h, w = rect.shape[:2]
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV).astype(np.float32)
    dh = np.minimum(np.abs(hsv[:, :, 0] - felt[0]), 180 - np.abs(hsv[:, :, 0] - felt[0]))
    d = 2.0 * dh + np.abs(hsv[:, :, 1] - felt[1]) / 2 + np.abs(hsv[:, :, 2] - felt[2]) / 3
    blob = ((d > thr) & (mask > 0)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, k, iterations=2)
    out = []
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if not (0.55 * math.pi * R * R <= area <= 1.5 * math.pi * R * R):
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if area < 0.7 * math.pi * r * r:
            continue  # not round: a smudge, a merged pair or a rail edge
        if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
            continue
        if any(math.hypot(x - p[0], y - p[1]) < 1.6 * R for p in pockets):
            continue
        if any(math.hypot(x - b.center[0], y - b.center[1]) < 1.2 * R for b in known):
            continue
        out.append(v.Ball(center=np.array([x, y], dtype=np.float32), radius=float(r),
                          mean_bgr=_patch_mean(rect, int(x), int(y), int(r))))
    return out


def analyze(data: bytes) -> Session:
    src = decode_image(data)
    corners = v.find_table_corners(src)
    if corners is None:
        raise AnalysisError(
            "Couldn't find the table. Take the photo from above with the whole felt in view and even lighting."
        )
    rect, _ = v.warp_to_topdown(src, corners, target_short=900)
    mask = v.detect_table_mask(rect)
    pockets = v.detect_pockets(rect, mask)
    balls = v.detect_balls(rect, mask)
    felt = _felt_hsv(rect, mask)
    if felt[1] < 35:
        raise AnalysisError("That doesn't look like a pool table (no felt colour found). Try another photo.")
    R0 = float(np.median([b.radius for b in balls])) if balls else 0.025 * rect.shape[1]
    balls += _blob_balls(rect, mask, felt, pockets, balls, R0)
    v.classify_balls(rect, balls)
    if len(balls) < 2:
        raise AnalysisError("Couldn't find the balls. Use a sharper, brighter photo from above.")
    sess = Session(id=uuid.uuid4().hex[:12], rect=rect, table_mask=mask, pockets=pockets, balls=balls,
                   felt_hsv=felt)
    _assign_ids(sess, sess.balls)
    sess.history.append({"round": 0, "action": "initial detection", "balls": len(balls)})
    return sess


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
    f = sess.felt_hsv
    dh = np.minimum(np.abs(hsv[:, :, 0] - f[0]), 180 - np.abs(hsv[:, :, 0] - f[0]))
    d = 2.0 * dh + np.abs(hsv[:, :, 1] - f[1]) / 2 + np.abs(hsv[:, :, 2] - f[2]) / 3
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
    """Looser global pass that only adds balls not already known or suppressed."""
    rect, mask = sess.rect, sess.table_mask
    h, w = rect.shape[:2]
    gray = cv2.medianBlur(cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY), 5)
    R = float(np.median([b.radius for b in sess.balls])) if sess.balls else 0.025 * w
    p2 = max(8, 20 - 3 * (sess.round + 1))
    cs = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(R * 1.8), param1=100, param2=p2,
                          minRadius=int(R * 0.8), maxRadius=int(R * 1.25))
    added = []
    if cs is not None:
        for x, y, r in np.round(cs[0]).astype(int):
            if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h or mask[y, x] == 0:
                continue
            if any(math.hypot(x - b.center[0], y - b.center[1]) < 1.5 * R for b in sess.balls):
                continue
            if any(math.hypot(x - sx, y - sy) < sr for sx, sy, sr in sess.suppressed):
                continue
            nb = v.Ball(center=np.array([x, y], dtype=np.float32), radius=float(r), mean_bgr=_patch_mean(rect, x, y, int(r)))
            added.append(nb)
    extra = [b for b in _blob_balls(rect, mask, sess.felt_hsv, sess.pockets, sess.balls + added, R, thr=max(20, 30 - 4 * sess.round))
             if not any(math.hypot(b.center[0] - sx, b.center[1] - sy) < sr for sx, sy, sr in sess.suppressed)]
    added += extra
    sess.balls.extend(added)
    return len(added)


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
    for b in sess.balls:
        b.label = "unknown"
    v.classify_balls(sess.rect, sess.balls)
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
    for p in sess.pockets:
        x, y = int(p[0]), int(p[1])
        cv2.rectangle(img, (x - 9, y - 9), (x + 9, y + 9), (0, 255, 255, 255), 2, cv2.LINE_AA)
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


def _draw_option(img, sess, o):
    cue = next((b for b in sess.balls if b.label == "cue"), None)
    if cue is None:
        return
    c, g = _pt(cue.center), _pt(o["ghost"])
    cv2.arrowedLine(img, c, g, (255, 255, 255, 255), 3, cv2.LINE_AA, tipLength=0.04)
    cv2.circle(img, g, 6, (255, 255, 255, 255), -1, cv2.LINE_AA)
    if o["kind"] == "offense":
        cv2.line(img, g, _pt(o["ball"]), (255, 120, 0, 255), 2, cv2.LINE_AA)
        cv2.line(img, _pt(o["ball"]), _pt(o["pocket"]), (60, 60, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(img, _pt(o["pocket"]), 8, (60, 60, 255, 255), -1, cv2.LINE_AA)
    else:
        cv2.line(img, g, _pt(o["ball"]), (255, 120, 0, 255), 2, cv2.LINE_AA)
        cv2.line(img, _pt(o["ball"]), _pt(o["object_end"]), (255, 120, 0, 255), 2, cv2.LINE_AA)
        cv2.line(img, g, _pt(o["cue_end"]), (255, 0, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(img, _pt(o["cue_end"]), int(cue.radius), (255, 0, 255, 255), 2, cv2.LINE_AA)


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


def _describe(o: dict) -> str:
    if o["kind"] == "offense":
        ph = o.get("physics") or {}
        cut = ph.get("cut_angle_degrees")
        bits = [f"Pot {o['target']} ({o['target_label']})"]
        if cut is not None:
            bits.append(f"about {cut:.0f}° cut {ph.get('cut_direction', '')}")
        if ph.get("force"):
            bits.append(f"{ph['force']['category']} speed")
        return ", ".join(bits)
    return (f"Safety: touch {o['target']} ({o['target_label']}) thin, leaving the cue ball where "
            f"{o['blocked']}; no pot intended")


def top_shots(sess: Session, group: str = "open", n: int = 3) -> list[dict]:
    """Best n options overall, always including at least one safety when one exists."""
    group = group if group in ("open", "solid", "stripe") else "open"
    off, de = _offense(sess, group), _defense(sess, group)
    omax = max([o["raw"] for o in off], default=1.0) or 1.0
    dmax = max([o["raw"] for o in de], default=1.0) or 1.0
    for o in off:
        o["rating"] = round(100 * o["raw"] / omax * 0.95, 1)   # a clean pot is the baseline
    for o in de:
        o["rating"] = round(100 * min(1.0, o["raw"] / max(dmax, 0.5)) * 0.8, 1)
    pool = sorted(off + de, key=lambda o: -o["rating"])
    chosen = pool[:n]
    if de and not any(o["kind"] == "defense" for o in chosen):
        chosen = chosen[:n - 1] + [de[0]]
    chosen.sort(key=lambda o: -o["rating"])
    for i, o in enumerate(chosen, 1):
        o["rank"] = i
        o["summary"] = _describe(o)
        o.pop("raw", None)
    return chosen
