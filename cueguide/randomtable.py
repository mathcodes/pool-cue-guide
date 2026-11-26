"""Random practice scenario: draws a top-down table photo with randomly placed balls and returns the
exact tap points (pockets + balls) so the session can be built without any detection guesswork."""
from __future__ import annotations

import random
from typing import Optional

import cv2
import numpy as np

from pathlib import Path

TEMPLATE = Path(__file__).parent / "assets" / "table_template.png"
R = 30
FX0, FY0, FX1, FY1 = 100, 98, 1480, 775        # felt area balls may sit in (inside the cushions)
COLORS = {1: (40, 200, 235), 2: (200, 70, 30), 3: (40, 40, 210), 4: (130, 40, 100),
          5: (30, 130, 240), 6: (50, 140, 30), 7: (30, 30, 120), 8: (15, 15, 15)}
POCKETS = [(82, 80), (789, 62), (1497, 80), (82, 791), (789, 809), (1497, 791)]


def _draw_ball(img, x, y, n):
    if n == 0:
        base = (240, 240, 240)
    else:
        base = COLORS[n if n <= 8 else n - 8]
    cv2.circle(img, (x + 5, y + 6), R, (60, 100, 55), -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), R, base, -1, cv2.LINE_AA)
    if n > 8:
        cv2.rectangle(img, (x - R + 2, y - 11), (x + R - 2, y + 11), (245, 245, 245), -1)
        cv2.circle(img, (x, y), R, base, 2, cv2.LINE_AA)
    if n > 0:
        cv2.circle(img, (x, y), 11, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.putText(img, str(n), (x - (10 if n > 9 else 5), y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.circle(img, (x - 8, y - 9), 4, (255, 255, 255), -1, cv2.LINE_AA)


def make(seed: Optional[int] = None, count: Optional[int] = None):
    """-> (jpg bytes, items [{x, y, kind}]). count = balls on the table incl. cue (6-16)."""
    rng = random.Random(seed)
    img = cv2.imread(str(TEMPLATE))
    H, W = img.shape[:2]
    total = max(6, min(16, count or rng.randint(8, 16)))
    nums = [8] + rng.sample([n for n in range(1, 16) if n != 8], total - 2)
    pts = []
    for n in [0] + nums:
        for _ in range(500):
            x, y = rng.randint(FX0 + R, FX1 - R), rng.randint(FY0 + R, FY1 - R)
            if all((x - a) ** 2 + (y - b) ** 2 > (2.5 * R) ** 2 for a, b, _ in pts) and \
               all((x - px) ** 2 + (y - py) ** 2 > (2.6 * R) ** 2 for px, py in POCKETS):
                pts.append((x, y, n))
                break
    for x, y, n in pts:
        _draw_ball(img, x, y, n)
    items = [{"x": float(x), "y": float(y), "kind": f"pocket{i + 1}"} for i, (x, y) in enumerate(POCKETS)]
    items += [{"x": float(x), "y": float(y), "kind": "cue" if n == 0 else f"ball{n}"} for x, y, n in pts]
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes(), items
