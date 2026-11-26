# Pool Cue Guide

Upload a photo of a pool table. The app finds and labels the balls (B1, B2, ...), draws a transparent overlay on the top-down image, and lets you fix mistakes: type a label, tick **Improve** (re-detect that ball) or **Remove** (not a ball), and re-run. Repeat until you press **Approved**, then it returns the 3 best options, always including a safety when one exists.

```bash
pip install -r requirements.txt
uvicorn cueguide.app:app --port 8000     # http://localhost:8000
python -m pytest -q tests
```

Deploy on Railway: connect the repo; the `Dockerfile` and `railway.json` are picked up (health check `/healthz`). Keep one worker: sessions are held in memory for an hour.

**Pocket markers (recommended).** Cover the six pockets with bright green (neon) discs. Their centres become the table outline: the four corner pockets set a perspective-correct top-down view, and the playing area is just inside them. With 4 markers (corners only) the side pockets are derived. With any other count the app falls back to automatic detection (felt colour + shape), which works best on top-down photos.

**Tap to place.** If detection is off (or fails), open **Tap to place** and tap the pockets and balls on your photo. Each tap adds a row; choose what it is from the dropdown (Pocket 1-6, cue ball, solids 1-7, 8-ball, stripes 9-15, or "number unknown"). You need at least the four corner pockets; the middle pockets are derived if you skip them. Taps are snapped to the real ball centres, your labels are kept through later review rounds, and **Start from detected** pre-fills the list from the automatic result.

Detection is heuristic (felt colour + Hough circles + colour blobs). Ratings and speed/spin numbers are estimates.

## Shot effects and your group

- Choose **Open table / Solids / Stripes** in the review step or on the results page; changing it re-ranks the options immediately.
- Each option is replayed by a small physics simulator (ball-ball collisions, cushion bounces, rolling friction, pockets, cue-ball follow/draw). The overlay draws a **dotted line** for every ball that moves and a **dotted circle** where it stops (`IN` = pocketed). The card lists the effects in words (where the cue ball ends up, which balls are knocked where). Shots that scratch or fail to pot are dropped.
- The simulator uses estimated constants, so treat paths as a guide, not a guarantee.

## Random practice table

Click **🎲 Random table** on the upload screen (or **New random table** while reviewing) to get a made-up table with randomly placed balls, drawn on `cueguide/assets/table_template.png`. Remove balls (tick Remove and Re-run, or delete a point in Tap to place) and add balls (tap the table and pick the ball), then Build and Approve as usual. API: `POST /api/random` with optional `{"balls": 6-16, "seed": n}`.
