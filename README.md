# Pool Cue Guide

Upload a photo of a pool table. The app finds and labels the balls (B1, B2, ...), draws a transparent overlay on the top-down image, and lets you fix mistakes: type a label, tick **Improve** (re-detect that ball) or **Remove** (not a ball), and re-run. Repeat until you press **Approved**, then it returns the 3 best options, always including a safety when one exists.

```bash
pip install -r requirements.txt
uvicorn cueguide.app:app --port 8000     # http://localhost:8000
python -m pytest -q tests
```

Deploy on Railway: connect the repo; the `Dockerfile` and `railway.json` are picked up (health check `/healthz`). Keep one worker: sessions are held in memory for an hour.

Detection is heuristic (felt colour + Hough circles + colour blobs). Ratings and speed/spin numbers are estimates.
