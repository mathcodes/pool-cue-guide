"""FastAPI service: upload a table photo, review labelled balls, approve, get the 3 best shots."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import engine, randomtable

log = logging.getLogger('cueguide')

MAX_UPLOAD = 12 * 1024 * 1024
MAX_SESSIONS = 20
TTL = 45 * 60
STATIC = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Pool Cue Guide")
SESSIONS: dict[str, engine.Session] = {}
OPTIONS: dict[str, list] = {}
GROUPS: dict[str, str] = {}


class Review(BaseModel):
    improve: list[str] = Field(default_factory=list, max_length=40)
    remove: list[str] = Field(default_factory=list, max_length=40)
    find_missing: bool = False


class Item(BaseModel):
    x: float
    y: float
    kind: str = Field(max_length=12)


class Manual(BaseModel):
    items: list[Item] = Field(max_length=40)


class Approve(BaseModel):
    group: str = "open"  # open | solid | stripe


def _get(sid: str) -> engine.Session:
    now = time.time()
    for k in [k for k, s in SESSIONS.items() if now - s.created > TTL]:
        SESSIONS.pop(k, None)
        OPTIONS.pop(k, None)
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "Session not found or expired. Upload the photo again.")
    return s


def _state(s: engine.Session, extra: Optional[dict] = None) -> dict:
    h, w = s.rect.shape[:2] if s.rect is not None else (s.src_size[1], s.src_size[0])
    out = {
        "id": s.id, "mode": s.mode, "message": s.message, "pockets": len(s.pockets), "round": s.round,
        "width": w, "height": h, "src_width": s.src_size[0], "src_height": s.src_size[1], "history": s.history,
        "balls": [{"id": b.id, "label": b.label, "x": round(float(b.center[0]), 1),
                   "y": round(float(b.center[1]), 1), "r": round(float(b.radius), 1)} for b in s.balls],
    }
    if extra:
        out.update(extra)
    return out


def _ready(s: engine.Session) -> engine.Session:
    if s.rect is None:
        raise HTTPException(409, "Tap the pockets and balls on the photo first.")
    return s


@app.post("/api/sessions")
async def create(file: UploadFile = File(...)):
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image is over 12 MB.")
    try:
        src = engine.decode_image(data)
    except engine.AnalysisError as e:
        raise HTTPException(422, str(e))
    try:
        s = engine.analyze_src(src)
    except engine.AnalysisError as e:
        s = engine.photo_session(src, str(e))   # let the user tap the pockets and balls instead
    if len(SESSIONS) >= MAX_SESSIONS:
        SESSIONS.pop(min(SESSIONS, key=lambda k: SESSIONS[k].created), None)
    SESSIONS[s.id] = s
    return _state(s)


_RAILS = json.loads((Path(__file__).parent / "data" / "rail_shots.json").read_text())


@app.get("/api/rails")
def rails():
    """Stored library of rail-shot examples and diamond systems (built by tools/gen_rails.py; no lookups)."""
    return _RAILS


class RandomReq(BaseModel):
    balls: Optional[int] = Field(None, ge=6, le=16)
    seed: Optional[int] = None


@app.post("/api/random")
def random_table(body: RandomReq = RandomReq()):
    """A made-up practice table with randomly placed balls, ready to review, edit and approve."""
    jpg, items = randomtable.make(body.seed, body.balls)
    src = engine.decode_image(jpg)
    s = engine.photo_session(src, "")
    try:
        engine.build_manual(s, items)
    except engine.AnalysisError as e:
        raise HTTPException(500, str(e))
    s.message = "random"
    if len(SESSIONS) >= MAX_SESSIONS:
        SESSIONS.pop(min(SESSIONS, key=lambda k: SESSIONS[k].created), None)
    SESSIONS[s.id] = s
    return _state(s, {"random": True, "items": engine.prefill_items(s)})


class Play(BaseModel):
    option: int = Field(ge=1, le=10)


@app.post("/api/sessions/{sid}/play")
def play(sid: str, body: Play):
    """Execute one of the suggested shots: balls move to the simulated end positions, then re-rank."""
    s = _ready(_get(sid))
    opts = OPTIONS.get(sid, [])
    if not 1 <= body.option <= len(opts):
        raise HTTPException(409, "Approve first to get shot options.")
    try:
        report = engine.play(s, opts[body.option - 1])
    except engine.AnalysisError as e:
        raise HTTPException(409, str(e))
    group = GROUPS.get(sid, "open")
    shots = engine.top_shots(s, group)
    OPTIONS[sid] = shots
    left = [b for b in s.balls if b.label not in ("cue",)]
    msg = "" if shots else ("Table cleared." if not left else "No clear shot from here.")
    return _state(s, {"approved": True, "group": group, "options": shots, "played": s.played, "report": report, "note": msg})


@app.post("/api/sessions/{sid}/undo")
def undo(sid: str):
    s = _ready(_get(sid))
    if not engine.undo(s):
        raise HTTPException(409, "Nothing to undo.")
    group = GROUPS.get(sid, "open")
    shots = engine.top_shots(s, group)
    OPTIONS[sid] = shots
    return _state(s, {"approved": True, "group": group, "options": shots, "played": s.played})


@app.get("/api/sessions/{sid}")
def state(sid: str):
    return _state(_get(sid))


@app.get("/api/sessions/{sid}/image.png")
def image(sid: str):
    ok, buf = cv2.imencode(".png", engine.render_base(_ready(_get(sid))))
    return Response(buf.tobytes(), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/sessions/{sid}/overlay.png")
def overlay(sid: str, option: int = 0, pick: str = ""):
    s = _ready(_get(sid))
    opts = OPTIONS.get(sid, [])
    o = opts[option - 1] if 1 <= option <= len(opts) else None
    ids = [p.strip().upper() for p in pick.split(",") if p.strip()]
    return Response(engine.overlay_png(s, o, ids), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/sessions/{sid}/photo.jpg")
def photo(sid: str):
    return Response(_get(sid).src_jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/sessions/{sid}/prefill")
def prefill(sid: str):
    return {"items": engine.prefill_items(_get(sid))}


@app.post("/api/sessions/{sid}/manual")
def manual(sid: str, body: Manual):
    s = _get(sid)
    OPTIONS.pop(sid, None)
    try:
        report = engine.build_manual(s, [i.model_dump() for i in body.items])
    except engine.AnalysisError as e:
        raise HTTPException(422, str(e))
    return _state(s, {"report": report})


@app.post("/api/sessions/{sid}/review")
def review(sid: str, body: Review):
    s = _ready(_get(sid))
    if not (body.improve or body.remove or body.find_missing):
        raise HTTPException(400, "Mark at least one ball to improve or remove.")
    OPTIONS.pop(sid, None)
    report = engine.refine(s, body.improve, body.remove, body.find_missing)
    return _state(s, {"report": report})


@app.post("/api/sessions/{sid}/approve")
def approve(sid: str, body: Approve):
    s = _ready(_get(sid))
    if not any(b.label == "cue" for b in s.balls):
        raise HTTPException(409, "No cue ball is marked. Improve or add the cue ball first.")
    try:
        shots = engine.top_shots(s, body.group)
    except Exception:  # never leave the user hanging on a spinner: report it
        log.exception("top_shots failed")
        raise HTTPException(500, "Could not work out shots for this layout. Try Back to review and adjust a ball.")
    OPTIONS[sid] = shots
    GROUPS[sid] = body.group
    return _state(s, {"approved": True, "group": body.group, "options": shots})


@app.get("/healthz")
def health():
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
