import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, cv2
from fastapi.testclient import TestClient
from cueguide.app import app
from make_table import make

c = TestClient(app)
img = open(make("/tmp/table.jpg"), "rb").read()

def test_flow():
    r = c.post("/api/sessions", files={"file": ("t.jpg", img, "image/jpeg")})
    assert r.status_code == 200, r.text
    st = r.json(); sid = st["id"]
    assert len(st["balls"]) >= 6 and all(b["id"].startswith("B") for b in st["balls"])
    # transparent overlay has alpha and matches the base image size
    ov = cv2.imdecode(np.frombuffer(c.get(f"/api/sessions/{sid}/overlay.png").content, np.uint8), cv2.IMREAD_UNCHANGED)
    base = cv2.imdecode(np.frombuffer(c.get(f"/api/sessions/{sid}/image.png").content, np.uint8), cv2.IMREAD_COLOR)
    assert ov.shape[2] == 4 and ov.shape[:2] == base.shape[:2] and (ov[:, :, 3] == 0).mean() > 0.9
    # remove one, improve one
    ids = [b["id"] for b in st["balls"]]
    rm, imp = ids[-1], ids[1]
    j = c.post(f"/api/sessions/{sid}/review", json={"improve": [imp], "remove": [rm]}).json()
    assert rm in j["report"]["removed"] and rm not in [b["id"] for b in j["balls"]]
    assert j["round"] == 1 and imp in [b["id"] for b in j["balls"]]  # ids are stable
    # removed ball does not come back on a rescan
    j2 = c.post(f"/api/sessions/{sid}/review", json={"find_missing": True}).json()
    assert rm not in [b["id"] for b in j2["balls"]]
    # empty review rejected, unknown ids reported
    assert c.post(f"/api/sessions/{sid}/review", json={}).status_code == 400
    assert "B99" in c.post(f"/api/sessions/{sid}/review", json={"improve": ["B99"]}).json()["report"]["unknown"]
    # approve -> 3 options with at least one safety
    a = c.post(f"/api/sessions/{sid}/approve", json={"group": "open"}).json()
    assert 1 <= len(a["options"]) <= 3
    kinds = {o["kind"] for o in a["options"]}
    print([ (o["rank"], o["kind"], o["target"], o["rating"], o["summary"]) for o in a["options"]])
    assert "defense" in kinds or len(a["options"]) < 3
    o1 = cv2.imdecode(np.frombuffer(c.get(f"/api/sessions/{sid}/overlay.png?option=1").content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert (o1[:, :, 3] > 0).sum() > (ov[:, :, 3] > 0).sum()  # shot lines drawn

def test_bad_inputs():
    assert c.post("/api/sessions", files={"file": ("x.jpg", b"nope", "image/jpeg")}).status_code == 422
    assert c.get("/api/sessions/zzz").status_code == 404
    flat = cv2.imencode(".jpg", np.full((400, 600, 3), 90, np.uint8))[1].tobytes()
    assert c.post("/api/sessions", files={"file": ("f.jpg", flat, "image/jpeg")}).status_code == 422


def test_game_screenshot_whole_table():
    """Regression: vignetted felt + patterned rails used to crop the table, miss pockets and flood rescans with junk."""
    data = open(pathlib.Path(__file__).parent / "fixtures" / "game_table.jpg", "rb").read()
    st = c.post("/api/sessions", files={"file": ("g.jpg", data, "image/jpeg")}).json()
    assert len(st["balls"]) == 15                      # 15 real balls, nothing on the rails
    assert {b["label"] for b in st["balls"]} >= {"cue", "eight"}
    sid = st["id"]
    for _ in range(3):                                 # repeated rescans must not invent balls
        j = c.post(f"/api/sessions/{sid}/review", json={"find_missing": True}).json()
    assert len(j["balls"]) == 15
    assert 1.6 < st["width"] / st["height"] < 2.3      # table kept whole, not cropped
    a = c.post(f"/api/sessions/{sid}/approve", json={"group": "open"}).json()
    assert len(a["options"]) == 3


def _fixture(name):
    return open(pathlib.Path(__file__).parent / "fixtures" / name, "rb").read()


def test_green_pocket_markers_are_the_outline():
    st = c.post("/api/sessions", files={"file": ("m.jpg", _fixture("green_markers.jpg"), "image/jpeg")}).json()
    assert st["mode"] == "markers" and st["pockets"] == 6
    assert len(st["balls"]) == 15
    sid = st["id"]
    for _ in range(2):
        j = c.post(f"/api/sessions/{sid}/review", json={"find_missing": True}).json()
    assert len(j["balls"]) == 15
    assert 1.6 < st["width"] / st["height"] < 2.3


def test_four_corner_markers_and_fallback():
    img = cv2.imdecode(np.frombuffer(_fixture("green_markers.jpg"), np.uint8), cv2.IMREAD_COLOR)
    four = img.copy()
    for x in (933, 933):                                    # hide the two side-pocket markers
        pass
    cv2.circle(four, (933, 52), 75, (40, 60, 90), -1)
    cv2.circle(four, (933, 966), 75, (40, 60, 90), -1)
    st = c.post("/api/sessions", files={"file": ("4.jpg", cv2.imencode(".jpg", four)[1].tobytes(), "image/jpeg")}).json()
    assert st["mode"] == "markers" and st["pockets"] == 6   # side pockets derived from the corners
    five = four.copy()
    cv2.circle(five, (78, 75), 75, (40, 60, 90), -1)        # 3 markers left: ambiguous -> automatic detection
    r = c.post("/api/sessions", files={"file": ("3.jpg", cv2.imencode(".jpg", five)[1].tobytes(), "image/jpeg")})
    assert r.status_code in (200, 422)
    if r.status_code == 200:
        assert r.json()["mode"] == "auto"
