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
    r = c.post("/api/sessions", files={"file": ("f.jpg", flat, "image/jpeg")})
    assert r.status_code == 200 and r.json()["mode"] == "none"        # no table found -> tap it by hand
    sid = r.json()["id"]
    assert c.post(f"/api/sessions/{sid}/approve", json={"group": "open"}).status_code == 409   # needs points first
    assert c.get(f"/api/sessions/{sid}/photo.jpg").status_code == 200


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


POCKETS = [(78, 75), (933, 55), (1790, 75), (1790, 958), (933, 965), (78, 958)]   # game_table.jpg, original pixels
BALLS = {"cue": (762, 497), "ball8": (1426, 529), "ball11": (1090, 312), "ball2": (1370, 352), "ball9": (1374, 633)}


K = 1600 / 1836   # the server keeps a photo at most 1600 px wide; the tap editor works in that space


def _items(pockets=POCKETS, balls=BALLS):
    pockets = [(x * K, y * K) for x, y in pockets]
    balls = {k: (x * K, y * K) for k, (x, y) in balls.items()}
    return [{"x": x, "y": y, "kind": f"pocket{i}"} for i, (x, y) in enumerate(pockets, 1)] + \
           [{"x": x, "y": y, "kind": k} for k, (x, y) in balls.items()]


def _photo_session():
    flat = cv2.imencode(".jpg", cv2.imdecode(np.frombuffer(_fixture("game_table.jpg"), np.uint8), cv2.IMREAD_COLOR))[1].tobytes()
    st = c.post("/api/sessions", files={"file": ("g.jpg", flat, "image/jpeg")}).json()
    return st["id"], st


def test_manual_taps_build_table_and_keep_labels():
    sid, _ = _photo_session()
    r = c.post(f"/api/sessions/{sid}/manual", json={"items": _items()})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["mode"] == "manual" and j["pockets"] == 6 and len(j["balls"]) == 5
    assert j["report"]["snapped"] == 5                                    # taps snapped to real ball centres
    by = {b["id"]: b["label"] for b in j["balls"]}
    assert by["CUE"] == "cue" and by["#8"] == "eight" and by["#11"] == "stripe" and by["#2"] == "solid"
    # a review round must not relabel the user's choices
    j2 = c.post(f"/api/sessions/{sid}/review", json={"improve": ["#11", "CUE"]}).json()
    assert {b["id"]: b["label"] for b in j2["balls"]} == by
    a = c.post(f"/api/sessions/{sid}/approve", json={"group": "open"}).json()
    assert 1 <= len(a["options"]) <= 3


def test_manual_four_corners_only_and_errors():
    sid, _ = _photo_session()
    corners = [POCKETS[i] for i in (0, 2, 3, 5)]
    r = c.post(f"/api/sessions/{sid}/manual", json={"items": _items(corners)})
    assert r.status_code == 200 and r.json()["pockets"] == 6               # middle pockets derived
    # too few pockets / unlabeled / duplicate number -> clear 422s
    assert c.post(f"/api/sessions/{sid}/manual", json={"items": _items(POCKETS[:3])}).status_code == 422
    bad = _items(); bad[-1]["kind"] = ""
    assert c.post(f"/api/sessions/{sid}/manual", json={"items": bad}).status_code == 422
    dup = _items(); dup[-1]["kind"] = "ball8"
    assert c.post(f"/api/sessions/{sid}/manual", json={"items": dup}).status_code == 422


def test_prefill_round_trip():
    sid = c.post("/api/sessions", files={"file": ("g.jpg", _fixture("game_table.jpg"), "image/jpeg")}).json()["id"]
    items = c.get(f"/api/sessions/{sid}/prefill").json()["items"]
    assert sum(i["kind"].startswith("pocket") for i in items) == 6 and len(items) == 21
    j = c.post(f"/api/sessions/{sid}/manual", json={"items": items}).json()
    assert j["mode"] == "manual" and len(j["balls"]) == 15


def test_shots_are_simulated_and_group_changes_ranking():
    from cueguide import engine as e
    s = e.analyze(open("tests/fixtures/game_table.jpg", "rb").read())
    for g in ("open", "solid", "stripe"):
        opts = e.top_shots(s, g)
        assert opts and any(o["kind"] == "defense" for o in opts)
        for o in opts:
            sim = o["sim"]
            assert sim["moves"][0]["label"] == "cue" and not sim["cue_scratch"]
            assert sim["first"] == o["target"] and o["effects"]
            if o["kind"] == "offense":
                assert any(m["id"] == o["target"] and m["pocketed"] for m in sim["moves"])
            assert e.overlay_png(s, o)[:4] == b"\x89PNG"
    solids = {o["target_label"] for o in e.top_shots(s, "solid") if o["kind"] == "offense"}
    assert solids <= {"solid"}


def test_simulate_cushion_and_friction():
    from cueguide import engine as e
    s = e.analyze(open("tests/fixtures/game_table.jpg", "rb").read())
    cue = next(b for b in s.balls if b.label == "cue")
    soft = e.simulate(s, (1, 0), 3.0)["moves"]
    assert soft[0]["rails"] == 0
    hard = e.simulate(s, (0, -1), 22.0)["moves"][0]
    assert hard["rails"] >= 1 or hard["pocketed"]


def test_random_table_endpoint_and_edit():
    from fastapi.testclient import TestClient
    from cueguide.app import app
    c = TestClient(app)
    j = c.post("/api/random", json={"balls": 10, "seed": 4}).json()
    assert j["random"] and j["mode"] == "manual" and len(j["balls"]) == 10 and len(j["items"]) == 16
    items = [i for i in j["items"] if i["kind"] != "ball8"]          # remove the 8, then rebuild
    r = c.post(f"/api/sessions/{j['id']}/manual", json={"items": items})
    assert r.status_code == 200 and len(r.json()["balls"]) == 9
    a = c.post(f"/api/sessions/{j['id']}/approve", json={"group": "open"}).json()
    assert a["options"]
