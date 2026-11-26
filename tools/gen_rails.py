"""Builds cueguide/data/rail_shots.json: stored rail-shot examples, computed once (no lookups at runtime).
Table units are diamonds: x 0..8 along the long rail, y 0..4 across; corner pockets at (0,0) (8,0) (0,4) (8,4),
side pockets at (4,0) and (4,4). Every geometric example is an exact equal-angle (no-english) reflection path."""
import json, math
from pathlib import Path

W, H = 8.0, 4.0
SRC = {
    "dd": ("Dr. Dave: diamond systems for kicks", "https://drdavepoolinfo.com/faq/bank-kick/diamond-systems/"),
    "c5": ("Dr. Dave: Corner-5 system", "https://drdavepoolinfo.com/faq/bank-kick/corner-5/"),
    "lg": ("Liberty Games: pool diamond system", "https://www.libertygames.co.uk/pool-diamond-system/"),
    "pd": ("PoolDawg: how to make kick and bank shots", "https://www.pooldawg.com/article/how-to-make-kick-bank-shots/"),
}


def refl(p, rail):
    x, y = p
    return {"T": (x, -y), "B": (x, 2 * H - y), "L": (-x, y), "R": (2 * W - x, y)}[rail]


def trace(C, T, rails):
    img = T
    for r in reversed(rails):
        img = refl(img, r)
    d = (img[0] - C[0], img[1] - C[1])
    pts, p = [C], C
    for r in rails:
        if r in "TB":
            ry = 0 if r == "T" else H
            if abs(d[1]) < 1e-9: raise ValueError("parallel")
            t = (ry - p[1]) / d[1]
        else:
            rx = 0 if r == "L" else W
            if abs(d[0]) < 1e-9: raise ValueError("parallel")
            t = (rx - p[0]) / d[0]
        if t <= 1e-9: raise ValueError(f"rail {r} behind")
        q = (p[0] + d[0] * t, p[1] + d[1] * t)
        if not (-1e-6 <= q[0] <= W + 1e-6 and -1e-6 <= q[1] <= H + 1e-6): raise ValueError(f"contact off table {q}")
        pts.append(q)
        d = (d[0], -d[1]) if r in "TB" else (-d[0], d[1])
        p = q
    pts.append(T)
    # the last leg must actually end at the target: heading check
    last = (T[0] - pts[-2][0], T[1] - pts[-2][1])
    if last[0] * d[0] + last[1] * d[1] <= 0: raise ValueError("target behind last rail")
    return pts


def label(q):
    q = (round(q[0] * 4) / 4, round(q[1] * 4) / 4)
    if abs(q[1]) < 1e-6 or abs(q[1] - H) < 1e-6:
        return f"{q[0]:g}"
    return f"{q[1]:g}"


def shot(id, title, cat, summary, how, tips, C, T, rails, pocket=None, src=("pd",), kind="kick"):
    pts = trace(C, T, rails)
    contacts = [{"x": round(x, 3), "y": round(y, 3), "label": label((x, y)), "rail": r} for (x, y), r in zip(pts[1:-1], rails)]
    return {"id": id, "title": title, "category": cat, "summary": summary, "how": how, "tips": tips,
            "table": {"kind": kind, "cue": list(C) if kind == "kick" else None,
                      "object": list(T) if kind == "kick" else list(C), "pocket": list(pocket) if pocket else None,
                      "path": [[round(x, 3), round(y, 3)] for x, y in pts], "contacts": contacts},
            "sources": list(src)}


def text(id, title, cat, summary, how, tips, src, spin=None):
    d = {"id": id, "title": title, "category": cat, "summary": summary, "how": how, "tips": tips, "table": None, "sources": list(src)}
    if spin: d["spin"] = spin
    return d


L = []
L.append(shot("kick-1rail-equal", "One-rail kick: equal distances (the mirror)", "kick",
    "Cue ball and object ball are the same distance from the rail you kick off. Aim at the diamond halfway between them.",
    ["Count the diamonds along the rail between where the two balls sit (their straight-down projections).", "Halve the count: that is the aim diamond on the rail.", "Aim where the diamond's line meets the cushion edge, not at the diamond itself."],
    ["Medium speed is the baseline for the diamond systems.", "A hard stroke shortens the rebound; a soft stroke lets it run long."],
    (1, 3), (5, 3), "B", src=("dd", "lg", "pd")))
L.append(shot("kick-1rail-unequal", "One-rail kick: unequal distances", "kick",
    "When the object ball is twice as far from the rail as the cue ball, the contact point is one third of the way along, measured from the cue ball.",
    ["Compare the two balls' distances to the rail (in diamonds).", "Contact divides the gap between them in the same ratio as those distances: nearer ball = shorter share.", "Here 1 diamond vs 2 diamonds: contact is 1/3 of the way from the cue ball's side."],
    ["If you can't judge a ratio, picture the mirror image of the object ball behind the rail and aim straight at it."],
    (2, 3), (5, 2), "B", src=("dd", "pd")))
L.append(shot("kick-end-rail", "One-rail kick off the end rail", "kick",
    "Same mirror rule on the short rail: count the diamonds up and down the end rail (3 on the short rail).",
    ["Project both balls onto the end rail.", "Aim at the point that divides the gap in the ratio of their distances from the rail.", "Short-rail diamonds are numbered 1 to 3 from the top."],
    ["Expect to adjust: end-rail cushions often play a little different from the side rails."],
    (3, 1), (1, 3), "L", src=("dd", "lg")))
L.append(shot("kick-near-rail", "Kicking to a ball that hugs the rail", "kick",
    "When the object ball is close to the rail, the contact point lands very near it. Aim just short of the ball.",
    ["Use the mirror: the object ball's image sits just behind the rail, so the line meets the cushion almost at the ball.", "Keep the first-rail contact clear of the object ball so you don't hit it on the rail."],
    ["Shallow angles are sensitive to speed: stroke firmly and smoothly."], (2, 3), (7, 0.5), "T", src=("dd", "pd")))
L.append(shot("kick-2rail-long-end", "Two-rail kick: long rail, then end rail", "kick",
    "Use two rails to get around a blocking ball. Unfold the table: aim at the object ball's mirror image across both rails.",
    ["Mirror the object ball across the end rail, then across the long rail.", "Shoot at that double image; the line crosses the long rail at the first contact point.", "Check the numbers on the diagram, then adjust a little with speed and spin."],
    ["Running english (spin toward the direction the ball is travelling along the rail) lengthens the rebound; reverse english shortens it.", "Roughly a diamond per tip of spin, but test your table."],
    (2, 1.5), (7, 1), ["B", "R"], src=("dd", "lg", "pd")))
L.append(shot("kick-2rail-end-long", "Two-rail kick: end rail, then long rail", "kick",
    "The other order: hit the short rail first, then a long rail. The unfolded line shows both contact diamonds.",
    ["Mirror the object ball across the long rail, then across the end rail.", "Aim at the double image.", "Compare the two contact diamonds with the 'Plus' and mirror rules before you shoot."],
    ["Soft strokes open the angle and go long; hard strokes tighten it."],
    (5, 3), (2, 1), ["R", "T"], src=("dd", "lg")))
L.append(shot("kick-3rail", "Three-rail kick around the table", "kick",
    "Long rail, end rail, long rail. The longest standard kick; use it when the whole near side is blocked.",
    ["Unfold three times: mirror the object ball across the last, middle and first rail in turn.", "Shoot at the final image and read the first contact diamond from the diagram.", "Corner-5 style systems turn this into a diamond calculation (see the systems below)."],
    ["Test a three-rail kick to the third diamond to learn whether your table plays long or short."],
    (1, 1), (3, 3), ["B", "R", "T"], src=("dd", "c5", "pd")))
L.append(shot("bank-cross-corner", "Cross-corner bank off a long rail", "bank",
    "A ball in the far half banked off the long rail into the corner pocket on the other side.",
    ["Mirror the pocket across the rail the ball will hit.", "The object ball's line to that mirror pocket crosses the rail at the bank diamond.", "Angle in equals angle out: keep the cue ball out of the object ball's path."],
    ["Banks are speed sensitive: medium-firm, no side spin to start with.", "Hit the ball with a slightly softer stroke than a pot of the same distance."],
    (3, 1), (8, 0), ["B"], pocket=(8, 0), src=("dd", "lg"), kind="bank"))
L.append(shot("bank-side-pocket", "Bank into the side pocket", "bank",
    "From near the long rail, bank off the opposite long rail into the side pocket.",
    ["Mirror the side pocket across the opposite rail.", "Line the object ball up with the mirror pocket.", "The diagram shows the contact diamond."],
    ["Aim for the middle of the side-pocket mouth; the pocket is forgiving on a good bank."], (2, 1.2), (4, 0), ["B"], pocket=(4, 0), src=("dd", "lg"), kind="bank"))
L.append(shot("bank-long-end", "Long bank off the end rail", "bank",
    "Banking off the short rail toward the far corner: a very shallow angle, so contact is close to where the ball is.",
    ["Mirror the far corner across the end rail.", "Shoot the object ball along that line.", "Short-rail contact is read in diamonds 1 to 3 from the top."],
    ["Shallow banks run long on slow cloth and tight on fast cloth."], (3, 2.5), (8, 4), ["L"], pocket=(8, 4), src=("dd",), kind="bank"))
L.append(shot("bank-2rail", "Two-rail bank into the corner", "bank",
    "A ball banked off the long rail then the end rail and into the far corner. Unfold with two mirrors.",
    ["Mirror the pocket across the end rail, then across the long rail.", "The object ball's line to the double mirror pocket gives both contact points.", "Use stun, no english, at first."],
    ["Add a touch of running english to keep the ball from dying on the second rail."], (6.5, 1.5), (8, 0), ["B", "L"], pocket=(8, 0), src=("dd",), kind="bank"))

L.append(text("sys-mirror", "System: halve the count (mirror)", "system",
    "The simplest diamond system for one-rail kicks. Count the diamonds from the cue ball's side to the object ball's side, halve it, and aim at that diamond.",
    ["Example (PoolDawg): 4 diamonds apart means aim at the 2nd diamond; 7 apart means aim at the 3.5 point.", "If the balls aren't the same distance from the rail, project your aim so you still form the same mirrored triangle."],
    ["Aim where the diamond's line meets the cushion, not at the marking.", "Medium speed, no side spin."], ("pd", "lg", "dd")))
L.append(text("sys-plus2", "System: Plus Two (two-rail kicks)", "system",
    "A diamond calculation for two-rail kicks. Per the Liberty Games guide: whatever diamond you aim for, the cue ball's start position shifts back two places to give the contact.",
    ["Example (Liberty Games): aim at the second diamond from the right on the bottom rail; the cue ball contacts two positions back from its start, sending the yellow into the top-left corner.", "Numbering conventions differ between sources; learn one convention and test it on your table."],
    ["Plus System is recommended by Dr. Dave for two- and three-rail kicks."], ("lg", "dd")))
L.append(text("sys-corner5", "System: Corner-5 (three-rail kicks)", "system",
    "The best-known system for three-rail kicks off a long rail. Number the cue ball's start and your aim diamond, then subtract.",
    ["Liberty Games' example: a cue ball starting over the corner pocket counts as 5; aim at diamond 2 on the bottom rail; 5 minus 2 is 3, so the ball meets the opposite rail at diamond 3.", "Full numbering rules and worked benchmarks are on Dr. Dave's Corner-5 pages."],
    ["Corner-5 assumes no side spin and a standard 9 ft table.", "Dr. Dave notes it also covers two-, three- and four-rail paths off a long rail."], ("c5", "lg", "dd")))
L.append(text("sys-sixes", "System: the Sixes (four-rail targets)", "system",
    "Patrick Johnson's diamond-to-diamond parallel lines: the separation value and the fourth-rail target always add up to 6.",
    ["Separation 6 diamonds: target the corner pocket. 5: target the 1st diamond. 4: the 2nd. 3: the 3rd.", "That is why it is called the Sixes system."],
    ["It uses visual diamond alignments rather than the arithmetic of Corner-5."], ("dd",)))
L.append(text("sys-spin-speed", "Spin and speed: tuning the angle", "spin",
    "Spin and speed change where a rail shot goes, even when the aim is perfect.",
    ["Running english (toward the direction the ball travels along the rail) opens the angle and runs long; reverse english tightens it.", "Rule of thumb from PoolDawg: one tip of spin moves the path by about one diamond; two tips about two. Extreme english adds little.", "Harder strokes keep the ball straighter and shorten the rebound; softer strokes open the angle and go long."],
    ["Use running english on every multi-rail shot so the ball keeps moving after the first rail.", "Pick one speed for your practice and keep it."], ("pd",)))
L.append(text("sys-table-test", "Test your table first", "spin",
    "Tables play long or short. A 30-second test makes every system more accurate.",
    ["Shoot a three-rail kick to the third diamond and see whether the ball arrives long or short.", "If your table plays long, aim a little before the diamond you would normally use.", "Practice drill: make consecutive bank shots, starting with 5 in a row and building to 10 or more."],
    ["Cloth speed, humidity and cushion age all move the numbers."], ("pd",)))

out = {"table": {"w": W, "h": H, "note": "units are diamonds; x along the long rail"}, "sources": {k: {"title": t, "url": u} for k, (t, u) in SRC.items()}, "shots": L}
Path(__file__).resolve().parent.parent.joinpath("cueguide/data/rail_shots.json").write_text(json.dumps(out, indent=1))
for s in L:
    t = s["table"]
    print(f"{s['id']:22s} {s['category']:6s}", "" if not t else [c["label"] for c in t["contacts"]])
print(len(L), "entries")
