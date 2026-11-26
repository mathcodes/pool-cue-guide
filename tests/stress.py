"""Random-table stress test: python tests/stress.py [n_seeds]. Every layout must rank, play, undo without errors."""
import sys, json
sys.path.insert(0, ".")
from cueguide import engine as e, randomtable as rt
bad = 0
for seed in range(int(sys.argv[1]) if len(sys.argv) > 1 else 40):
    jpg, items = rt.make(seed)
    s = e.photo_session(e.decode_image(jpg), "")
    e.build_manual(s, items)
    for g in ("open", "solid", "stripe"):
        try:
            t = e.top_shots(s, g)
            json.dumps(t, default=float)
            if t:
                e.play(s, t[0]); e.top_shots(s, g); e.undo(s)
        except Exception as ex:
            bad += 1
            print("FAIL", seed, g, type(ex).__name__, ex)
print("bad:", bad)
sys.exit(1 if bad else 0)
