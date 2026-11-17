#!/usr/bin/env python
"""Print the range of each feature per class (optionally under one stress transform)."""
import glob, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import setvision as sv, evaluate as ev, stress  # noqa

fn = stress.TRANSFORMS[sys.argv[1]] if len(sys.argv) > 1 else (lambda x: x)
items = list(json.load(open(os.path.join(ROOT, "tests/labels.json"))).items())
if len(sys.argv) <= 1:
    items += [(p, [ev.label_from_filename(p)]) for p in sorted(glob.glob(os.path.join(ROOT, "set_training_images/*.jpg")))]
rows = []
for p, truth in items:
    cards = sv.detect_cards(fn(sv.shrink(sv.load_image(os.path.join(ROOT, p)))))
    if len(cards) != len(truth):
        print("skip (count)", p); continue
    # match by multiset is not possible feature-wise; assume reading order
    for t, c in zip(truth, cards):
        n, sh, col, shape = t.split()
        rows.append((sh, col, shape, c["features"], os.path.basename(p), ev.describe(c) == t))
for key, classes, idx in [("fill", sv.SHADINGS, 0), ("solidity", sv.SHAPES, 2), ("extent", sv.SHAPES, 2)]:
    print(key)
    for k in classes:
        v = sorted((r[3][key], r[4]) for r in rows if r[idx] == k)
        if v:
            print("  %-9s n=%2d  %s ... %s" % (k, len(v), " ".join("%.3f" % x[0] for x in v[:4]), " ".join("%.3f" % x[0] for x in v[-4:])))
print("hue")
for k in sv.COLORS:
    print("  %-7s" % k, sorted(round(r[3]["hue"]) for r in rows if r[1] == k))
