#!/usr/bin/env python
"""
Check that the JavaScript port (docs/setvision.js) agrees with setvision.py.

Every test image is shrunk in Python, handed to both implementations as raw
pixels, and the results are compared card by card. Needs `node` on the PATH.

  python tests/js_parity.py [more photos ...]
"""
import glob
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import setvision as sv  # noqa: E402
import evaluate as ev  # noqa: E402
import stress  # noqa: E402


def run_js(rgb):
    h, w = rgb.shape[:2]
    rgba = np.dstack([rgb, np.full((h, w), 255, np.uint8)])
    with tempfile.NamedTemporaryFile(suffix=".rgba", delete=False) as f:
        f.write(rgba.tobytes())
    try:
        out = subprocess.check_output(["node", os.path.join(ROOT, "tests", "run_js.js"), f.name, str(w), str(h)])
    finally:
        os.unlink(f.name)
    return json.loads(out)


def main():
    with open(os.path.join(ROOT, "tests", "labels.json")) as f:
        boards = json.load(f)
    items = [(k, v, None) for k, v in boards.items()]
    for p in sorted(glob.glob(os.path.join(ROOT, "set_training_images", "*.jpg"))):
        items.append((os.path.relpath(p, ROOT), [ev.label_from_filename(p)], None))
    # a few degraded boards too, where the two are most likely to drift apart
    for t in ["rotate 17", "blur 3", "noise sd12", "480 px", "perspective", "light gradient"]:
        items += [(k, v, t) for k, v in boards.items()]
    # any extra photos named on the command line are compared too, at full
    # resolution (no labels needed)
    items += [(os.path.abspath(p), None, None) for p in sys.argv[1:]]

    cards_total = disagreements = js_wrong = 0
    worst = dict(hue=0.0, fill=0.0, solidity=0.0, extent=0.0, stripe=0.0)
    slowest = 0
    for name, truth, t in items:
        rgb = sv.load_image(os.path.join(ROOT, name))
        if truth is not None:
            rgb = sv.shrink(rgb)      # extra photos go in at full size, to exercise the JS resize
        if t:
            rgb = sv.shrink(np.ascontiguousarray(stress.TRANSFORMS[t](rgb)))
        py = sv.detect_cards(rgb)
        res = run_js(rgb)
        js = res["cards"]
        slowest = max(slowest, res["ms"])
        tag = "%s%s" % (name, " [%s]" % t if t else "")
        cards_total += len(py)
        if len(py) != len(js):
            disagreements += abs(len(py) - len(js))
            print("COUNT  %-60s python %d cards, js %d" % (tag, len(py), len(js)))
            continue
        for a, b in zip(py, js):
            if ev.describe(a) != ev.describe(b):
                disagreements += 1
                print("DIFF   %-60s python %-26s js %-26s" % (tag, ev.describe(a), ev.describe(b)))
            for k in worst:
                d = abs(a["features"][k] - b["features"][k])
                if k == "hue":
                    d = min(d, 360 - d)
                worst[k] = max(worst[k], d)
        if truth and not t:
            js_wrong += sum(1 for x, b in zip(truth, js) if x != ev.describe(b))
    print("\n%d cards compared, %d disagreements; js wrong on %d labelled cards; slowest image %d ms in node" % (
        cards_total, disagreements, js_wrong, slowest))
    print("largest feature differences:", {k: round(v, 4) for k, v in worst.items()})
    return 1 if disagreements or js_wrong else 0


if __name__ == "__main__":
    sys.exit(main())
