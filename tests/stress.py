#!/usr/bin/env python
"""
Re-run the labelled board photos through synthetic degradations (rotation,
dim light, colour casts, blur, low resolution, heavy JPEG, noise) to see how
much margin the detector has.

  python tests/stress.py
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import setvision as sv  # noqa: E402


def rotate(deg):
    def f(rgb):
        h, w = rgb.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        c, s = abs(M[0, 0]), abs(M[0, 1])
        W, H = int(w * c + h * s), int(h * c + w * s)
        M[0, 2] += W / 2 - w / 2
        M[1, 2] += H / 2 - h / 2
        return cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(40, 35, 30))
    return f


def gain(r, g, b):
    return lambda rgb: np.clip(rgb.astype(np.float32) * [r, g, b], 0, 255).astype(np.uint8)


def blur(sigma):
    return lambda rgb: cv2.GaussianBlur(rgb, (0, 0), sigma)


def small(side):
    return lambda rgb: sv.shrink(rgb, side)


def jpeg(q):
    def f(rgb):
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return f


def noise(sd):
    def f(rgb):
        rng = np.random.RandomState(0)
        return np.clip(rgb + rng.normal(0, sd, rgb.shape), 0, 255).astype(np.uint8)
    return f


def gradient(rgb):
    """Strong left-to-right light falloff."""
    h, w = rgb.shape[:2]
    ramp = np.linspace(1.0, 0.45, w, dtype=np.float32)[None, :, None]
    return np.clip(rgb * ramp, 0, 255).astype(np.uint8)


def perspective(rgb):
    h, w = rgb.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0.22 * w, 0.1 * h], [0.78 * w, 0.1 * h], [w, h], [0, h]])
    return cv2.warpPerspective(rgb, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(40, 35, 30))


TRANSFORMS = {
    "rotate 90": rotate(90), "rotate 17": rotate(17), "rotate 45": rotate(45), "rotate 180": rotate(180),
    "dim x0.5": gain(.5, .5, .5), "dim x0.3": gain(.3, .3, .3),
    # (overexposure is not here: clipping these small test photos erases the
    # faint tint that is all that is left of the stripes at this resolution)
    "warm cast": gain(1.0, 0.82, 0.6), "cool cast": gain(0.7, 0.85, 1.0), "green cast": gain(0.8, 1.0, 0.75),
    "blur 1.5": blur(1.5), "blur 3": blur(3),
    "1000 px": small(1000), "640 px": small(640), "480 px": small(480),
    "jpeg q25": jpeg(25), "noise sd12": noise(12), "light gradient": gradient, "perspective": perspective,
}


def main():
    with open(os.path.join(ROOT, "tests", "labels.json")) as f:
        boards = json.load(f)
    images = {k: sv.shrink(sv.load_image(os.path.join(ROOT, k))) for k in boards}
    bad = 0
    for tname, fn in TRANSFORMS.items():
        total = ok = 0
        notes = []
        for name, truth in boards.items():
            cards = sv.detect_cards(fn(images[name]))
            got = ["%d %s %s %s" % (c["number"], c["shading"], c["color"], c["shape"]) for c in cards]
            # rotations change the reading order, so compare as multisets
            pool = list(got)
            hit = 0
            for t in truth:
                if t in pool:
                    pool.remove(t)
                    hit += 1
            total += len(truth)
            ok += hit
            if hit != len(truth) or pool:
                notes.append("%s: %d/%d%s" % (name, hit, len(truth), (" extra/wrong %s" % pool) if pool else ""))
        bad += total - ok
        print("%-15s %3d / %3d  %s" % (tname, ok, total, "; ".join(notes)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
