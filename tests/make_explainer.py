#!/usr/bin/env python
"""
Regenerate the figures and data behind docs/how.html (the "How it works" page)
by running the real pipeline on board.jpg and the labelled photos.

  python tests/make_explainer.py [HELDOUT_IMAGE_DATA]

Writes docs/how/*.jpg|png and docs/how/data.json. If the image-data folder of
https://github.com/nicolashahn/set-solver is given, the labelled board photos
from it are added to the charts (as numbers only; no images are copied).
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import setvision as sv  # noqa: E402
import evaluate as ev  # noqa: E402

OUT = os.path.join(ROOT, "docs", "how")


def save(name, rgb, quality=82):
    path = os.path.join(OUT, name)
    img = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if name.endswith(".jpg"):
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def grey(a):
    """0..1 float -> RGB uint8."""
    g = np.clip(a * 255, 0, 255).astype(np.uint8)
    return np.dstack([g, g, g])


def describe(c):
    return "%d %s %s %s" % (c["number"], c["shading"], c["color"], c["shape"])


def card_figures(rgb, cards):
    """Step 1: finding the cards on the board."""
    small = sv.shrink(rgb, 1000)
    k = small.shape[1] / float(rgb.shape[1])
    save("board.jpg", small)

    white = small.min(axis=2)
    save("whiteness.jpg", grey(white / 255.0))
    t = sv.otsu(white)
    save("threshold.png", grey((white > t).astype(float)))

    out = small.copy()
    for c in cards:
        q = np.int32(np.array(c["quad"]) * k)
        cv2.polylines(out, [q], True, (47, 129, 247), 4, cv2.LINE_AA)
        for p in q:
            cv2.circle(out, tuple(int(v) for v in p), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, tuple(int(v) for v in p), 6, (47, 129, 247), 2, cv2.LINE_AA)
    save("outlines.jpg", out)


def card_steps(card, prefix):
    """Steps 2-4 for one card: warped, normalised, ink, symbols."""
    d = card["_debug"]
    save(prefix + "-warped.jpg", d["card"], 90)
    save(prefix + "-normalised.jpg", np.clip(d["norm"], 0, 1.05) / 1.05 * 255, 90)
    save(prefix + "-ink.png", grey(np.clip(d["sig"] * 1.6, 0, 1)))
    filled = np.zeros(d["sig"].shape, np.uint8)
    for s in d["symbols"]:
        filled[s["y"]:s["y"] + s["h"], s["x"]:s["x"] + s["w"]][s["mask"]] = 255
    out = np.dstack([filled] * 3).copy()
    for s in d["symbols"]:
        cv2.rectangle(out, (s["x"] - 1, s["y"] - 1), (s["x"] + s["w"], s["y"] + s["h"]), (47, 129, 247), 1)
    save(prefix + "-symbols.png", out)


def symbol_crop(card, i=0, pad=10):
    """One symbol of a warped card, with its interior ('core') outlined."""
    d = card["_debug"]
    s, f = d["symbols"][i], d["feats"][i]
    y0, y1 = max(s["y"] - pad, 0), min(s["y"] + s["h"] + pad, sv.CARD_H)
    x0, x1 = max(s["x"] - pad, 0), min(s["x"] + s["w"] + pad, sv.CARD_W)
    big = 3
    crop = cv2.resize(np.clip(d["card"][y0:y1, x0:x1], 0, 255).astype(np.uint8), None, fx=big, fy=big,
                      interpolation=cv2.INTER_CUBIC)
    core = np.zeros((y1 - y0, x1 - x0), np.uint8)
    core[s["y"] - y0:s["y"] - y0 + s["h"], s["x"] - x0:s["x"] - x0 + s["w"]] = f["core"].astype(np.uint8)
    core = cv2.resize(core, None, fx=big, fy=big, interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = crop.copy()
    cv2.drawContours(out, cnts, -1, (47, 129, 247), 2, cv2.LINE_AA)
    return out, f


def shape_figure(card):
    """Filled symbol with its bounding box and convex hull."""
    d = card["_debug"]
    s, f = d["symbols"][0], d["feats"][0]
    pad, big = 6, 3
    m = np.zeros((s["h"] + 2 * pad, s["w"] + 2 * pad), np.uint8)
    m[pad:pad + s["h"], pad:pad + s["w"]] = s["mask"].astype(np.uint8)
    m = cv2.resize(m, None, fx=big, fy=big, interpolation=cv2.INTER_NEAREST)
    H, W = m.shape
    out = np.full((H, W, 3), 255, np.uint8)
    out[m > 0] = (190, 196, 204)
    cv2.rectangle(out, (pad * big, pad * big), (W - pad * big - 1, H - pad * big - 1), (101, 109, 118), 2)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    cv2.polylines(out, [hull], True, (47, 129, 247), 3, cv2.LINE_AA)
    return out, f


def collect(path, truth, source):
    """Features of every correctly matched card in one labelled photo."""
    rows = []
    cards = sv.detect_cards(sv.load_image(path))
    if len(cards) != len(truth):
        return rows
    for t, c in zip(truth, cards):
        n, shading, color, shape = t.split()
        f = c["features"]
        rows.append(dict(source=source, truth=t, read=describe(c), number=int(n), color=color, shading=shading,
                         shape=shape, hue=round(f["hue"], 1), fill=round(f["fill"], 3),
                         solidity=round(f["solidity"], 4), extent=round(f["extent"], 4),
                         stripe=round(f["stripe"], 3)))
    return rows


def collect_unordered(path, truth, source):
    """Same, for photos labelled as an unordered list (matched on number, colour and shape)."""
    rows = []
    pool = list(truth)
    for c in sv.detect_cards(sv.load_image(path)):
        key = (c["number"], c["color"], c["shape"])
        match = [t for t in pool if (int(t.split()[0]), t.split()[2], t.split()[3]) == key]
        if len(match) != 1:
            continue
        pool.remove(match[0])
        n, shading, color, shape = match[0].split()
        f = c["features"]
        rows.append(dict(source=source, truth=match[0], read=describe(c), number=int(n), color=color,
                         shading=shading, shape=shape, hue=round(f["hue"], 1), fill=round(f["fill"], 3),
                         solidity=round(f["solidity"], 4), extent=round(f["extent"], 4),
                         stripe=round(f["stripe"], 3)))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    rgb = sv.load_image(os.path.join(ROOT, "board.jpg"))
    cards = sv.detect_cards(rgb)
    by_name = {describe(c): c for c in cards}
    card_figures(rgb, cards)

    # the walk-through card: two striped red ovals
    card_steps(by_name["2 striped red oval"], "card")

    # shading: open, striped, solid diamonds from the same board
    data = dict(shading=[], shapes=[], features=[])
    for name, key in [("1 open purple diamond", "open"), ("1 striped green diamond", "striped"),
                      ("2 solid purple diamond", "solid")]:
        img, f = symbol_crop(by_name[name])
        save("shading-%s.jpg" % key, img, 88)
        data["shading"].append(dict(key=key, card=name, fill=round(f["fill"], 3), stripe=round(f["stripe"], 3),
                                    profile=[round(v, 3) for v in f["profile"]]))

    for name, key in [("1 open red diamond", "diamond"), ("1 open green oval", "oval"),
                      ("1 open green squiggle", "squiggle")]:
        img, f = shape_figure(by_name[name])
        save("shape-%s.png" % key, img)
        data["shapes"].append(dict(key=key, solidity=round(f["solidity"], 3), extent=round(f["extent"], 3)))

    # every labelled card, for the charts
    with open(os.path.join(ROOT, "tests", "labels.json")) as fh:
        boards = json.load(fh)
    for name, truth in boards.items():
        data["features"] += collect(os.path.join(ROOT, name), truth, "repo")
    for p in sorted(glob.glob(os.path.join(ROOT, "set_training_images", "*.jpg"))):
        data["features"] += collect(p, [ev.label_from_filename(p)], "repo")
    if len(sys.argv) > 1:
        import heldout
        for name, text in heldout.GAMES.items():
            truth = [t.strip() for t in text.replace("\n", " ").split(",")]
            data["features"] += collect_unordered(os.path.join(sys.argv[1], "set-games", name), truth, "other")

    data["thresholds"] = dict(
        hue=dict(red_below=55, green_below=195, red_from=330),
        fill=dict(open_below=0.10, faint_below=0.20, faint_stripe_below=0.36, solid_from=0.84),
        shapes={k: dict(extent=v[0], solidity=v[1]) for k, v in sv.SHAPE_PROTOTYPES.items()},
        pitch=sv.PITCH)
    with open(os.path.join(OUT, "data.json"), "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    print("%d cards in the charts (%d from other photos)" % (
        len(data["features"]), sum(r["source"] == "other" for r in data["features"])))


if __name__ == "__main__":
    main()
