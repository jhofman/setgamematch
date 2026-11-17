#!/usr/bin/env python
"""
Score the detector against hand labels.

  python tests/evaluate.py            # all images
  python tests/evaluate.py --debug D  # also write per-image montages of the warped cards to D

Board photos are labelled in tests/labels.json (cards in reading order).
The single-card images in set_training_images/ are labelled by file name.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import setvision as sv  # noqa: E402

WORDS = {"one": 1, "two": 2, "three": 3, "empty": "open", "shaded": "striped", "full": "solid"}


def label_from_filename(name):
    num, shading, color, shape = os.path.basename(name).split("-")[:4]
    return "%d %s %s %s" % (WORDS[num], WORDS[shading], color, shape.rstrip("s"))


def describe(card):
    return "%d %s %s %s" % (card["number"], card["shading"], card["color"], card["shape"])


def montage(cards, path):
    tiles = []
    for c in cards:
        d = c["_debug"]
        a = np.clip(d["card"], 0, 255).astype(np.uint8)
        b = np.clip(d["norm"] * 230, 0, 255).astype(np.uint8)
        s = np.clip(d["sig"] * 255, 0, 255).astype(np.uint8)
        s = np.stack([s, s, s], axis=2)
        for sym in d["symbols"]:
            cv2.rectangle(s, (sym["x"], sym["y"]), (sym["x"] + sym["w"], sym["y"] + sym["h"]), (255, 80, 80), 1)
        f = c["features"]
        txt = "%s h%.0f f%.2f s%.2f e%.2f" % (describe(c), f["hue"], f["fill"], f["solidity"], f["extent"])
        row = np.concatenate([a, b, s], axis=1)
        bar = np.full((22, row.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.concatenate([bar, row], axis=0))
    if tiles:
        cv2.imwrite(path, cv2.cvtColor(np.concatenate(tiles, axis=0), cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", help="directory for debug montages")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.debug:
        os.makedirs(args.debug, exist_ok=True)

    with open(os.path.join(ROOT, "tests", "labels.json")) as f:
        boards = json.load(f)
    items = [(os.path.join(ROOT, k), v) for k, v in boards.items()]
    for p in sorted(glob.glob(os.path.join(ROOT, "set_training_images", "*.jpg"))):
        items.append((p, [label_from_filename(p)]))

    total = correct = missed = extra = 0
    attr_err = dict(number=0, shading=0, color=0, shape=0)
    for path, truth in items:
        cards = sv.detect_cards(sv.load_image(path))
        name = os.path.relpath(path, ROOT)
        if args.debug:
            montage(cards, os.path.join(args.debug, name.replace("/", "_") + ".png"))
        got = [describe(c) for c in cards]
        total += len(truth)
        if len(got) != len(truth):
            # count how many labels are matched by some detection
            pool = list(got)
            hit = 0
            for t in truth:
                if t in pool:
                    pool.remove(t)
                    hit += 1
            correct += hit
            missed += max(0, len(truth) - len(got))
            extra += max(0, len(got) - len(truth))
            print("COUNT  %-55s expected %d cards, found %d (%d match)" % (name, len(truth), len(got), hit))
            if args.verbose:
                print("        truth:", truth)
                print("        found:", got)
            continue
        for t, g, c in zip(truth, got, cards):
            if t == g:
                correct += 1
            else:
                for key, a, b in zip(["number", "shading", "color", "shape"], t.split(), g.split()):
                    if a != b:
                        attr_err[key] += 1
                f = c["features"]
                print("WRONG  %-55s %-26s got %-26s hue=%.0f fill=%.2f sol=%.2f ext=%.2f" % (
                    name, t, g, f["hue"], f["fill"], f["solidity"], f["extent"]))
    print("\n%d / %d cards correct (%.1f%%); %d missed, %d spurious; attribute errors: %s" % (
        correct, total, 100.0 * correct / total, missed, extra, attr_err))
    return 0 if correct == total and not extra else 1


if __name__ == "__main__":
    sys.exit(main())
