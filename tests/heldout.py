#!/usr/bin/env python
"""
Score the detector on photos it was not tuned on: the image-data folder of
https://github.com/nicolashahn/set-solver (not copied into this repo).

  git clone --depth 1 https://github.com/nicolashahn/set-solver /tmp/set-solver
  python tests/heldout.py /tmp/set-solver/image-data [--sheets DIR]

Three kinds of check:
  * set-games/        board photos, hand-labelled below (compared as multisets)
  * all-cards/labeled one tightly cropped card per file, labelled by file name;
                      the whole image is read as the card (no detection)
  * all-cards/raw     many cards per photo, some touching; no labels, but a
                      card can only appear once per photo, so repeated
                      readings are errors
"""
import argparse
import collections
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import setvision as sv  # noqa: E402

SAME_TWELVE = """1 striped green diamond, 2 striped green diamond, 1 solid purple diamond, 3 solid purple oval,
    3 open red oval, 1 open red oval, 2 striped purple diamond, 2 striped red squiggle,
    1 striped purple oval, 2 open purple oval, 3 solid red squiggle, 2 striped red oval"""
TEN = """1 striped purple squiggle, 1 open green squiggle, 2 open red squiggle, 1 striped green diamond,
    %s, 1 striped red diamond, 3 striped red oval, 3 solid purple diamond, 3 striped green oval,
    2 open green squiggle, 2 striped green diamond, 2 striped green oval"""
GAMES = {
    "setgame0.jpg": SAME_TWELVE, "setgame4.jpg": SAME_TWELVE, "setgame5.jpg": SAME_TWELVE, "setgame6.jpg": SAME_TWELVE,
    "setgame7.jpg": """1 solid green oval, 2 solid red diamond, 3 solid purple diamond, 3 open purple diamond,
        1 open purple oval, 3 open purple oval, 3 solid red squiggle, 2 open green diamond,
        2 striped red diamond, 3 solid purple oval, 1 striped green oval, 3 striped red squiggle""",
    "setgame8.jpg": """3 striped red squiggle, 2 open green diamond, 1 open red oval, 2 solid purple oval,
        1 striped green oval, 3 striped purple oval, 2 solid purple diamond, 1 solid red oval, 3 open red diamond,
        3 open purple oval, 1 open purple oval, 1 open red squiggle""",
    "setgame10.jpg": TEN % "1 solid purple squiggle",
    "setgame11.jpg": TEN % "2 solid red oval",
    "setgame12.jpg": """2 solid green oval, 1 solid purple diamond, 3 solid green squiggle, 3 solid red diamond,
        1 solid green diamond, 3 solid green diamond, 3 striped red diamond, 1 open green diamond,
        2 striped green squiggle, 3 open green squiggle, 2 striped red squiggle, 2 striped red oval,
        1 striped red squiggle, 3 striped purple squiggle, 1 solid green squiggle""",
    "setgame13.jpg": """2 open red oval, 2 open purple diamond, 3 open purple squiggle, 3 open purple oval,
        3 striped green squiggle, 3 solid red oval, 2 solid purple diamond, 2 open green diamond, 3 open red squiggle,
        1 solid red oval, 3 open red diamond, 3 striped purple diamond, 1 open purple oval, 1 open red squiggle,
        2 solid green squiggle""",
}
WORDS = {"single": 1, "double": 2, "triple": 3, "outline": "open", "stripes": "striped", "solid": "solid",
         "capsule": "oval"}


def describe(c):
    return "%d %s %s %s" % (c["number"], c["shading"], c["color"], c["shape"])


def feats(c):
    return " ".join("%s=%.2f" % (k, v) for k, v in c["features"].items())


def sheet(cards, path):
    tiles = []
    for c in cards:
        t = np.full((122, 150, 3), 255, np.uint8)
        t[22:] = cv2.resize(np.clip(c["_debug"]["card"], 0, 255).astype(np.uint8), (150, 100))
        cv2.putText(t, describe(c)[:22], (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(t)
    if not tiles:
        return
    cols = 8
    while len(tiles) % cols:
        tiles.append(np.full((122, 150, 3), 255, np.uint8))
    rows = [np.concatenate(tiles[i:i + cols], axis=1) for i in range(0, len(tiles), cols)]
    cv2.imwrite(path, cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_data")
    ap.add_argument("--sheets", help="directory for contact sheets of what was read")
    ap.add_argument("--only", choices=["games", "labeled", "raw"])
    args = ap.parse_args()
    D = args.image_data
    if args.sheets:
        os.makedirs(args.sheets, exist_ok=True)

    if args.only in (None, "games"):
        total = right = spurious = 0
        for name, text in GAMES.items():
            truth = [t.strip() for t in text.replace("\n", " ").split(",")]
            cards = sv.detect_cards(sv.load_image(os.path.join(D, "set-games", name)))
            if args.sheets:
                sheet(cards, os.path.join(args.sheets, name + ".png"))
            pool = list(truth)
            wrong = []
            for c in cards:
                if describe(c) in pool:
                    pool.remove(describe(c))
                else:
                    wrong.append("%s [%s]" % (describe(c), feats(c)))
            total += len(truth)
            right += len(truth) - len(pool)
            spurious += max(0, len(cards) - len(truth))
            print("%-14s %2d / %2d%s" % (name, len(truth) - len(pool), len(truth),
                                         "" if not (pool or wrong) else "\n     not found: %s\n     instead:   %s" % (pool, wrong)))
        print("games: %d / %d cards read correctly, %d spurious detections\n" % (right, total, spurious))

    if args.only in (None, "labeled"):
        total = right = 0
        for p in sorted(glob.glob(os.path.join(D, "all-cards", "labeled", "*.jpg"))):
            color, num, shade, shape = os.path.basename(p)[:-4].split("-")
            truth = "%d %s %s %s" % (WORDS[num], WORDS[shade], color, WORDS.get(shape, shape))
            rgb = sv.shrink(sv.load_image(p))
            h, w = rgb.shape[:2]
            card = sv.read_card(rgb, np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64))
            total += 1
            if card is not None and describe(card) == truth:
                right += 1
            else:
                print("  %-40s read as %s" % (truth, "nothing" if card is None else "%s [%s]" % (describe(card), feats(card))))
        print("labeled: %d / %d\n" % (right, total))

    if args.only in (None, "raw"):
        found = repeats = 0
        for p in sorted(glob.glob(os.path.join(D, "all-cards", "raw", "*.jpg"))):
            if "copy" in p:
                continue
            cards = sv.detect_cards(sv.load_image(p))
            if args.sheets:
                sheet(cards, os.path.join(args.sheets, "raw_" + os.path.basename(p) + ".png"))
            seen = collections.Counter(describe(c) for c in cards)
            dup = sum(v - 1 for v in seen.values())
            found += len(cards)
            repeats += dup
            print("%-24s %2d cards read, %d repeated %s" % (os.path.basename(p), len(cards), dup,
                                                           [k for k, v in seen.items() if v > 1] or ""))
        print("raw: %d cards read, %d repeats (each repeat is at least one misreading)" % (found, repeats))


if __name__ == "__main__":
    main()
