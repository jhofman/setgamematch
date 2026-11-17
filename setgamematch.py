#!/usr/bin/env python
"""
Find the sets in a photo of a Set board.

  python setgamematch.py board.jpg

prints the cards and the sets, and writes board_labeled.jpg.
"""
import os
import sys

import cv2
import numpy as np

import setvision as sv

# BGR colours used to outline the cards of each set
SET_COLORS = [(66, 133, 244), (60, 180, 75), (200, 80, 200), (0, 200, 230),
              (230, 150, 30), (120, 120, 255), (160, 200, 60), (90, 90, 90)]


def annotate(rgb, cards, sets):
    im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    unit = max(im.shape[:2]) / 1000.0            # scale line widths with the image
    for card in cards:
        q = np.int32(card["quad"])
        cv2.polylines(im, [q], True, (160, 160, 160), max(1, int(2 * unit)))
        x, y = q.min(axis=0)
        cv2.putText(im, sv.card_name(card), (int(x), int(y - 6 * unit)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45 * unit, (255, 255, 255), max(1, int(3 * unit)), cv2.LINE_AA)
        cv2.putText(im, sv.card_name(card), (int(x), int(y - 6 * unit)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45 * unit, (0, 0, 0), max(1, int(1 * unit)), cv2.LINE_AA)
    # each set gets its own colour; cards in several sets get nested outlines
    depth = [0] * len(cards)
    for n, triple in enumerate(sets):
        colour = SET_COLORS[n % len(SET_COLORS)]
        for i in triple:
            q = np.array(cards[i]["quad"])
            c = q.mean(axis=0)
            shrink = 1.0 - 0.07 * depth[i]
            depth[i] += 1
            cv2.polylines(im, [np.int32(c + (q - c) * shrink)], True, colour, max(2, int(5 * unit)))
    return im


def main(argv):
    if len(argv) != 2:
        sys.stderr.write("usage: %s <image>\n" % argv[0])
        return 1
    infile = argv[1]
    try:
        rgb = sv.load_image(infile)
    except IOError as e:
        sys.stderr.write("error: %s\n" % e)
        return 1

    cards = sv.detect_cards(rgb)
    if not cards:
        sys.stderr.write("no cards found\n")
        return 0
    print("%d cards:" % len(cards))
    for i, card in enumerate(cards):
        print("  %2d. %s" % (i + 1, sv.card_name(card)))

    sets = sv.find_sets(cards)
    print("%d set%s%s" % (len(sets), "" if len(sets) == 1 else "s", ":" if sets else ""))
    for n, triple in enumerate(sets):
        print("  set #%d: %s" % (n + 1, ", ".join(sv.card_name(cards[i]) for i in triple)))

    base, ext = os.path.splitext(infile)
    outfile = "%s_labeled%s" % (base, ext)
    cv2.imwrite(outfile, annotate(rgb, cards, sets))
    print("wrote %s" % outfile)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
