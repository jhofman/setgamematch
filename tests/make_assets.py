#!/usr/bin/env python
"""Regenerate docs/icons/*.png and docs/sample.jpg (needs Pillow)."""
import math
import os
import sys

import cv2
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import setvision as sv  # noqa: E402


def squiggle(n=48, amp=6.5, r=11.5):
    """Outline of a squiggle in a 40 x 80 box: a wavy centre line with a round-capped thick stroke.
    (docs/app.js draws the same curve.)"""
    left, right = [], []
    for i in range(n + 1):
        t = i / float(n)
        y = 12 + 56 * t
        x = 20 + amp * math.sin(2 * math.pi * t)
        dx, dy = amp * 2 * math.pi * math.cos(2 * math.pi * t), 56.0
        L = math.hypot(dx, dy)
        nx, ny = dy / L, -dx / L
        right.append((x + r * nx, y + r * ny))
        left.append((x - r * nx, y - r * ny))

    def cap(a, b):   # semicircle from a to b
        cx, cy = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        a0 = math.atan2(a[1] - cy, a[0] - cx)
        return [(cx + r * math.cos(a0 + math.pi * k / 12), cy + r * math.sin(a0 + math.pi * k / 12)) for k in range(1, 12)]

    return right + cap(right[-1], left[-1]) + left[::-1] + cap(left[0], right[0])


def icon(size, pad):
    S = size * 4
    img = Image.new("RGB", (S, S), (9, 105, 218))   # the app accent, #0969da
    d = ImageDraw.Draw(img)
    m = int(S * pad)
    cw = S - 2 * m
    ch = int(cw * 0.68)
    top = (S - ch) // 2
    d.rounded_rectangle([m, top, m + cw, top + ch], radius=int(cw * 0.07), fill=(255, 255, 255))
    sh = ch * 0.62
    sc = sh / 80.0
    sw = 40 * sc
    gap = cw * 0.065
    x0 = m + (cw - (3 * sw + 2 * gap)) / 2
    y0 = top + (ch - sh) / 2

    def tr(pts, k):
        return [(x0 + k * (sw + gap) + x * sc, y0 + y * sc) for x, y in pts]

    d.polygon(tr([(20, 2), (38, 40), (20, 78), (2, 40)], 0), fill=(229, 53, 43))
    (x, y), (x2, y2) = tr([(4, 2), (36, 78)], 1)
    d.rounded_rectangle([x, y, x2, y2], radius=int(16 * sc), outline=(26, 158, 75), width=max(3, int(sc * 6)),
                        fill=(255, 255, 255))
    d.polygon(tr(squiggle(), 2), fill=(106, 63, 160))
    return img.resize((size, size), Image.LANCZOS)


def main():
    out = os.path.join(ROOT, "docs", "icons")
    os.makedirs(out, exist_ok=True)
    icon(512, 0.10).save(os.path.join(out, "icon-512.png"))
    icon(192, 0.10).save(os.path.join(out, "icon-192.png"))
    icon(180, 0.10).save(os.path.join(out, "apple-touch-icon.png"))
    icon(512, 0.20).save(os.path.join(out, "icon-maskable-512.png"))
    icon(32, 0.06).save(os.path.join(out, "favicon-32.png"))
    im = sv.shrink(sv.load_image(os.path.join(ROOT, "board.jpg")), 1600)
    cv2.imwrite(os.path.join(ROOT, "docs", "sample.jpg"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 82])


if __name__ == "__main__":
    main()
