"""
Computer vision for the game of Set: find the cards in a photo and read
the four attributes (number, colour, shading, shape) of each one.

The pipeline is deliberately simple and built from primitives that are easy
to mirror in plain JavaScript (see docs/setvision.js, which is a line-for-line
port of this file used by the static website):

  1. shrink the photo and threshold on "whiteness" (min of R, G, B) to find
     bright blobs; keep the blobs that look like card-shaped quadrilaterals
  2. warp every card to a canonical 300x200 image
  3. divide out the card's own background colour (a smooth fit to the white
     part of the card), which removes lighting gradients and colour casts
  4. threshold the remaining "ink", fill the outlines and take connected
     components to get the symbols; the count is the number
  5. colour from the hue of the strongest ink, shading from how much ink is
     inside the outline, shape from the geometry of the filled symbol
"""

import itertools

import cv2
import numpy as np

NUMBERS = [1, 2, 3]
COLORS = ["red", "green", "purple"]
SHADINGS = ["open", "striped", "solid"]
SHAPES = ["diamond", "oval", "squiggle"]

WORK_MAX = 1600      # longest side of the working image
CARD_W, CARD_H = 300, 200   # canonical card size
MARGIN = 8           # pixels ignored around the edge of a warped card


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def otsu(values, lo=0, hi=256):
    """Otsu threshold of integer values in [lo, hi)."""
    hist = np.bincount(values.ravel().astype(np.int64), minlength=256)[:256].astype(np.float64)
    total = hist.sum()
    sum_all = (hist * np.arange(256)).sum()
    best_t, best_var, w0, sum0 = 0, -1.0, 0.0, 0.0
    for t in range(256):
        w0 += hist[t]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += t * hist[t]
        m0, m1 = sum0 / w0, (sum_all - sum0) / w1
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t


def convex_hull(points):
    """Andrew's monotone chain. points: (n, 2) array. Returns hull as (m, 2) float array."""
    pts = sorted(set(map(tuple, np.asarray(points).tolist())))
    if len(pts) <= 2:
        return np.array(pts, dtype=np.float64)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=np.float64)


def polygon_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _intersect(p1, p2, p3, p4):
    """Intersection of line p1-p2 with line p3-p4 (None if parallel)."""
    d1, d2 = p2 - p1, p4 - p3
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / den
    return p1 + t * d1


def hull_to_quad(hull):
    """
    Reduce a convex polygon to the (approximately) smallest enclosing
    quadrilateral by repeatedly deleting the edge whose removal adds the
    least area (its two neighbours are extended until they meet).
    """
    poly = [np.array(p, dtype=np.float64) for p in hull]
    while len(poly) > 4:
        n = len(poly)
        best = None
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]          # edge to delete
            prev, nxt = poly[(i - 1) % n], poly[(i + 2) % n]
            x = _intersect(prev, a, nxt, b)
            if x is None:
                continue
            # x must lie beyond a (from prev) and beyond b (from nxt)
            if np.dot(x - a, a - prev) < 0 or np.dot(x - b, b - nxt) < 0:
                continue
            added = polygon_area(np.array([a, x, b]))
            if best is None or added < best[0]:
                best = (added, i, x)
        if best is None:
            return None
        _, i, x = best
        j = (i + 1) % n
        poly[i] = x
        del poly[j]
    return np.array(poly)


def row_extents(labels, label, x0, y0, w, h):
    """Left-most and right-most pixel of a component on every row (for the hull)."""
    pts = []
    sub = labels[y0:y0 + h, x0:x0 + w] == label
    for r in range(h):
        xs = np.flatnonzero(sub[r])
        if xs.size:
            pts.append((x0 + xs[0], y0 + r))
            pts.append((x0 + xs[-1] + 1, y0 + r))
            pts.append((x0 + xs[0], y0 + r + 1))
            pts.append((x0 + xs[-1] + 1, y0 + r + 1))
    return np.array(pts)


# --------------------------------------------------------------------------
# 1. find cards
# --------------------------------------------------------------------------
def load_image(path):
    """Read an image as RGB uint8, honouring EXIF rotation."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError("could not read image: %s" % path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def shrink(rgb, max_side=WORK_MAX):
    h, w = rgb.shape[:2]
    s = max(h, w) / float(max_side)
    if s <= 1:
        return rgb
    return cv2.resize(rgb, (int(w / s + 0.5), int(h / s + 0.5)), interpolation=cv2.INTER_AREA)


def find_card_quads(rgb):
    """
    Return 4x2 arrays (corner coordinates in rgb) for card-like blobs. Several
    thresholds are tried, so the same card may appear more than once and some
    candidates are junk; detect_cards() sorts that out.
    """
    H, W = rgb.shape[:2]
    # detection runs on a small copy: fast, and it blurs away the symbols' detail
    s = max(H, W) / 800.0
    s = max(s, 1.0)
    small = cv2.resize(rgb, (int(W / s + 0.5), int(H / s + 0.5)), interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    white = small.min(axis=2)

    quads = []
    er = max(1, int(max(h, w) / 400.0 + 0.5))   # erosion radius in pixels
    # Otsu first; then stricter thresholds in case cards merged with a bright background
    t0 = otsu(white)
    for t in [t0, t0 + (255 - t0) * 0.25, t0 + (255 - t0) * 0.5, t0 - t0 * 0.3]:
        mask = (white > t).astype(np.uint8)
        # opening detaches cards that just touch each other or a glare patch
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=er)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        for i in range(1, n):
            x0, y0, bw, bh, area = stats[i]
            if area < 0.002 * w * h or area > 0.9 * w * h:
                continue
            hull = convex_hull(row_extents(labels, i, x0, y0, bw, bh))
            if len(hull) < 4:
                continue
            hull_area = polygon_area(hull)
            quad = hull_to_quad(hull)
            if quad is None:
                continue
            quad_area = polygon_area(quad)
            if hull_area < 0.95 * quad_area:     # not a quadrilateral
                continue
            if area < 0.35 * hull_area:           # too hollow (a solid 3-symbol card is ~0.5)
                continue
            # undo the erosion (2 px on each side) and go back to full scale
            sides = [np.linalg.norm(quad[k] - quad[(k + 1) % 4]) for k in range(4)]
            if min(sides) < 0.25 * max(sides):
                continue
            a = (sides[0] + sides[2]) / 2
            b = (sides[1] + sides[3]) / 2
            aspect = max(a, b) / max(min(a, b), 1e-6)
            if aspect > 2.6:
                continue
            grown = np.array([grow_corner(quad, k, float(er)) for k in range(4)])
            quads.append(grown * s)

    return quads


def point_in_quad(p, quad):
    sign = 0
    for k in range(4):
        a, b = quad[k], quad[(k + 1) % 4]
        cr = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if cr != 0:
            if sign == 0:
                sign = 1 if cr > 0 else -1
            elif (cr > 0) != (sign > 0):
                return False
    return True


def grow_corner(quad, k, d):
    """Move corner k outwards so that both adjacent edges shift by d pixels."""
    p = quad[k]
    prev, nxt = quad[(k - 1) % 4], quad[(k + 1) % 4]
    u = (p - prev) / max(np.linalg.norm(p - prev), 1e-9)
    v = (p - nxt) / max(np.linalg.norm(p - nxt), 1e-9)
    sin = abs(u[0] * v[1] - u[1] * v[0])
    if sin < 1e-6:
        return p
    return p + (u + v) * d / sin


# --------------------------------------------------------------------------
# 2. warp a card to canonical size
# --------------------------------------------------------------------------
def order_quad(quad):
    """Rotate the corner order so that edge 0->1 is a long edge, clockwise in image coords."""
    q = np.array(quad, dtype=np.float64)
    # make clockwise (positive shoelace in image coordinates, y down)
    x, y = q[:, 0], q[:, 1]
    if np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)) < 0:
        q = q[::-1].copy()
    a = np.linalg.norm(q[0] - q[1]) + np.linalg.norm(q[2] - q[3])
    b = np.linalg.norm(q[1] - q[2]) + np.linalg.norm(q[3] - q[0])
    if b > a:
        q = np.roll(q, -1, axis=0)
    # start from the corner nearest the top-left so output is stable
    if q[2].sum() < q[0].sum():
        q = np.roll(q, -2, axis=0)
    return q


def warp_card(rgb, quad):
    q = order_quad(quad)
    long_side = (np.linalg.norm(q[0] - q[1]) + np.linalg.norm(q[2] - q[3])) / 2
    k = int(min(4, max(1, np.ceil(long_side / CARD_W - 0.25))))   # supersampling factor
    W, H = CARD_W * k, CARD_H * k
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
    big = cv2.warpPerspective(rgb, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if k > 1:
        big = big.reshape(CARD_H, k, CARD_W, k, 3).mean(axis=(1, 3))
    return big.astype(np.float32)


# --------------------------------------------------------------------------
# 3. background normalisation -> ink map
# --------------------------------------------------------------------------
def fit_background(card, use):
    """Least-squares quadratic surface per channel through the pixels flagged in `use`."""
    H, W = card.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    x = (xx / float(W) - 0.5).ravel()
    y = (yy / float(H) - 0.5).ravel()
    A = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)
    sel = use.ravel()
    As = A[sel]
    AtA = As.T @ As + 1e-6 * np.eye(6)
    bg = np.empty_like(card)
    for c in range(3):
        coef = np.linalg.solve(AtA, As.T @ card[:, :, c].ravel()[sel])
        bg[:, :, c] = (A @ coef).reshape(H, W)
    return np.maximum(bg, 1.0)


def normalise(card):
    """Return card / background (white == 1.0 in every channel)."""
    H, W = card.shape[:2]
    inner = np.zeros((H, W), bool)
    inner[MARGIN:H - MARGIN, MARGIN:W - MARGIN] = True
    # start from a flat background: the typical bright pixel
    flat = np.array([np.percentile(card[:, :, c][inner], 75) for c in range(3)], dtype=np.float32)
    norm = card / np.maximum(flat, 1.0)
    for _ in range(2):
        ink = 1.0 - norm.min(axis=2)
        chroma = norm.max(axis=2) - norm.min(axis=2)
        use = inner & (ink < 0.12) & (chroma < 0.10)
        if use.sum() < 0.2 * inner.sum():
            break
        bg = fit_background(card, use)
        norm = card / bg
    return norm


# --------------------------------------------------------------------------
# 4. symbols
# --------------------------------------------------------------------------
def fill_holes(mask):
    """Fill everything not connected to the image border."""
    H, W = mask.shape
    padded = np.zeros((H + 2, W + 2), np.uint8)
    padded[1:-1, 1:-1] = mask
    n, labels = cv2.connectedComponents((padded == 0).astype(np.uint8), connectivity=4)
    return (labels[1:-1, 1:-1] != labels[0, 0]).astype(np.uint8)


def find_symbols(norm):
    """
    Returns (list of symbols, ink map). Each symbol is a dict with the filled
    mask restricted to its bounding box.
    """
    H, W = norm.shape[:2]
    ink = np.clip(1.0 - norm.min(axis=2), 0, 1)
    chroma = np.clip(norm.max(axis=2) - norm.min(axis=2), 0, 1)
    # signal: coloured or dark relative to the card's white
    sig = np.maximum(ink, chroma)
    sig[:MARGIN], sig[-MARGIN:], sig[:, :MARGIN], sig[:, -MARGIN:] = 0, 0, 0, 0

    # the mask is made from a lightly smoothed copy so that sensor noise and
    # JPEG blocks do not leave speckles stuck to the outlines
    smooth = cv2.blur(sig, (3, 3), borderType=cv2.BORDER_REPLICATE)
    strong = np.percentile(smooth, 99.5)
    thr = max(0.07, 0.30 * strong)
    mask = (smooth > thr).astype(np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    # close 1-pixel gaps in thin outlines, fill, then shave off anything thin
    closed = cv2.erode(cv2.dilate(mask, k3), k3)
    filled = fill_holes(closed)
    filled = cv2.dilate(cv2.erode(filled, k3, iterations=2), k3, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    symbols = []
    for i in range(1, n):
        x0, y0, bw, bh, area = stats[i]
        if bh < 0.35 * H or bh > 0.95 * H:
            continue
        if bw < 0.08 * W or bw > 0.40 * W:
            continue
        if bh < 1.2 * bw:
            continue
        symbols.append(dict(x=int(x0), y=int(y0), w=int(bw), h=int(bh), area=int(area),
                            mask=(labels[y0:y0 + bh, x0:x0 + bw] == i)))
    symbols.sort(key=lambda s: s["x"])
    return symbols, sig


# --------------------------------------------------------------------------
# 5. attributes
# --------------------------------------------------------------------------
def rgb_to_hue(r, g, b):
    mx, mn = max(r, g, b), min(r, g, b)
    if mx - mn < 1e-6:
        return 0.0
    if mx == r:
        h = ((g - b) / (mx - mn)) % 6
    elif mx == g:
        h = (b - r) / (mx - mn) + 2
    else:
        h = (r - g) / (mx - mn) + 4
    return 60.0 * h


def symbol_features(norm, sig, sym):
    x, y, w, h = sym["x"], sym["y"], sym["w"], sym["h"]
    m = sym["mask"]
    s = sig[y:y + h, x:x + w]
    px = norm[y:y + h, x:x + w]

    # --- colour: mean of the strongest ink
    vals = s[m]
    top = np.percentile(vals, 85)
    strong = m & (s >= top)
    rgb = px[strong].mean(axis=0)
    hue = rgb_to_hue(*rgb)

    # --- shading: how inked is the inside, relative to the outline itself?
    # Colours are averaged before measuring ink so that stripes blend to a
    # tint and pixel noise cancels out.
    edge_ink = max(1.0 - float(rgb.min()), float(rgb.max() - rgb.min()), 1e-6)
    r = max(2, int(0.25 * w + 0.5))
    core = erode_mask(m, r)
    if core.sum() < 12:
        core = erode_mask(m, max(1, r // 2))
    if core.any():
        c = px[core].mean(axis=0)
        fill = max(1.0 - float(c.min()), float(c.max() - c.min())) / edge_ink
    else:
        fill = 0.0

    # --- shape
    ys, xs = np.nonzero(m)
    hull = convex_hull(np.concatenate([np.stack([xs, ys], 1), np.stack([xs + 1, ys], 1),
                                       np.stack([xs, ys + 1], 1), np.stack([xs + 1, ys + 1], 1)]))
    area = float(m.sum())
    solidity = area / max(polygon_area(hull), 1.0)
    extent = area / float(w * h)
    return dict(hue=hue, rgb=[float(v) for v in rgb], fill=fill, solidity=solidity, extent=extent,
                edge=edge_ink)


def erode_mask(mask, r):
    return cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=r,
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def classify_color(hue):
    if hue < 55 or hue >= 330:
        return "red"
    if hue < 195:
        return "green"
    return "purple"


def classify_shading(fill):
    if fill < 0.13:
        return "open"
    if fill < 0.84:
        return "striped"
    return "solid"


# typical (extent, solidity) of each shape, and how much each feature varies
SHAPE_PROTOTYPES = {"diamond": (0.55, 0.965), "oval": (0.885, 0.987), "squiggle": (0.745, 0.86)}
SHAPE_SCALE = (0.07, 0.05)


def classify_shape(solidity, extent):
    """Nearest prototype in (extent, solidity) space."""
    best, best_d = None, None
    for name in SHAPES:
        e, so = SHAPE_PROTOTYPES[name]
        d = ((extent - e) / SHAPE_SCALE[0]) ** 2 + ((solidity - so) / SHAPE_SCALE[1]) ** 2
        if best_d is None or d < best_d:
            best, best_d = name, d
    return best


def border_whiteness(sig):
    """
    Fraction of a band just inside the card edge that is blank. A real card
    is white all the way round; a quad that swallowed some table is not.
    """
    H, W = sig.shape
    lo, hi, corner = MARGIN, MARGIN + 6, 30
    band = np.concatenate([
        sig[lo:hi, corner:W - corner].ravel(), sig[H - hi:H - lo, corner:W - corner].ravel(),
        sig[corner:H - corner, lo:hi].ravel(), sig[corner:H - corner, W - hi:W - lo].ravel()])
    return float((band < 0.2).mean())


def read_card(rgb, quad):
    """Returns a dict describing the card, or None if it does not look like a Set card."""
    q = order_quad(quad)
    a = np.linalg.norm(q[0] - q[1]) + np.linalg.norm(q[2] - q[3])
    b = np.linalg.norm(q[1] - q[2]) + np.linalg.norm(q[3] - q[0])
    card = read_card_oriented(rgb, q)
    if card is None and a < 1.35 * b:
        # nearly square (a card seen at an angle): the long side may be the other one
        card = read_card_oriented(rgb, np.roll(q, -1, axis=0))
    return card


def read_card_oriented(rgb, q):
    card = warp_card(rgb, q)
    norm = normalise(card)
    symbols, sig = find_symbols(norm)
    if not symbols:
        return None
    if border_whiteness(sig) < 0.85:
        return None
    # keep the symbols that agree in size with the biggest one
    hmax = max(s["h"] for s in symbols)
    symbols = [s for s in symbols if s["h"] > 0.75 * hmax]
    if not 1 <= len(symbols) <= 3:
        return None
    feats = [symbol_features(norm, sig, s) for s in symbols]

    # average the features over the symbols (hue on the circle)
    ang = np.deg2rad([f["hue"] for f in feats])
    hue = float(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360)
    fill = float(np.median([f["fill"] for f in feats]))
    solidity = float(np.median([f["solidity"] for f in feats]))
    extent = float(np.median([f["extent"] for f in feats]))
    return dict(
        quad=[[float(x), float(y)] for x, y in q],
        number=len(symbols),
        color=classify_color(hue),
        shading=classify_shading(fill),
        shape=classify_shape(solidity, extent),
        features=dict(hue=hue, fill=fill, solidity=solidity, extent=extent),
        _debug=dict(card=card, norm=norm, sig=sig, symbols=symbols),
    )


def sort_cards(cards):
    """Reading order: rows top to bottom, left to right within a row."""
    if not cards:
        return cards
    centres = [np.mean(c["quad"], axis=0) for c in cards]
    heights = []
    for c in cards:
        q = np.array(c["quad"])
        heights.append(min(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[1] - q[2])))
    row_h = float(np.median(heights))
    order = sorted(range(len(cards)), key=lambda i: centres[i][1])
    rows, cur = [], [order[0]]
    for i in order[1:]:
        if centres[i][1] - np.mean([centres[j][1] for j in cur]) > 0.6 * row_h:
            rows.append(cur)
            cur = [i]
        else:
            cur.append(i)
    rows.append(cur)
    out = []
    for r in rows:
        out.extend(sorted(r, key=lambda i: centres[i][0]))
    return [cards[i] for i in out]


def detect_cards(rgb):
    """Full pipeline on an RGB uint8 image. Card quads are in the coordinates of `rgb`."""
    work = shrink(rgb)
    scale = rgb.shape[1] / float(work.shape[1])
    # biggest candidates first; a candidate overlapping a card we already
    # accepted is the same card again (or a piece of it) and is skipped
    quads = sorted(find_card_quads(work), key=lambda q: -polygon_area(q))
    cards, kept = [], []
    for quad in quads:
        c = quad.mean(axis=0)
        if any(point_in_quad(c, k) or point_in_quad(k.mean(axis=0), quad) for k in kept):
            continue
        card = read_card(work, quad)
        if card is not None:
            kept.append(quad)
            card["quad"] = [[x * scale, y * scale] for x, y in card["quad"]]
            cards.append(card)
    return sort_cards(cards)


# --------------------------------------------------------------------------
# the game itself
# --------------------------------------------------------------------------
def card_code(card):
    return (NUMBERS.index(card["number"]), COLORS.index(card["color"]),
            SHADINGS.index(card["shading"]), SHAPES.index(card["shape"]))


def card_name(card):
    return "%d %s %s %s%s" % (card["number"], card["shading"], card["color"], card["shape"],
                              "" if card["number"] == 1 else "s")


def find_sets(cards):
    """All triples of indices into `cards` that form a set."""
    codes = [card_code(c) for c in cards]
    sets = []
    for i, j, k in itertools.combinations(range(len(cards)), 3):
        if all((codes[i][a] + codes[j][a] + codes[k][a]) % 3 == 0 for a in range(4)):
            sets.append((i, j, k))
    return sets
