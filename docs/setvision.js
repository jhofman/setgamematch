/*
 * Computer vision for the game of Set, in plain JavaScript with no dependencies.
 *
 * This is a function-for-function port of setvision.py (see that file for the
 * explanation of the pipeline); tests/js_parity.py checks that the two agree.
 * If you change one, change the other.
 *
 *   const cards = SetVision.detectCards(imageData);   // ImageData-like: {data: RGBA, width, height}
 *   const sets  = SetVision.findSets(cards);          // [[i, j, k], ...]
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.SetVision = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const NUMBERS = [1, 2, 3];
  const COLORS = ["red", "green", "purple"];
  const SHADINGS = ["open", "striped", "solid"];
  const SHAPES = ["diamond", "oval", "squiggle"];

  const WORK_MAX = 2400;
  const CARD_W = 300, CARD_H = 200;
  const MARGIN = 8;

  // ------------------------------------------------------------------ helpers
  function otsu(values) {
    const hist = new Float64Array(256);
    for (let i = 0; i < values.length; i++) hist[values[i]]++;
    let total = 0, sumAll = 0;
    for (let t = 0; t < 256; t++) { total += hist[t]; sumAll += t * hist[t]; }
    let bestT = 0, bestVar = -1, w0 = 0, sum0 = 0;
    for (let t = 0; t < 256; t++) {
      w0 += hist[t];
      if (w0 === 0) continue;
      const w1 = total - w0;
      if (w1 === 0) break;
      sum0 += t * hist[t];
      const m0 = sum0 / w0, m1 = (sumAll - sum0) / w1;
      const v = w0 * w1 * (m0 - m1) * (m0 - m1);
      if (v > bestVar) { bestVar = v; bestT = t; }
    }
    return bestT;
  }

  /** Andrew's monotone chain; points are [x, y]. */
  function convexHull(points) {
    const pts = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const uniq = [];
    for (const p of pts) {
      const q = uniq[uniq.length - 1];
      if (!q || q[0] !== p[0] || q[1] !== p[1]) uniq.push(p);
    }
    if (uniq.length <= 2) return uniq;
    const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lower = [], upper = [];
    for (const p of uniq) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
      lower.push(p);
    }
    for (let i = uniq.length - 1; i >= 0; i--) {
      const p = uniq[i];
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
      upper.push(p);
    }
    lower.pop(); upper.pop();
    return lower.concat(upper);
  }

  function polygonArea(poly) {
    let s = 0;
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i], b = poly[(i + 1) % poly.length];
      s += a[0] * b[1] - a[1] * b[0];
    }
    return Math.abs(s) / 2;
  }

  function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }

  function centroid(poly) {
    let x = 0, y = 0;
    for (const p of poly) { x += p[0]; y += p[1]; }
    return [x / poly.length, y / poly.length];
  }

  function intersect(p1, p2, p3, p4) {
    const d1x = p2[0] - p1[0], d1y = p2[1] - p1[1];
    const d2x = p4[0] - p3[0], d2y = p4[1] - p3[1];
    const den = d1x * d2y - d1y * d2x;
    if (Math.abs(den) < 1e-9) return null;
    const t = ((p3[0] - p1[0]) * d2y - (p3[1] - p1[1]) * d2x) / den;
    return [p1[0] + t * d1x, p1[1] + t * d1y];
  }

  /** Reduce a convex polygon to a quadrilateral by deleting the cheapest edges. */
  function hullToQuad(hull) {
    const poly = hull.map((p) => [p[0], p[1]]);
    while (poly.length > 4) {
      const n = poly.length;
      let best = null;
      for (let i = 0; i < n; i++) {
        const a = poly[i], b = poly[(i + 1) % n];
        const prev = poly[(i - 1 + n) % n], nxt = poly[(i + 2) % n];
        const x = intersect(prev, a, nxt, b);
        if (!x) continue;
        if ((x[0] - a[0]) * (a[0] - prev[0]) + (x[1] - a[1]) * (a[1] - prev[1]) < 0) continue;
        if ((x[0] - b[0]) * (b[0] - nxt[0]) + (x[1] - b[1]) * (b[1] - nxt[1]) < 0) continue;
        const added = polygonArea([a, x, b]);
        if (!best || added < best.added) best = { added, i, x };
      }
      if (!best) return null;
      const j = (best.i + 1) % n;
      poly[best.i] = best.x;
      poly.splice(j, 1);
    }
    return poly;
  }

  /**
   * 3x3 erosion (or dilation) of a 0/1 mask, as a horizontal pass then a vertical
   * one. Pixels outside the image take the value `outside`, or are ignored if null.
   */
  function morph(mask, w, h, isErode, iterations, outside) {
    // erosion is an AND over the neighbourhood, dilation an OR; a pixel
    // outside the image is neutral unless a value was given for it
    const edge = outside === null ? (isErode ? 1 : 0) : (outside ? 1 : 0);
    let src = mask;
    const tmp = new Uint8Array(w * h);
    for (let it = 0; it < iterations; it++) {
      const dst = new Uint8Array(w * h);
      for (let y = 0; y < h; y++) {
        const row = y * w;
        for (let x = 0; x < w; x++) {
          const l = x > 0 ? src[row + x - 1] : edge, r = x < w - 1 ? src[row + x + 1] : edge;
          tmp[row + x] = isErode ? src[row + x] & l & r : src[row + x] | l | r;
        }
      }
      for (let y = 0; y < h; y++) {
        const row = y * w;
        for (let x = 0; x < w; x++) {
          const u = y > 0 ? tmp[row - w + x] : edge, d = y < h - 1 ? tmp[row + w + x] : edge;
          dst[row + x] = isErode ? tmp[row + x] & u & d : tmp[row + x] | u | d;
        }
      }
      src = dst;
    }
    return src;
  }
  const erode = (m, w, h, n, outside = null) => morph(m, w, h, true, n, outside);
  const dilate = (m, w, h, n, outside = null) => morph(m, w, h, false, n, outside);

  /** Grey-scale 3x3 erosion (min) or dilation (max) of a Float32Array, ignoring pixels outside. */
  function greyMorph(src, w, h, isErode, iterations) {
    const pick = isErode ? Math.min : Math.max;
    let a = src;
    const tmp = new Float32Array(w * h);
    for (let it = 0; it < iterations; it++) {
      const dst = new Float32Array(w * h);
      for (let y = 0; y < h; y++) {
        const row = y * w;
        for (let x = 0; x < w; x++) {
          let v = a[row + x];
          if (x > 0) v = pick(v, a[row + x - 1]);
          if (x < w - 1) v = pick(v, a[row + x + 1]);
          tmp[row + x] = v;
        }
      }
      for (let y = 0; y < h; y++) {
        const row = y * w;
        for (let x = 0; x < w; x++) {
          let v = tmp[row + x];
          if (y > 0) v = pick(v, tmp[row - w + x]);
          if (y < h - 1) v = pick(v, tmp[row + w + x]);
          dst[row + x] = v;
        }
      }
      a = dst;
    }
    return a;
  }

  /**
   * Connected components of a 0/1 mask. Returns {labels (Int32Array, 0 = background),
   * stats: [{x, y, w, h, area}, ...]} with stats[i] describing label i + 1.
   */
  function connectedComponents(mask, w, h, eight) {
    const labels = new Int32Array(w * h);
    const stats = [];
    const stack = new Int32Array(w * h);
    for (let start = 0; start < w * h; start++) {
      if (!mask[start] || labels[start]) continue;
      const label = stats.length + 1;
      let top = 0, area = 0, x0 = w, x1 = -1, y0 = h, y1 = -1;
      stack[top++] = start;
      labels[start] = label;
      while (top) {
        const p = stack[--top];
        const x = p % w, y = (p - x) / w;
        area++;
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
        for (let dy = -1; dy <= 1; dy++) {
          const yy = y + dy;
          if (yy < 0 || yy >= h) continue;
          for (let dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;
            if (!eight && dx && dy) continue;
            const xx = x + dx;
            if (xx < 0 || xx >= w) continue;
            const q = yy * w + xx;
            if (mask[q] && !labels[q]) { labels[q] = label; stack[top++] = q; }
          }
        }
      }
      stats.push({ x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1, area });
    }
    return { labels, stats };
  }

  /** Corner points of the left-most and right-most pixel of a region on each row. */
  function rowExtents(test, x0, y0, bw, bh) {
    const pts = [];
    for (let r = 0; r < bh; r++) {
      let first = -1, last = -1;
      for (let c = 0; c < bw; c++) {
        if (test(x0 + c, y0 + r)) { if (first < 0) first = c; last = c; }
      }
      if (first >= 0) {
        pts.push([x0 + first, y0 + r], [x0 + last + 1, y0 + r], [x0 + first, y0 + r + 1], [x0 + last + 1, y0 + r + 1]);
      }
    }
    return pts;
  }

  /** Move the k-th smallest value of a[lo..hi] to position k (quickselect, in place). */
  function selectKth(a, k) {
    let lo = 0, hi = a.length - 1;
    while (lo < hi) {
      const pivot = a[(lo + hi) >> 1];
      let i = lo, j = hi;
      while (i <= j) {
        while (a[i] < pivot) i++;
        while (a[j] > pivot) j--;
        if (i <= j) { const t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
      }
      if (k <= j) hi = j; else if (k >= i) lo = i; else return;
    }
  }

  /** numpy-style percentile (linear interpolation) of a Float32Array; reorders it in place. */
  function percentile(values, p) {
    const n = values.length;
    const pos = (p / 100) * (n - 1);
    const lo = Math.floor(pos);
    selectKth(values, lo);
    if (lo + 1 >= n) return values[lo];
    let next = Infinity;                       // the following order statistic
    for (let i = lo + 1; i < n; i++) if (values[i] < next) next = values[i];
    return values[lo] + (next - values[lo]) * (pos - lo);
  }

  /** Solve A x = b in place (Gaussian elimination with partial pivoting). A is n x n as array of rows. */
  function solve(A, b) {
    const n = b.length;
    for (let c = 0; c < n; c++) {
      let piv = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
      if (Math.abs(A[piv][c]) < 1e-12) return null;
      [A[c], A[piv]] = [A[piv], A[c]];
      [b[c], b[piv]] = [b[piv], b[c]];
      for (let r = c + 1; r < n; r++) {
        const f = A[r][c] / A[c][c];
        if (!f) continue;
        for (let k = c; k < n; k++) A[r][k] -= f * A[c][k];
        b[r] -= f * b[c];
      }
    }
    const x = new Array(n);
    for (let r = n - 1; r >= 0; r--) {
      let s = b[r];
      for (let k = r + 1; k < n; k++) s -= A[r][k] * x[k];
      x[r] = s / A[r][r];
    }
    return x;
  }

  // ---------------------------------------------------------- 1. find cards
  /**
   * Area-averaging resize (like cv2.INTER_AREA when shrinking) of an RGBA image
   * to w2 x h2. Returns Float32Array of RGB triples. Each source row is squeezed
   * horizontally into a small buffer and then spread over the output rows it
   * overlaps, so memory stays small even for a 12-megapixel photo.
   */
  function resizeArea(img, w2, h2) {
    const { data, width: w, height: h } = img;
    const sx = w / w2, sy = h / h2;
    // horizontal spans: output column x2 covers source columns [x0, x1] with
    // partial weights at the two ends
    const x0 = new Int32Array(w2), x1 = new Int32Array(w2);
    const wl = new Float32Array(w2), wr = new Float32Array(w2);
    for (let x2 = 0; x2 < w2; x2++) {
      const a = x2 * sx, b = Math.min(w, (x2 + 1) * sx);
      x0[x2] = Math.floor(a);
      x1[x2] = Math.min(w - 1, Math.ceil(b) - 1);
      wl[x2] = Math.min(b, x0[x2] + 1) - a;
      wr[x2] = x1[x2] > x0[x2] ? b - x1[x2] : 0;
    }
    const out = new Float32Array(w2 * h2 * 3);
    const row = new Float32Array(w2 * 3);
    const inv = 1 / (sx * sy);
    for (let y = 0; y < h; y++) {
      const base = y * w * 4;
      for (let x2 = 0; x2 < w2; x2++) {
        let p = base + x0[x2] * 4;
        let r = wl[x2] * data[p], g = wl[x2] * data[p + 1], bl = wl[x2] * data[p + 2];
        for (let i = x0[x2] + 1; i < x1[x2]; i++) {
          p += 4;
          r += data[p]; g += data[p + 1]; bl += data[p + 2];
        }
        if (x1[x2] > x0[x2]) {
          p = base + x1[x2] * 4;
          r += wr[x2] * data[p]; g += wr[x2] * data[p + 1]; bl += wr[x2] * data[p + 2];
        }
        row[x2 * 3] = r; row[x2 * 3 + 1] = g; row[x2 * 3 + 2] = bl;
      }
      // output rows that this source row overlaps
      const first = Math.min(h2 - 1, Math.floor(y / sy)), last = Math.min(h2 - 1, Math.floor((y + 1) / sy));
      for (let y2 = first; y2 <= last; y2++) {
        const wt = (Math.min(y + 1, (y2 + 1) * sy) - Math.max(y, y2 * sy)) * inv;
        if (wt <= 0) continue;
        const o = y2 * w2 * 3;
        for (let k = 0; k < w2 * 3; k++) out[o + k] += wt * row[k];
      }
    }
    return out;
  }

  /** Shrink an RGBA image so its longest side is at most maxSide. */
  function shrink(img, maxSide = WORK_MAX) {
    const s = Math.max(img.width, img.height) / maxSide;
    if (s <= 1) return img;
    const w2 = Math.floor(img.width / s + 0.5), h2 = Math.floor(img.height / s + 0.5);
    const rgb = resizeArea(img, w2, h2);
    const data = new Uint8ClampedArray(w2 * h2 * 4);
    for (let i = 0; i < w2 * h2; i++) {
      data[i * 4] = Math.round(rgb[i * 3]);
      data[i * 4 + 1] = Math.round(rgb[i * 3 + 1]);
      data[i * 4 + 2] = Math.round(rgb[i * 3 + 2]);
      data[i * 4 + 3] = 255;
    }
    return { data, width: w2, height: h2 };
  }

  function growCorner(quad, k, d) {
    const p = quad[k], prev = quad[(k + 3) % 4], nxt = quad[(k + 1) % 4];
    const lu = Math.max(dist(p, prev), 1e-9), lv = Math.max(dist(p, nxt), 1e-9);
    const u = [(p[0] - prev[0]) / lu, (p[1] - prev[1]) / lu];
    const v = [(p[0] - nxt[0]) / lv, (p[1] - nxt[1]) / lv];
    const sin = Math.abs(u[0] * v[1] - u[1] * v[0]);
    if (sin < 1e-6) return p;
    return [p[0] + (u[0] + v[0]) * d / sin, p[1] + (u[1] + v[1]) * d / sin];
  }

  /** Candidate card quadrilaterals (in the coordinates of img); may contain duplicates and junk. */
  function findCardQuads(img) {
    const W = img.width, H = img.height;
    const s = Math.max(Math.max(H, W) / 800, 1);
    const w = Math.floor(W / s + 0.5), h = Math.floor(H / s + 0.5);
    const white = new Uint8Array(w * h);
    if (s > 1) {
      const small = resizeArea(img, w, h);
      for (let i = 0; i < w * h; i++) {
        // round each channel first, as an 8-bit resize would
        white[i] = Math.min(Math.round(small[i * 3]), Math.round(small[i * 3 + 1]), Math.round(small[i * 3 + 2]));
      }
    } else {
      for (let i = 0; i < w * h; i++) white[i] = Math.min(img.data[i * 4], img.data[i * 4 + 1], img.data[i * 4 + 2]);
    }

    const quads = [];
    const er = Math.max(1, Math.floor(Math.max(h, w) / 400 + 0.5));
    const t0 = otsu(white);
    for (const t of [t0, t0 + (255 - t0) * 0.25, t0 + (255 - t0) * 0.5, t0 - t0 * 0.3]) {
      let mask = new Uint8Array(w * h);
      for (let i = 0; i < w * h; i++) mask[i] = white[i] > t ? 1 : 0;
      mask = erode(mask, w, h, er);
      const { labels, stats } = connectedComponents(mask, w, h, false);
      for (let i = 0; i < stats.length; i++) {
        const st = stats[i], label = i + 1;
        if (st.area < 0.002 * w * h || st.area > 0.9 * w * h) continue;
        const hull = convexHull(rowExtents((x, y) => labels[y * w + x] === label, st.x, st.y, st.w, st.h));
        if (hull.length < 4) continue;
        const hullArea = polygonArea(hull);
        const quad = hullToQuad(hull);
        if (!quad) continue;
        if (hullArea < 0.95 * polygonArea(quad)) continue;   // not a quadrilateral
        if (st.area < 0.35 * hullArea) continue;              // too hollow
        const sides = [0, 1, 2, 3].map((k) => dist(quad[k], quad[(k + 1) % 4]));
        if (Math.min(...sides) < 0.25 * Math.max(...sides)) continue;
        const a = (sides[0] + sides[2]) / 2, b = (sides[1] + sides[3]) / 2;
        if (Math.max(a, b) / Math.max(Math.min(a, b), 1e-6) > 2.6) continue;
        quads.push([0, 1, 2, 3].map((k) => growCorner(quad, k, er)).map((p) => [p[0] * s, p[1] * s]));
      }
    }
    return quads;
  }

  function pointInQuad(p, quad) {
    let sign = 0;
    for (let k = 0; k < 4; k++) {
      const a = quad[k], b = quad[(k + 1) % 4];
      const cr = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
      if (cr !== 0) {
        if (sign === 0) sign = cr > 0 ? 1 : -1;
        else if ((cr > 0) !== (sign > 0)) return false;
      }
    }
    return true;
  }

  // ------------------------------------------------------------- 2. warping
  /** Corner order: clockwise on screen, edge 0->1 long, starting near the top-left. */
  function orderQuad(quad) {
    let q = quad.map((p) => [p[0], p[1]]);
    let s = 0;
    for (let i = 0; i < 4; i++) s += q[i][0] * q[(i + 1) % 4][1] - q[i][1] * q[(i + 1) % 4][0];
    if (s < 0) q.reverse();
    const a = dist(q[0], q[1]) + dist(q[2], q[3]);
    const b = dist(q[1], q[2]) + dist(q[3], q[0]);
    if (b > a) q = [q[1], q[2], q[3], q[0]];
    if (q[2][0] + q[2][1] < q[0][0] + q[0][1]) q = [q[2], q[3], q[0], q[1]];
    return q;
  }

  /** 3x3 homography (row-major, h[8] = 1) mapping the rectangle [0,W]x[0,H] onto quad q. */
  function rectToQuad(W, H, q) {
    const src = [[0, 0], [W, 0], [W, H], [0, H]];
    const A = [], b = [];
    for (let i = 0; i < 4; i++) {
      const [x, y] = src[i], [u, v] = q[i];
      A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
      A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
    }
    const hh = solve(A, b);
    return hh ? hh.concat([1]) : null;
  }

  /** Warp the card under quad q (already ordered) to CARD_W x CARD_H. Returns Float32 RGB. */
  function warpCard(img, q) {
    const longSide = (dist(q[0], q[1]) + dist(q[2], q[3])) / 2;
    const k = Math.min(4, Math.max(1, Math.ceil(longSide / CARD_W - 0.25)));   // supersampling
    const W = CARD_W * k, H = CARD_H * k;
    const M = rectToQuad(W, H, q);
    const out = new Float32Array(CARD_W * CARD_H * 3);
    if (!M) return out;
    const { data, width: sw, height: sh } = img;
    const norm = 1 / (k * k);
    for (let y = 0; y < H; y++) {
      const oy = (y / k) | 0;
      for (let x = 0; x < W; x++) {
        const den = M[6] * x + M[7] * y + 1;
        let u = (M[0] * x + M[1] * y + M[2]) / den;
        let v = (M[3] * x + M[4] * y + M[5]) / den;
        // bilinear sample, replicating the border
        u = Math.min(Math.max(u, 0), sw - 1);
        v = Math.min(Math.max(v, 0), sh - 1);
        const u0 = Math.floor(u), v0 = Math.floor(v);
        const u1 = Math.min(u0 + 1, sw - 1), v1 = Math.min(v0 + 1, sh - 1);
        const fu = u - u0, fv = v - v0;
        const p00 = (v0 * sw + u0) * 4, p01 = (v0 * sw + u1) * 4, p10 = (v1 * sw + u0) * 4, p11 = (v1 * sw + u1) * 4;
        const w00 = (1 - fu) * (1 - fv), w01 = fu * (1 - fv), w10 = (1 - fu) * fv, w11 = fu * fv;
        const o = (oy * CARD_W + ((x / k) | 0)) * 3;
        out[o] += norm * (w00 * data[p00] + w01 * data[p01] + w10 * data[p10] + w11 * data[p11]);
        out[o + 1] += norm * (w00 * data[p00 + 1] + w01 * data[p01 + 1] + w10 * data[p10 + 1] + w11 * data[p11 + 1]);
        out[o + 2] += norm * (w00 * data[p00 + 2] + w01 * data[p01 + 2] + w10 * data[p10 + 2] + w11 * data[p11 + 2]);
      }
    }
    return out;
  }

  // --------------------------------------------- 3. background normalisation
  const N = CARD_W * CARD_H;
  const isInner = (x, y) => x >= MARGIN && x < CARD_W - MARGIN && y >= MARGIN && y < CARD_H - MARGIN;

  /** Quadratic surface per channel fitted through the pixels flagged in `use`; returns Float32 RGB. */
  function fitBackground(card, use) {
    const AtA = [];
    for (let i = 0; i < 6; i++) AtA.push(new Array(6).fill(0));
    const Atb = [new Array(6).fill(0), new Array(6).fill(0), new Array(6).fill(0)];
    const f = new Float64Array(6);
    for (let y = 0; y < CARD_H; y++) {
      const yn = y / CARD_H - 0.5;
      for (let x = 0; x < CARD_W; x++) {
        const i = y * CARD_W + x;
        if (!use[i]) continue;
        const xn = x / CARD_W - 0.5;
        f[0] = 1; f[1] = xn; f[2] = yn; f[3] = xn * xn; f[4] = xn * yn; f[5] = yn * yn;
        for (let a = 0; a < 6; a++) {
          for (let b = a; b < 6; b++) AtA[a][b] += f[a] * f[b];
          for (let c = 0; c < 3; c++) Atb[c][a] += f[a] * card[i * 3 + c];
        }
      }
    }
    for (let a = 0; a < 6; a++) {
      for (let b = 0; b < a; b++) AtA[a][b] = AtA[b][a];
      AtA[a][a] += 1e-6;
    }
    const coef = [];
    for (let c = 0; c < 3; c++) coef.push(solve(AtA.map((r) => r.slice()), Atb[c].slice()) || [255, 0, 0, 0, 0, 0]);
    const bg = new Float32Array(N * 3);
    for (let y = 0; y < CARD_H; y++) {
      const yn = y / CARD_H - 0.5;
      for (let x = 0; x < CARD_W; x++) {
        const xn = x / CARD_W - 0.5, i = y * CARD_W + x;
        for (let c = 0; c < 3; c++) {
          const k = coef[c];
          bg[i * 3 + c] = Math.max(1, k[0] + k[1] * xn + k[2] * yn + k[3] * xn * xn + k[4] * xn * yn + k[5] * yn * yn);
        }
      }
    }
    return bg;
  }

  /** card / background, so that the card's white is 1.0 in every channel. */
  function normalise(card) {
    const innerCount = (CARD_W - 2 * MARGIN) * (CARD_H - 2 * MARGIN);
    const flat = [0, 0, 0];
    const buf = new Float32Array(innerCount);
    for (let c = 0; c < 3; c++) {
      let n = 0;
      for (let y = MARGIN; y < CARD_H - MARGIN; y++)
        for (let x = MARGIN; x < CARD_W - MARGIN; x++) buf[n++] = card[(y * CARD_W + x) * 3 + c];
      flat[c] = Math.max(percentile(buf, 75), 1);
    }
    let norm = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) for (let c = 0; c < 3; c++) norm[i * 3 + c] = card[i * 3 + c] / flat[c];
    for (let pass = 0; pass < 2; pass++) {
      const use = new Uint8Array(N);
      let count = 0;
      for (let y = MARGIN; y < CARD_H - MARGIN; y++) {
        for (let x = MARGIN; x < CARD_W - MARGIN; x++) {
          const i = y * CARD_W + x;
          const r = norm[i * 3], g = norm[i * 3 + 1], b = norm[i * 3 + 2];
          // background = colourless and not too dark (so a shadow counts as background)
          const mn = Math.min(r, g, b), mx = Math.max(r, g, b, 1e-6);
          if (1 - mn / mx < 0.10 && mn > 0.4) { use[i] = 1; count++; }
        }
      }
      if (count < 0.2 * innerCount) break;
      const bg = fitBackground(card, use);
      norm = new Float32Array(N * 3);
      for (let i = 0; i < N * 3; i++) norm[i] = card[i] / bg[i];
    }
    return norm;
  }

  // ------------------------------------------------------------- 4. symbols
  function fillHoles(mask, w, h) {
    // flood the background from the border; whatever is not reached is inside
    const W = w + 2, H = h + 2;
    const bg = new Uint8Array(W * H);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const inside = x > 0 && x <= w && y > 0 && y <= h;
        bg[y * W + x] = inside && mask[(y - 1) * w + (x - 1)] ? 0 : 1;
      }
    }
    const { labels } = connectedComponents(bg, W, H, false);
    const outer = labels[0];
    const out = new Uint8Array(w * h);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) out[y * w + x] = labels[(y + 1) * W + x + 1] !== outer ? 1 : 0;
    return out;
  }

  /**
   * Returns {symbols, sig, mask}. Each symbol has x, y, w, h, area and a mask over
   * its bounding box. With faint = true the ink threshold is lowered to pick up
   * symbols washed out by glare, at the cost of rougher outlines.
   */
  function findSymbols(norm, faint) {
    // Signal: how far each pixel's weakest channel falls below the local white
    // level (a grey-scale closing of the brightest channel). Thin outlines and
    // stripes register as dark ink, the inside of a solid symbol registers by
    // its saturation, and a colourless shadow across the card does not register.
    const brightest = new Float32Array(N), weakest = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const r = norm[i * 3], g = norm[i * 3 + 1], b = norm[i * 3 + 2];
      brightest[i] = Math.max(r, g, b);
      weakest[i] = Math.min(r, g, b);
    }
    const level = greyMorph(greyMorph(brightest, CARD_W, CARD_H, false, 7), CARD_W, CARD_H, true, 7);
    const sig = new Float32Array(N);
    for (let y = MARGIN; y < CARD_H - MARGIN; y++) {
      for (let x = MARGIN; x < CARD_W - MARGIN; x++) {
        const i = y * CARD_W + x;
        sig[i] = Math.min(Math.max(1 - weakest[i] / Math.max(level[i], 1e-6), 0), 1);
      }
    }
    // 3x3 box blur with replicated border
    const smooth = new Float32Array(N);
    for (let y = 0; y < CARD_H; y++) {
      for (let x = 0; x < CARD_W; x++) {
        let s = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const yy = Math.min(Math.max(y + dy, 0), CARD_H - 1);
          for (let dx = -1; dx <= 1; dx++) s += sig[yy * CARD_W + Math.min(Math.max(x + dx, 0), CARD_W - 1)];
        }
        smooth[y * CARD_W + x] = s / 9;
      }
    }
    const strong = percentile(Float32Array.from(smooth), 99.5);
    let thr = Math.max(0.07, 0.30 * strong);
    if (faint) {
      // just above the card's own noise floor
      const inner = new Float32Array((CARD_W - 2 * MARGIN) * (CARD_H - 2 * MARGIN));
      let n = 0;
      for (let y = MARGIN; y < CARD_H - MARGIN; y++)
        for (let x = MARGIN; x < CARD_W - MARGIN; x++) inner[n++] = smooth[y * CARD_W + x];
      const q10 = percentile(inner, 10), q50 = percentile(inner, 50);
      thr = Math.min(thr, Math.max(0.05, q50 + 6 * (q50 - q10)));
    }
    const mask = new Uint8Array(N);
    for (let i = 0; i < N; i++) mask[i] = smooth[i] > thr ? 1 : 0;
    const closed = erode(dilate(mask, CARD_W, CARD_H, 1), CARD_W, CARD_H, 1);
    let filled = fillHoles(closed, CARD_W, CARD_H);
    filled = dilate(erode(filled, CARD_W, CARD_H, 2), CARD_W, CARD_H, 2);

    const { labels, stats } = connectedComponents(filled, CARD_W, CARD_H, true);
    const symbols = [];
    stats.forEach((st, idx) => {
      if (st.h < 0.35 * CARD_H || st.h > 0.95 * CARD_H) return;
      if (st.w < 0.08 * CARD_W || st.w > 0.40 * CARD_W) return;
      if (st.h < 1.2 * st.w) return;
      const m = new Uint8Array(st.w * st.h);
      for (let y = 0; y < st.h; y++)
        for (let x = 0; x < st.w; x++) m[y * st.w + x] = labels[(st.y + y) * CARD_W + st.x + x] === idx + 1 ? 1 : 0;
      symbols.push({ x: st.x, y: st.y, w: st.w, h: st.h, area: st.area, mask: m });
    });
    symbols.sort((a, b) => a.x - b.x);
    return { symbols, sig, mask };
  }

  // ---------------------------------------------------------- 5. attributes
  function rgbToHue(r, g, b) {
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    if (mx - mn < 1e-6) return 0;
    let h;
    if (mx === r) h = ((((g - b) / (mx - mn)) % 6) + 6) % 6;
    else if (mx === g) h = (b - r) / (mx - mn) + 2;
    else h = (r - g) / (mx - mn) + 4;
    return 60 * h;
  }

  /** How far a (normalised) colour is from the card's white: dark or coloured. */
  const inkOf = (r, g, b) => Math.max(1 - Math.min(r, g, b), Math.max(r, g, b) - Math.min(r, g, b));

  /** Periodicity of the row-to-row ink profile inside a symbol: 0 (none) towards 1 (clean stripes). */
  function stripeScore(profile) {
    const n = profile.length;
    if (n < 24) return 0;
    const half = 4, d = [];
    for (let i = half; i < n - half; i++) {
      let s = 0;
      for (let j = i - half; j <= i + half; j++) s += profile[j];
      d.push(profile[i] - s / (2 * half + 1));
    }
    let v = 0;
    for (const x of d) v += x * x;
    if (v < 1e-12) return 0;
    let best = 0;
    for (let lag = 3; lag <= 10; lag++) {
      let s = 0;
      for (let i = 0; i + lag < d.length; i++) s += d[i] * d[i + lag];
      best = Math.max(best, s / v);
    }
    return best;
  }

  function symbolFeatures(norm, sig, sym) {
    const { x, y, w, h, mask: m } = sym;
    const at = (cx, cy) => (y + cy) * CARD_W + x + cx;

    // What white looks like right here: a ring just outside the symbol. All
    // colours below are relative to it, which cancels shadows and whatever
    // the card-wide background fit got wrong.
    const pad = 7;
    const X0 = Math.max(x - pad, 0), Y0 = Math.max(y - pad, 0);
    const X1 = Math.min(x + w + pad, CARD_W), Y1 = Math.min(y + h + pad, CARD_H);
    const bw = X1 - X0, bh = Y1 - Y0;
    const big = new Uint8Array(bw * bh);
    for (let cy = 0; cy < h; cy++)
      for (let cx = 0; cx < w; cx++) big[(y - Y0 + cy) * bw + (x - X0 + cx)] = m[cy * w + cx];
    const near = dilate(big, bw, bh, 3), far = dilate(near, bw, bh, 3);
    const white = [0, 0, 0];
    let ringCount = 0;
    for (let cy = 0; cy < bh; cy++) {
      for (let cx = 0; cx < bw; cx++) {
        if (!far[cy * bw + cx] || near[cy * bw + cx]) continue;
        const i = ((Y0 + cy) * CARD_W + X0 + cx) * 3;
        white[0] += norm[i]; white[1] += norm[i + 1]; white[2] += norm[i + 2];
        ringCount++;
      }
    }
    for (let c = 0; c < 3; c++) white[c] = ringCount >= 20 ? Math.max(white[c] / ringCount, 1e-3) : 1;

    // colour: mean of the strongest ink
    const vals = new Float32Array(sym.area);
    let n = 0;
    for (let cy = 0; cy < h; cy++)
      for (let cx = 0; cx < w; cx++) if (m[cy * w + cx]) vals[n++] = sig[at(cx, cy)];
    const top = percentile(vals, 85);
    let r = 0, g = 0, b = 0, cnt = 0;
    for (let cy = 0; cy < h; cy++) {
      for (let cx = 0; cx < w; cx++) {
        const i = at(cx, cy);
        if (m[cy * w + cx] && sig[i] >= top) { r += norm[i * 3]; g += norm[i * 3 + 1]; b += norm[i * 3 + 2]; cnt++; }
      }
    }
    r /= cnt * white[0]; g /= cnt * white[1]; b /= cnt * white[2];
    const hue = rgbToHue(r, g, b);

    // shading: how inked is the inside, relative to the outline itself?
    const edgeInk = Math.max(inkOf(r, g, b), 1e-6);
    const rad = Math.max(2, Math.floor(0.25 * w + 0.5));
    const count = (a) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s; };
    let core = erode(m, w, h, rad, 0);
    if (count(core) < 12) core = erode(m, w, h, Math.max(1, rad >> 1), 0);
    // Stripes run along the rows of the warped card, so average each row of
    // the core separately and look at the inkiest rows; stripes too fine to
    // resolve still show up as an overall tint.
    const profile = [];
    let cr = 0, cg = 0, cb = 0, cc = 0;
    for (let cy = 0; cy < h; cy++) {
      let rr = 0, rg = 0, rb = 0, rn = 0;
      for (let cx = 0; cx < w; cx++) {
        if (!core[cy * w + cx]) continue;
        const i = at(cx, cy) * 3;
        rr += norm[i]; rg += norm[i + 1]; rb += norm[i + 2]; rn++;
      }
      cr += rr; cg += rg; cb += rb; cc += rn;
      if (rn >= 3) profile.push(inkOf(rr / rn / white[0], rg / rn / white[1], rb / rn / white[2]));
    }
    const stripe = stripeScore(profile);
    let inside = cc > 0 ? inkOf(cr / cc / white[0], cg / cc / white[1], cb / cc / white[2]) : 0;
    if (profile.length >= 5) {
      // resolved stripes are mostly gaps, so go by the inkiest rows, to the
      // extent that the rows really do alternate
      const peak = percentile(Float32Array.from(profile), 80);
      inside += stripe * Math.max(peak - inside, 0);
    }
    const fill = inside / edgeInk;

    // shape
    const hull = convexHull(rowExtents((px, py) => m[py * w + px] === 1, 0, 0, w, h));
    const solidity = sym.area / Math.max(polygonArea(hull), 1);
    const extent = sym.area / (w * h);
    return { hue, fill, solidity, extent, stripe };
  }

  function classifyColor(hue) {
    if (hue < 55 || hue >= 330) return "red";
    if (hue < 195) return "green";
    return "purple";
  }

  function classifyShading(fill, stripe) {
    if (fill < 0.10) return "open";
    // a little ink inside but no rhythm to it: glare or noise on an open symbol
    if (fill < 0.20 && stripe < 0.36) return "open";
    if (fill < 0.84) return "striped";
    return "solid";
  }

  const SHAPE_PROTOTYPES = { diamond: [0.55, 0.965], oval: [0.885, 0.987], squiggle: [0.745, 0.86] };
  const SHAPE_SCALE = [0.07, 0.05];

  function shapeD(name, solidity, extent) {
    const [e, so] = SHAPE_PROTOTYPES[name];
    return ((extent - e) / SHAPE_SCALE[0]) ** 2 + ((solidity - so) / SHAPE_SCALE[1]) ** 2;
  }

  function classifyShape(solidity, extent) {
    let best = null, bestD = Infinity;
    for (const name of SHAPES) {
      const d = shapeD(name, solidity, extent);
      if (d < bestD) { best = name; bestD = d; }
    }
    return best;
  }

  /** How unlike any Set symbol the shape is (0 = textbook). */
  const shapeDistance = (solidity, extent) => Math.min(...SHAPES.map((n) => shapeD(n, solidity, extent)));

  // Symbols sit in fixed slots: one in the middle, two either side of the
  // middle, or three a full pitch apart (PITCH = slot spacing / card length).
  const PITCH = 0.283;

  /**
   * The number on the card judged by where the symbols are, which still works
   * when glare has wiped one out. Returns {number (or null), misfit}.
   */
  function numberFromSlots(symbols) {
    const d = symbols.map((s) => ((s.x + s.w / 2) / CARD_W - 0.5) / PITCH);
    const odd = Math.max(...d.map((v) => Math.abs(v - Math.floor(v + 0.5))));
    const even = Math.max(...d.map((v) => Math.abs(Math.abs(v) - 0.5)));
    if (even < odd) return { number: symbols.length <= 2 ? 2 : null, misfit: even };
    if (Math.max(...d.map(Math.abs)) > 1.5) return { number: null, misfit: odd };
    if (d.some((v) => Math.abs(v) > 0.5)) return { number: 3, misfit: odd };
    return { number: symbols.length === 1 ? 1 : null, misfit: odd };
  }

  /** Fraction of the card covered by ink that is not part of a symbol. */
  function strayInk(mask, symbols) {
    const m = Uint8Array.from(mask);
    for (const s of symbols) {
      const y0 = Math.max(s.y - 4, 0), y1 = Math.min(s.y + s.h + 4, CARD_H);
      const x0 = Math.max(s.x - 4, 0), x1 = Math.min(s.x + s.w + 4, CARD_W);
      for (let y = y0; y < y1; y++) m.fill(0, y * CARD_W + x0, y * CARD_W + x1);
    }
    let n = 0;
    for (let i = 0; i < N; i++) n += m[i];
    return n / N;
  }

  function borderWhiteness(sig) {
    const lo = MARGIN, hi = MARGIN + 6, corner = 30;
    let blank = 0, total = 0;
    const visit = (x, y) => { total++; if (sig[y * CARD_W + x] < 0.2) blank++; };
    for (let x = corner; x < CARD_W - corner; x++)
      for (let d = lo; d < hi; d++) { visit(x, d); visit(x, CARD_H - 1 - d); }
    for (let y = corner; y < CARD_H - corner; y++)
      for (let d = lo; d < hi; d++) { visit(d, y); visit(CARD_W - 1 - d, y); }
    return blank / total;
  }

  const median = (v) => {
    const s = v.slice().sort((a, b) => a - b), n = s.length;
    return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
  };

  function readCardOriented(img, q) {
    const card = warpCard(img, q);
    const norm = normalise(card);
    let { symbols, sig, mask } = findSymbols(norm, false);
    if (!symbols.length) ({ symbols, sig, mask } = findSymbols(norm, true));
    if (!symbols.length) return null;
    if (borderWhiteness(sig) < 0.85) return null;
    const hmax = Math.max(...symbols.map((s) => s.h));
    symbols = symbols.filter((s) => s.h > 0.75 * hmax);
    if (symbols.length < 1 || symbols.length > 3) return null;
    const feats = symbols.map((s) => symbolFeatures(norm, sig, s));

    let sn = 0, cs = 0;
    for (const f of feats) { sn += Math.sin(f.hue * Math.PI / 180); cs += Math.cos(f.hue * Math.PI / 180); }
    const hue = ((Math.atan2(sn / feats.length, cs / feats.length) * 180 / Math.PI) % 360 + 360) % 360;
    // glare can only wash ink out, so the most inked symbol is the best witness
    const fill = Math.max(...feats.map((f) => f.fill));
    const stripe = Math.max(...feats.map((f) => f.stripe));
    const solidity = median(feats.map((f) => f.solidity));
    const extent = median(feats.map((f) => f.extent));
    const { number, misfit } = numberFromSlots(symbols);
    // things that are white and rectangular but not Set cards (the box, the
    // back of the deck, a phone) fail at least one of these
    if (number === null || misfit > 0.3 || strayInk(mask, symbols) > 0.07 || shapeDistance(solidity, extent) > 12) return null;
    return {
      quad: q.map((p) => [p[0], p[1]]),
      number,
      color: classifyColor(hue),
      shading: classifyShading(fill, stripe),
      shape: classifyShape(solidity, extent),
      features: { hue, fill, solidity, extent, stripe },
    };
  }

  /** Describe the card under a quadrilateral, or null if it is not a Set card. */
  function readCard(img, quad) {
    const q = orderQuad(quad);
    const a = dist(q[0], q[1]) + dist(q[2], q[3]);
    const b = dist(q[1], q[2]) + dist(q[3], q[0]);
    let card = readCardOriented(img, q);
    if (!card && a < 1.35 * b) card = readCardOriented(img, [q[1], q[2], q[3], q[0]]);
    return card;
  }

  /** Reading order: rows top to bottom, left to right within a row. */
  function sortCards(cards) {
    if (!cards.length) return cards;
    const centres = cards.map((c) => centroid(c.quad));
    const rowH = median(cards.map((c) => Math.min(dist(c.quad[0], c.quad[1]), dist(c.quad[1], c.quad[2]))));
    const order = cards.map((_, i) => i).sort((i, j) => centres[i][1] - centres[j][1]);
    const rows = [];
    let cur = [order[0]];
    for (const i of order.slice(1)) {
      const meanY = cur.reduce((s, j) => s + centres[j][1], 0) / cur.length;
      if (centres[i][1] - meanY > 0.6 * rowH) { rows.push(cur); cur = [i]; } else cur.push(i);
    }
    rows.push(cur);
    const out = [];
    for (const r of rows) for (const i of r.sort((i, j) => centres[i][0] - centres[j][0])) out.push(cards[i]);
    return out;
  }

  /**
   * The whole pipeline. `img` is ImageData-like ({data: RGBA bytes, width, height}).
   * Returns cards in reading order; quads are in the coordinates of `img`.
   */
  function detectCards(img) {
    const work = shrink(img);
    const scale = img.width / work.width;
    const quads = findCardQuads(work).sort((a, b) => polygonArea(b) - polygonArea(a));
    const cards = [], kept = [];
    for (const quad of quads) {
      const c = centroid(quad);
      if (kept.some((k) => pointInQuad(c, k) || pointInQuad(centroid(k), quad))) continue;
      const card = readCard(work, quad);
      if (card) {
        kept.push(quad);
        card.quad = card.quad.map((p) => [p[0] * scale, p[1] * scale]);
        cards.push(card);
      }
    }
    return sortCards(cards);
  }

  // ---------------------------------------------------------------- the game
  function cardCode(card) {
    return [NUMBERS.indexOf(card.number), COLORS.indexOf(card.color),
      SHADINGS.indexOf(card.shading), SHAPES.indexOf(card.shape)];
  }

  function cardName(card) {
    return `${card.number} ${card.shading} ${card.color} ${card.shape}${card.number === 1 ? "" : "s"}`;
  }

  /** All triples of indices into `cards` that form a set. */
  function findSets(cards) {
    const codes = cards.map(cardCode);
    const sets = [];
    for (let i = 0; i < cards.length; i++)
      for (let j = i + 1; j < cards.length; j++)
        for (let k = j + 1; k < cards.length; k++)
          if ([0, 1, 2, 3].every((a) => (codes[i][a] + codes[j][a] + codes[k][a]) % 3 === 0)) sets.push([i, j, k]);
    return sets;
  }

  return { NUMBERS, COLORS, SHADINGS, SHAPES, WORK_MAX, detectCards, findSets, cardName, cardCode, shrink };
});
