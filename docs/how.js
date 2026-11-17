/* "How it works" page: theme toggle, the slot diagram and the charts (drawn from how/data.json). */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const NS = "http://www.w3.org/2000/svg";
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const dark = () => document.documentElement.dataset.theme === "dark";
  // the three Set colours, lightened on the dark theme so they stay legible
  const ink = (c) => (dark() ? { red: "#ff7b72", green: "#3fb950", purple: "#b58cf7" } : { red: "#e5352b", green: "#1a9e4b", purple: "#6a3fa0" })[c];

  // ----------------------------------------------------------- theme toggle
  const MOON = '<path d="M9.6 1.2a6.8 6.8 0 1 0 5.2 9.9A5.6 5.6 0 0 1 9.6 1.2Z"/>';
  const SUN = '<circle cx="8" cy="8" r="3.2"/><path d="M8 0.5v2M8 13.5v2M0.5 8h2M13.5 8h2M2.7 2.7l1.4 1.4M11.9 11.9l1.4 1.4M2.7 13.3l1.4-1.4M11.9 4.1l1.4-1.4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>';
  const toggle = $("theme-toggle");
  function showTheme() {
    toggle.querySelector("svg").innerHTML = dark() ? SUN : MOON;
    toggle.setAttribute("aria-label", dark() ? "Switch to light theme" : "Switch to dark theme");
  }
  toggle.addEventListener("click", () => {
    const next = dark() ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) { /* private mode */ }
    showTheme();
    drawAll();
  });
  showTheme();

  // ---------------------------------------------------------------- helpers
  function el(name, attrs, parent) {
    const e = document.createElementNS(NS, name);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function text(parent, x, y, str, attrs) {
    const t = el("text", Object.assign({ x, y, fill: css("--text-2"), "font-size": 12 }, attrs || {}), parent);
    t.textContent = str;
    return t;
  }
  function svgFor(container, height) {
    container.innerHTML = "";
    const width = Math.max(280, container.clientWidth);
    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height, role: "img" }, container);
    return { svg, width };
  }
  // deterministic jitter so dots don't stack exactly
  const jitter = (i) => ((Math.sin(i * 12.9898) * 43758.5453) % 1 + 1) % 1 - 0.5;

  // ---------------------------------------------------------------- tooltip
  const tip = $("tooltip");
  function showTip(evt, html) {
    tip.innerHTML = html;
    tip.hidden = false;
    const pad = 12, r = tip.getBoundingClientRect();
    let x = evt.clientX + pad, y = evt.clientY - r.height - pad;
    if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - pad;
    if (y < 8) y = evt.clientY + pad;
    tip.style.left = Math.max(8, x) + "px";
    tip.style.top = y + "px";
  }
  const hideTip = () => { tip.hidden = true; };
  document.addEventListener("scroll", hideTip, { passive: true });

  const SOURCE = { repo: "photo in this repo", other: "photo from another project" };
  function dot(svg, x, y, r, fill, row, value) {
    const g = el("g", { class: "dot", tabindex: "0", "aria-label": `${row.truth}: ${value}` }, svg);
    el("circle", { cx: x, cy: y, r: 10, fill: "transparent" }, g);            // bigger hit target
    el("circle", { cx: x, cy: y, r, fill, stroke: css("--bg"), "stroke-width": 1.5 }, g);
    const html = `<strong>${row.truth}</strong><br>${value}<br><span class="muted">${SOURCE[row.source]}</span>`;
    g.addEventListener("pointerenter", (e) => showTip(e, html));
    g.addEventListener("pointermove", (e) => showTip(e, html));
    g.addEventListener("pointerleave", hideTip);
    g.addEventListener("focus", () => { const b = g.getBoundingClientRect(); showTip({ clientX: b.right, clientY: b.top }, html); });
    g.addEventListener("blur", hideTip);
  }

  function axisX(svg, x0, x1, y, ticks, scale, fmt) {
    el("line", { x1: x0, x2: x1, y1: y, y2: y, stroke: css("--border") }, svg);
    for (const t of ticks) {
      const x = scale(t);
      el("line", { x1: x, x2: x, y1: y, y2: y + 4, stroke: css("--border") }, svg);
      text(svg, x, y + 17, fmt(t), { "text-anchor": "middle", "font-size": 11 });
    }
  }

  // -------------------------------------------------------------- the charts
  let DATA = null;

  /** One row per class, a dot per card along a single measured value. */
  function stripChart(container, opts) {
    const rowH = 34, top = opts.top || 26, left = 70, right = 14, bottom = 30;
    const { svg, width } = svgFor(container, top + rowH * opts.rows.length + bottom);
    svg.setAttribute("aria-label", opts.label);
    const sx = (v) => left + (Math.min(Math.max(v, opts.min), opts.max) - opts.min) / (opts.max - opts.min) * (width - left - right);
    const y0 = top, y1 = top + rowH * opts.rows.length;

    if (opts.background) opts.background(svg, sx, y0, y1);
    opts.rows.forEach((name, r) => {
      const cy = top + rowH * (r + 0.5);
      if (r) el("line", { x1: left, x2: width - right, y1: top + rowH * r, y2: top + rowH * r, stroke: css("--border"), "stroke-dasharray": "2 3", opacity: 0.6 }, svg);
      text(svg, left - 10, cy + 4, name, { "text-anchor": "end", fill: css("--text") });
    });
    axisX(svg, left, width - right, y1, opts.ticks, sx, opts.fmt);
    if (opts.foreground) opts.foreground(svg, sx, y0, y1);
    DATA.features.forEach((row, i) => {
      const r = opts.rows.indexOf(row[opts.rowKey]);
      if (r < 0) return;
      const cy = top + rowH * (r + 0.5) + jitter(i) * (rowH - 14);
      dot(svg, sx(row[opts.value]), cy, 3.5, opts.color(row), row, opts.describe(row));
    });
  }

  function drawHue() {
    const t = DATA.thresholds.hue;
    stripChart($("chart-hue"), {
      label: "Strip chart of measured hue for each card, in rows by true colour. Red cards sit near 0 and 360 degrees, green near 100 to 165, purple near 230 to 310.",
      rows: ["red", "green", "purple"], rowKey: "color", value: "hue", min: 0, max: 360,
      ticks: [0, 60, 120, 180, 240, 300, 360], fmt: (v) => v + "°",
      color: (row) => ink(row.color),
      describe: (row) => `hue ${Math.round(row.hue)}°`,
      background: (svg, sx, y0, y1) => {
        const bands = [[0, t.red_below, "red"], [t.red_below, t.green_below, "green"], [t.green_below, t.red_from, "purple"], [t.red_from, 360, "red"]];
        bands.forEach(([a, b, name], k) => {
          el("rect", { x: sx(a), y: y0, width: sx(b) - sx(a), height: y1 - y0, fill: k % 2 ? css("--bg-3") : css("--bg-2") }, svg);
          if (b - a > 40) text(svg, (sx(a) + sx(b)) / 2, y0 - 8, name, { "text-anchor": "middle", "font-size": 11 });
        });
      },
    });
  }

  function drawFill() {
    const t = DATA.thresholds.fill;
    stripChart($("chart-fill"), {
      label: "Strip chart of inside ink for each card, in rows by true shading. Open cards sit near 0, striped between about 0.13 and 0.5, solid near 1.",
      rows: ["open", "striped", "solid"], rowKey: "shading", value: "fill", min: 0, max: 1.05,
      ticks: [0, 0.2, 0.4, 0.6, 0.8, 1], fmt: (v) => v.toFixed(1),
      color: (row) => ink(row.color),
      describe: (row) => `inside ink ${row.fill.toFixed(2)}, row rhythm ${row.stripe.toFixed(2)}`,
      background: (svg, sx, y0, y1) => {
        el("rect", { x: sx(t.open_below), y: y0, width: sx(t.faint_below) - sx(t.open_below), height: y1 - y0, fill: css("--bg-2") }, svg);
      },
      foreground: (svg, sx, y0, y1) => {
        for (const [v, label] of [[t.open_below, "0.10"], [t.solid_from, "0.84"]]) {
          el("line", { x1: sx(v), x2: sx(v), y1: y0 - 4, y2: y1, stroke: css("--text-2"), "stroke-dasharray": "4 3" }, svg);
          text(svg, sx(v), y0 - 8, label, { "text-anchor": "middle", "font-size": 11 });
        }
      },
    });
  }

  function drawShape() {
    const container = $("chart-shape");
    const top = 12, left = 52, right = 14, bottom = 40;
    const height = 280;
    const { svg, width } = svgFor(container, height);
    svg.setAttribute("aria-label", "Scatter plot of extent against solidity for each card. Diamonds cluster near extent 0.55 and solidity 0.96, ovals near 0.88 and 0.99, squiggles near 0.75 and 0.86.");
    const ex = [0.45, 0.95], so = [0.8, 1.0];
    const sx = (v) => left + (v - ex[0]) / (ex[1] - ex[0]) * (width - left - right);
    const sy = (v) => top + (so[1] - v) / (so[1] - so[0]) * (height - top - bottom);
    // recessive grid
    for (const v of [0.85, 0.9, 0.95, 1.0]) {
      el("line", { x1: left, x2: width - right, y1: sy(v), y2: sy(v), stroke: css("--border"), opacity: 0.6 }, svg);
      text(svg, left - 8, sy(v) + 4, v.toFixed(2), { "text-anchor": "end", "font-size": 11 });
    }
    axisX(svg, left, width - right, height - bottom, [0.5, 0.6, 0.7, 0.8, 0.9], sx, (v) => v.toFixed(1));
    text(svg, (left + width - right) / 2, height - 4, "extent: share of the bounding box filled", { "text-anchor": "middle", "font-size": 11 });
    text(svg, 12, (top + height - bottom) / 2, "solidity", { "text-anchor": "middle", "font-size": 11, transform: `rotate(-90 12 ${(top + height - bottom) / 2})` });

    DATA.features.forEach((row, i) => {
      dot(svg, sx(row.extent) + jitter(i) * 2, sy(row.solidity) + jitter(i + 7) * 2, 3.5, ink(row.color), row,
        `extent ${row.extent.toFixed(2)}, solidity ${row.solidity.toFixed(2)}`);
    });
    // direct labels beside each cluster
    const P = DATA.thresholds.shapes;
    const label = (name, dx, dy, anchor) => text(svg, sx(P[name].extent) + dx, sy(P[name].solidity) + dy, name + "s",
      { "text-anchor": anchor, fill: css("--text"), "font-weight": 600, "font-size": 12 });
    label("diamond", 14, -14, "start");
    label("oval", -16, 22, "end");
    label("squiggle", 20, 4, "start");
  }

  function drawSparks() {
    for (const box of document.querySelectorAll(".spark")) {
      const s = DATA.shading.find((d) => d.key === box.dataset.key);
      if (!s) continue;
      const height = 44, pad = 4;
      const { svg, width } = svgFor(box, height);
      svg.setAttribute("aria-label", `Ink in each row of the ${s.key} symbol's middle: inside ink ${s.fill.toFixed(2)}`);
      const n = s.profile.length;
      const sx = (i) => pad + i / (n - 1) * (width - 2 * pad);
      const sy = (v) => height - pad - Math.min(v, 1.1) / 1.1 * (height - 2 * pad);
      el("line", { x1: pad, x2: width - pad, y1: sy(0), y2: sy(0), stroke: css("--border") }, svg);
      el("line", { x1: pad, x2: width - pad, y1: sy(1), y2: sy(1), stroke: css("--border"), "stroke-dasharray": "2 3" }, svg);
      el("polyline", { points: s.profile.map((v, i) => `${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(" "), fill: "none", stroke: css("--accent"), "stroke-width": 2, "stroke-linejoin": "round" }, svg);
    }
  }

  function drawSlots() {
    const svg = $("slots");
    svg.innerHTML = "";
    const pitch = DATA ? DATA.thresholds.pitch : 0.283;
    const W = 180, H = 120, gap = 50;
    const accent = css("--accent"), border = css("--border-hover"), muted = css("--text-2");
    [[1, [0]], [2, [-0.5, 0.5]], [3, [-1, 0, 1]]].forEach(([n, slots], k) => {
      const x0 = 15 + k * (W + gap - 15), y0 = 8;
      el("rect", { x: x0, y: y0, width: W, height: H, rx: 10, fill: "#fff", stroke: border, "stroke-width": 1.5 }, svg);
      // all possible slot positions as faint ticks
      for (const s of [-1, -0.5, 0, 0.5, 1]) {
        const x = x0 + W / 2 + s * pitch * W;
        el("line", { x1: x, x2: x, y1: y0 + 8, y2: y0 + H - 8, stroke: "#c9d1d9", "stroke-dasharray": "2 3" }, svg);
      }
      for (const s of slots) {
        const x = x0 + W / 2 + s * pitch * W;
        el("rect", { x: x - 13, y: y0 + 22, width: 26, height: H - 44, rx: 13, fill: "none", stroke: accent, "stroke-width": 3 }, svg);
      }
      text(svg, x0 + W / 2, y0 + H + 18, n === 1 ? "one: middle" : n === 2 ? "two: half a step out" : "three: a step out", { "text-anchor": "middle", fill: muted, "font-size": 24 });
    });
    svg.setAttribute("viewBox", `0 0 ${15 + 3 * (W + gap - 15) - gap + 15 + 15} ${H + 40}`);
  }

  function fillShapeNumbers() {
    for (const s of DATA.shapes) {
      const elt = document.querySelector(`.num[data-shape="${s.key}"]`);
      if (elt) elt.textContent = `extent ${s.extent.toFixed(2)} · solidity ${s.solidity.toFixed(2)}`;
    }
    for (const n of document.querySelectorAll(".n-cards")) n.textContent = DATA.features.length;
  }

  function drawAll() {
    drawSlots();
    if (!DATA) return;
    drawHue(); drawFill(); drawShape(); drawSparks();
  }

  drawSlots();
  fetch("how/data.json")
    .then((r) => r.json())
    .then((d) => { DATA = d; fillShapeNumbers(); drawAll(); })
    .catch(() => { for (const c of document.querySelectorAll(".chart")) c.textContent = "The charts could not be loaded."; });

  let timer = 0;
  window.addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(drawAll, 120); });
})();
