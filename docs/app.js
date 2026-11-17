/* Set Finder: the page logic. The vision lives in setvision.js. */
(function () {
  "use strict";

  const SV = window.SetVision;
  const $ = (id) => document.getElementById(id);
  const INK = { red: "#e5352b", green: "#1a9e4b", purple: "#6a3fa0" };
  // outline colours for the sets (chosen to stay distinct on a photo)
  const HIGHLIGHT = "#2f81f7";   // outline of the highlighted set on the photo

  const SHOW_MAX = 1600;   // longest side of the photo as displayed
  const FULL_MAX = 4096;   // longest side handed to the detector (canvas size limits on phones)

  const state = { cards: [], sets: [], selected: null, image: null };

  // ---------------------------------------------------------- card drawings
  /** Outline of a squiggle in a 40 x 80 box (same curve as tests/make_assets.py). */
  const SQUIGGLE = (function () {
    const n = 48, amp = 6.5, r = 11.5, left = [], right = [];
    for (let i = 0; i <= n; i++) {
      const t = i / n, y = 12 + 56 * t, x = 20 + amp * Math.sin(2 * Math.PI * t);
      const dx = amp * 2 * Math.PI * Math.cos(2 * Math.PI * t), dy = 56, L = Math.hypot(dx, dy);
      right.push([x + r * dy / L, y - r * dx / L]);
      left.push([x - r * dy / L, y + r * dx / L]);
    }
    const cap = (a, b) => {
      const cx = (a[0] + b[0]) / 2, cy = (a[1] + b[1]) / 2, a0 = Math.atan2(a[1] - cy, a[0] - cx), out = [];
      for (let k = 1; k < 12; k++) out.push([cx + r * Math.cos(a0 + Math.PI * k / 12), cy + r * Math.sin(a0 + Math.PI * k / 12)]);
      return out;
    };
    const pts = right.concat(cap(right[n], left[n]), left.slice().reverse(), cap(left[0], right[0]));
    return pts.map((p) => p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" ");
  })();

  function symbolMarkup(card) {
    const ink = INK[card.color];
    const fill = card.shading === "solid" ? ink : card.shading === "striped" ? `url(#stripe-${card.color})` : "#fff";
    const paint = `fill="${fill}" stroke="${ink}" stroke-width="3.5" stroke-linejoin="round"`;
    if (card.shape === "diamond") return `<polygon points="20,3 37,40 20,77 3,40" ${paint}/>`;
    if (card.shape === "oval") return `<rect x="5" y="3" width="30" height="74" rx="15" ${paint}/>`;
    return `<polygon points="${SQUIGGLE}" ${paint}/>`;
  }

  /** An SVG drawing of a card, 150 x 100. */
  function cardSVG(card) {
    const scale = 0.72, w = 40 * scale, gap = 9;
    const total = card.number * w + (card.number - 1) * gap;
    let symbols = "";
    for (let i = 0; i < card.number; i++) {
      const x = (150 - total) / 2 + i * (w + gap), y = (100 - 80 * scale) / 2;
      symbols += `<g transform="translate(${x.toFixed(1)} ${y.toFixed(1)}) scale(${scale})">${symbolMarkup(card)}</g>`;
    }
    return `<svg class="card" viewBox="0 0 150 100" role="img" aria-label="${SV.cardName(card)}">` +
      `<rect class="face" x="1" y="1" width="148" height="98" rx="9"/>${symbols}</svg>`;
  }

  // ------------------------------------------------------------ photo canvas
  const photo = $("photo");
  const ctx = photo.getContext("2d", { willReadFrequently: true });

  function tracePath(quad, shrinkBy) {
    const cx = quad.reduce((s, p) => s + p[0], 0) / 4, cy = quad.reduce((s, p) => s + p[1], 0) / 4;
    ctx.beginPath();
    quad.forEach((p, i) => {
      const x = cx + (p[0] - cx) * (1 - shrinkBy), y = cy + (p[1] - cy) * (1 - shrinkBy);
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    });
    ctx.closePath();
  }

  function draw() {
    if (!state.image) return;
    const unit = Math.max(photo.width, photo.height) / 1000;
    ctx.putImageData(state.image, 0, 0);
    ctx.lineJoin = "round";

    if (state.selected !== null && state.sets[state.selected]) {
      // spotlight one set: dim everything except its three cards
      const triple = state.sets[state.selected];
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, photo.width, photo.height);
      for (const i of triple) {
        const q = state.cards[i].quad;
        ctx.moveTo(q[0][0], q[0][1]);
        for (let k = 3; k >= 1; k--) ctx.lineTo(q[k][0], q[k][1]);   // opposite winding punches a hole
        ctx.closePath();
      }
      ctx.fillStyle = "rgba(0, 0, 0, 0.62)";
      ctx.fill("evenodd");
      ctx.restore();
      ctx.strokeStyle = HIGHLIGHT;
      ctx.lineWidth = 6 * unit;
      for (const i of triple) { tracePath(state.cards[i].quad, 0); ctx.stroke(); }
      return;
    }

    // nothing highlighted: a thin outline round every card that was found
    ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
    ctx.lineWidth = 3 * unit;
    for (const card of state.cards) { tracePath(card.quad, 0); ctx.stroke(); }
  }

  // ---------------------------------------------------------------- results
  function plural(n, word) { return `${n} ${word}${n === 1 ? "" : "s"}`; }

  function render() {
    state.sets = SV.findSets(state.cards);
    if (state.selected !== null && state.selected >= state.sets.length) state.selected = state.sets.length ? 0 : null;
    const summary = $("summary"), sets = $("sets"), cards = $("cards");

    if (!state.cards.length) {
      summary.innerHTML = "No cards found<small>Try again with the cards on a darker, plain surface, " +
        "all fully in the frame and not touching.</small>";
    } else if (!state.sets.length) {
      summary.innerHTML = `No sets among ${plural(state.cards.length, "card")}<small>` +
        "If the cards below look right, deal three more.</small>";
    } else {
      summary.innerHTML = `${plural(state.sets.length, "set")} among ${plural(state.cards.length, "card")}`;
    }
    const tip = $("sets-box").querySelector("small");
    tip.textContent = state.sets.length > 1 ? "Tap one to show it on the photo" : "Shown on the photo";
    $("sets-box").hidden = !state.sets.length;
    $("cards-box").hidden = !state.cards.length;
    $("set-count").textContent = state.sets.length;
    $("card-count").textContent = state.cards.length;

    sets.innerHTML = "";
    state.sets.forEach((triple, n) => {
      const b = document.createElement("button");
      b.className = "set";
      b.setAttribute("aria-pressed", String(state.selected === n));
      b.setAttribute("aria-label", "Set " + (n + 1) + ": " + triple.map((i) => SV.cardName(state.cards[i])).join(", "));
      b.innerHTML = triple.map((i) => cardSVG(state.cards[i])).join("");
      b.addEventListener("click", () => {
        state.selected = state.selected === n ? null : n;
        render();
        if (state.selected !== null) $("stage").scrollIntoView({ behavior: "smooth", block: "nearest" });
      });
      sets.appendChild(b);
    });

    cards.innerHTML = "";
    state.cards.forEach((card, i) => {
      const b = document.createElement("button");
      b.className = "card-button";
      b.innerHTML = cardSVG(card);
      b.addEventListener("click", () => openEditor(i));
      cards.appendChild(b);
    });
    draw();
  }

  // ----------------------------------------------------------------- editor
  const editor = $("editor");
  let editing = -1;
  const FIELDS = [
    ["number", SV.NUMBERS], ["color", SV.COLORS], ["shading", SV.SHADINGS], ["shape", SV.SHAPES],
  ];

  function openEditor(i) {
    editing = i;
    const card = state.cards[i];
    $("editor-preview").innerHTML = cardSVG(card);
    $("editor-fields").innerHTML = FIELDS.map(([key, options]) =>
      `<fieldset class="field"><legend>${key}</legend>` + options.map((o) =>
        `<label><input type="radio" name="${key}" value="${o}"${card[key] === o ? " checked" : ""}>${o}</label>`
      ).join("") + "</fieldset>").join("");
    editor.showModal();
  }

  $("editor-fields").addEventListener("change", (e) => {
    const card = state.cards[editing];
    if (!card) return;
    card[e.target.name] = e.target.name === "number" ? Number(e.target.value) : e.target.value;
    $("editor-preview").innerHTML = cardSVG(card);
  });
  $("editor-remove").addEventListener("click", () => {
    state.cards.splice(editing, 1);
    editor.close();
  });
  editor.addEventListener("close", () => { state.selected = 0; render(); });

  // tapping a card on the photo opens it too
  photo.addEventListener("click", (e) => {
    const r = photo.getBoundingClientRect();
    // the canvas is letterboxed by object-fit: contain
    const scale = Math.min(r.width / photo.width, r.height / photo.height);
    const x = (e.clientX - r.left - (r.width - photo.width * scale) / 2) / scale;
    const y = (e.clientY - r.top - (r.height - photo.height * scale) / 2) / scale;
    const hit = state.cards.findIndex((c) => inside([x, y], c.quad));
    if (hit >= 0) openEditor(hit);
    else if (state.selected !== null) { state.selected = null; render(); }
  });

  function inside(p, quad) {
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

  // ------------------------------------------------------------- processing
  let worker = null;
  let job = 0;

  /**
   * Run the detector, off the main thread when possible. `getPixels` returns a
   * fresh ImageData each time it is called (the buffer is handed to the worker).
   */
  function detect(getPixels) {
    return new Promise((resolve) => {
      const fallback = () => resolve(SV.detectCards(getPixels()));
      try {
        if (!worker) worker = new Worker("worker.js");
      } catch (err) {
        return setTimeout(fallback, 30);
      }
      const id = ++job;
      const onMessage = (e) => {
        if (e.data.id !== id) return;
        worker.removeEventListener("message", onMessage);
        if (e.data.error) fallback(); else resolve(e.data.cards);
      };
      worker.addEventListener("message", onMessage);
      worker.onerror = () => { worker = null; fallback(); };
      const pixels = getPixels();
      worker.postMessage({ id, data: pixels.data, width: pixels.width, height: pixels.height }, [pixels.data.buffer]);
    });
  }

  async function loadBitmap(source) {
    if (window.createImageBitmap) {
      try {
        return await createImageBitmap(source, { imageOrientation: "from-image" });
      } catch (err) { /* fall through to <img> */ }
    }
    return new Promise((resolve, reject) => {
      const img = new Image();
      const url = URL.createObjectURL(source);
      img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
      img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("could not read that image")); };
      img.src = url;
    });
  }

  async function process(source) {
    $("start").hidden = true;
    $("result").hidden = false;
    $("again").hidden = false;
    state.cards = []; state.sets = []; state.selected = null; state.image = null;
    $("summary").textContent = "";
    $("sets").innerHTML = ""; $("cards").innerHTML = "";
    $("sets-box").hidden = true; $("cards-box").hidden = true;
    $("busy").hidden = false;
    try {
      const bitmap = await loadBitmap(source);
      const bw = bitmap.width || bitmap.naturalWidth, bh = bitmap.height || bitmap.naturalHeight;

      // what you see: a modest copy of the photo
      const shown = Math.min(1, SHOW_MAX / Math.max(bw, bh));
      photo.width = Math.round(bw * shown);
      photo.height = Math.round(bh * shown);
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(bitmap, 0, 0, photo.width, photo.height);
      state.image = ctx.getImageData(0, 0, photo.width, photo.height);
      window.scrollTo(0, 0);

      // what the detector sees: the photo's own pixels (it does its own, more
      // careful, shrinking; a browser's quick resize can smear fine stripes)
      const full = Math.min(1, FULL_MAX / Math.max(bw, bh));
      const big = document.createElement("canvas");
      big.width = Math.round(bw * full);
      big.height = Math.round(bh * full);
      const bigCtx = big.getContext("2d", { willReadFrequently: true });
      bigCtx.imageSmoothingQuality = "high";
      bigCtx.drawImage(bitmap, 0, 0, big.width, big.height);
      if (bitmap.close) bitmap.close();
      const cards = await detect(() => bigCtx.getImageData(0, 0, big.width, big.height));
      const k = photo.width / big.width;
      for (const card of cards) card.quad = card.quad.map((p) => [p[0] * k, p[1] * k]);
      big.width = big.height = 0;      // release the memory promptly
      state.cards = cards;
      state.selected = 0;     // highlight the first set straight away (if there is one)
      render();
    } catch (err) {
      $("summary").innerHTML = "Could not read that photo<small>" + String(err.message || err) + "</small>";
    } finally {
      $("busy").hidden = true;
    }
  }

  // ------------------------------------------------------------------ wiring
  for (const id of ["camera", "library"]) {
    $(id).addEventListener("change", (e) => {
      const file = e.target.files && e.target.files[0];
      e.target.value = "";
      if (file) process(file);
    });
  }
  $("sample").addEventListener("click", async () => {
    const blob = await (await fetch("sample.jpg")).blob();
    process(blob);
  });
  $("again").addEventListener("click", () => {
    $("result").hidden = true;
    $("again").hidden = true;
    $("start").hidden = false;
  });

  // light / dark toggle (the initial theme is set by a script in <head>)
  const MOON = '<path d="M9.6 1.2a6.8 6.8 0 1 0 5.2 9.9A5.6 5.6 0 0 1 9.6 1.2Z"/>';
  const SUN = '<circle cx="8" cy="8" r="3.2"/><path d="M8 0.5v2M8 13.5v2M0.5 8h2M13.5 8h2M2.7 2.7l1.4 1.4M11.9 11.9l1.4 1.4M2.7 13.3l1.4-1.4M11.9 4.1l1.4-1.4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>';
  const toggle = $("theme-toggle");
  function showTheme() {
    const dark = document.documentElement.dataset.theme === "dark";
    toggle.querySelector("svg").innerHTML = dark ? SUN : MOON;
    toggle.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.content = getComputedStyle(document.body).backgroundColor;
  }
  toggle.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) { /* private mode */ }
    showTheme();
  });
  showTheme();

  // service workers only exist on https (and localhost)
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("sw.js").catch(() => {});
  }

  // for tests and the console
  window.SetFinder = { state, process };
})();
