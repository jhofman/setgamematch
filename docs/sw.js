/*
 * Service worker: keeps a copy of the app so it opens offline once installed.
 * Online, files always come from the network (and refresh the copy), so a new
 * deploy shows up immediately; the cached copy is only used when offline.
 */
const CACHE = "set-finder-v5";
const FILES = [
  "./", "index.html", "style.css", "app.js", "setvision.js", "worker.js", "manifest.webmanifest",
  "how.html", "how.js", "how/data.json",
  "how/board.jpg", "how/whiteness.jpg", "how/threshold.png", "how/outlines.jpg",
  "how/card-warped.jpg", "how/card-normalised.jpg", "how/card-ink.png", "how/card-symbols.png",
  "how/shading-open.jpg", "how/shading-striped.jpg", "how/shading-solid.jpg",
  "how/shape-diamond.png", "how/shape-oval.png", "how/shape-squiggle.png",
  "sample.jpg", "icons/icon-192.png", "icons/icon-512.png", "icons/apple-touch-icon.png", "icons/favicon-32.png",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(FILES)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin) return;
  e.respondWith(
    caches.open(CACHE).then((cache) =>
      fetch(e.request)
        .then((res) => { if (res.ok) cache.put(e.request, res.clone()); return res; })
        .catch(() => cache.match(e.request, { ignoreSearch: true }))
    )
  );
});
