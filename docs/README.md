# This folder is the live website

Everything in `docs/` on `master` is published by GitHub Pages at
https://jhofman.github.io/setgamematch/ (Settings → Pages → Deploy from a branch: master, /docs).
It is the Set Finder web app and its "How it works" page, not documentation.

* `index.html`, `app.js`, `style.css`: the app
* `setvision.js`: the card detector, a port of `../setvision.py`
* `worker.js`: runs the detector off the main thread
* `sw.js`, `manifest.webmanifest`, `icons/`: offline use and "Add to Home Screen"
* `how.html`, `how.js`, `how/`: the "How it works" page and its figures
* `.nojekyll`: tells Pages to serve the files as they are

Don't put anything here that shouldn't be public. After changing files, bump `CACHE` in `sw.js`
so installed copies pick up the new version.
