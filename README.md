# Set, Game, Match
Computer vision for the game of Set: photograph the table, get every set.

**Try it: https://jhofman.github.io/setgamematch/** (works on a phone, and can be added to the home screen).

[Set][1] is a great pattern matching game.
You put down a board of 12 cards, each of which has 4 attributes: shape, number, color, and pattern.
A set is any combination of 3 cards for which each attribute is the same (e.g., all ovals) or different (e.g., an oval, diamond, and squiggle).
The idea is to be fast and to identify sets as quickly as possible.

The [odds][2] that a randomly selected board of 12 cards contains at least one set is about 30 to 1, but it's tough to be certain that no set exists on any given board.

There are two versions of the same detector here:

* `setvision.py`, a Python module (OpenCV + NumPy), with a command-line script
* `docs/`, **the live website**: a static web app that runs entirely in the browser and installs to a phone's home screen. Despite the name, `docs/` is not documentation; it's the folder GitHub Pages serves.

<img src="board_labeled.jpg" width=600px/>

## Python

	pip install -r requirements.txt
	python setgamematch.py board.jpg

prints the cards and the sets, and writes `board_labeled.jpg`.
`setgamematch_live.py` does the same on frames from your machine's camera (it needs `opencv-python` rather than the headless build, and has not been run since the rewrite).

## Website (`docs/`)

The live site at https://jhofman.github.io/setgamematch/ is exactly the contents of `docs/` on `master`.
GitHub Pages is set to "Deploy from a branch: master, /docs", so anything merged into `docs/` on `master` goes live within a minute or two, and anything put in that folder is public.
(`docs/` is the only folder name besides the repository root that Pages can serve from a branch; `docs/.nojekyll` stops Pages from running the files through Jekyll.)

It is plain HTML, CSS and JavaScript: no build step, no server-side code, nothing leaves the phone.
`docs/setvision.js` is a function-for-function port of `setvision.py` with no dependencies.

To try it locally:

	python3 -m http.server 8765 --directory docs

and open http://localhost:8765.

To host it somewhere else as well, copy the folder (e.g. `rsync -av docs/ you@host:/var/www/set/`) and serve it as static files.

It has to be served over **https** for the camera button, offline use and "Add to Home Screen" to work (GitHub Pages does this for you).
On the phone, open the site, then Share → Add to Home Screen (iOS) or the install prompt (Android).

After changing anything under `docs/`, bump `CACHE` in `docs/sw.js` so installed copies drop their old offline cache.

## How it works

1. Shrink the photo and threshold on whiteness (the minimum of R, G, B) at a few levels; keep the blobs that are convex quadrilaterals of a plausible shape.
2. Warp each one to a 300x200 card.
3. Divide out the card's own background colour (a smooth surface fitted to its white parts), which removes colour casts and lighting gradients.
4. Find ink relative to the local white level, fill the outlines, and take connected components as symbols. A candidate that does not look like a Set card (ink in the wrong places, symbols off their slots, shapes unlike any symbol) is dropped.
5. Read the attributes:
   * **number** from which slots the symbols occupy, so a symbol erased by glare is still counted
   * **colour** from the hue of the strongest ink, relative to the white just outside the symbol
   * **shading** from how much ink is inside the outline compared with the outline itself, measured row by row so that fine stripes register; whether the rows alternate breaks ties when glare has faded the stripes
   * **shape** from the filled symbol's solidity and extent

## How well it works

| Photos | Cards read correctly |
| --- | --- |
| 5 board photos and 36 single-card images in this repo (`tests/evaluate.py`) | 88 / 88 |
| The same boards under 18 synthetic degradations: rotation, dim light, colour casts, blur, low resolution, heavy JPEG, noise, perspective (`tests/stress.py`) | 935 / 936 |
| 10 board photos from [nicolashahn/set-solver][5], with glare, shadows, steep angles and clutter (`tests/heldout.py`) | 125 / 126, no false cards |
| All 81 cards of a different deck, cropped, from the same source | 81 / 81 |

The first two rows are photos the detector was developed on. The last two were first run after it passed those, but then guided a round of fixes (faint stripes, shadows, glare, the box being read as a card), so they are no longer strictly unseen.
The one board card still missed is an outline almost entirely erased by glare.
The website gets the same answers as Python on all of these (`tests/js_parity.py` compares the two card by card).

Known limits:

* Cards need a background darker than they are. A white tablecloth will not work.
* Cards that touch or overlap are often missed. In a photo of all 81 cards laid edge to edge it finds 69.
* Strong glare can erase an outline or stripes completely; the app lets you tap a card to correct it.

## Tests

	python tests/evaluate.py                 # labelled photos in this repo
	python tests/stress.py                   # the same, degraded
	python tests/js_parity.py                # JavaScript port vs Python (needs node)
	git clone --depth 1 https://github.com/nicolashahn/set-solver /tmp/set-solver
	python tests/heldout.py /tmp/set-solver/image-data

`tests/margins.py` prints the range of each feature per class, which is what the thresholds in `setvision.py` were set from.
`tests/make_assets.py` regenerates the app icons and sample photo (needs Pillow), and `tests/make_explainer.py` regenerates the figures and chart data behind the "How it works" page (`docs/how.html`).

## History

**2013.** Jake wrote the first version by hand in January 2013: threshold the whole photo, treat bright blobs with holes in them as cards, classify each symbol's colour and shading by matching its average colour against a hand-tuned table, and its shape by how much of its bounding box it fills.
It worked on the photo it was tuned on and not reliably on much else; the lighting only had to change a little.
It was tidied up for GitHub in 2015, and in December 2025 a GitHub Copilot agent ported it to Python 3, but the approach, and its fragility, stayed the same.

**2026.** In November 2026 Jake gave Claude Fable 5.2 (running in Claude Code) this prompt, and went to bed:

> this is an old repo for computer vision to solve a set board. the computer vision part never worked quite right. i'd like you to fix that. first in local python code and then in something that i could easily deploy on my linode as a website. take a look, let me know if you have questions, and after that i'm going to bed and will see what you've done in the morning.

Fable asked three questions first. The answers set the brief: a static site if possible, "nearly flawless" detection but 90–95% acceptable, and

> start classical, train if you need to, use the existing images, you can muck with them or find others to increase the training set if need be. but i'd love if something off the shelf / fine-tuned worked with relatively little data. humans can do it, why can't code?! :)

Roughly how the night went (about an hour of work, from the first commit to the last):

1. **Read the old code and photos**, and found the root problem: every decision depended on absolute colours, so any change in lighting broke it.
2. **Rewrote the pipeline** around flattening each card and dividing out its own white (the steps above), and turned the photos already in the repo into a labelled test: 5 boards and 36 single cards, 88 cards in all. It reached 88 / 88.
3. **Stress-tested it** by degrading those photos on purpose (blur, noise, colour casts, low resolution, perspective) and fixed what broke.
4. **Ported it to JavaScript** for the website, with a test that feeds identical pixels to both versions and requires identical answers, and built the installable web app around it.
5. **Checked it on photos it had never seen**, downloaded from Nicolas Hahn's set-solver. The first run read only 88 of 126 cards correctly, almost all misses being faint stripes read as empty. That led to the row-by-row stripe measure, measuring ink against the local white (shadows), reading the number from symbol positions (glare), and rejecting white rectangles that aren't cards (the box). Afterwards: 125 / 126, and 81 / 81 for the other project's full deck.

No model was trained: once a card is flattened and its lighting removed, each attribute is one or two numbers with wide gaps between the classes, so a few thresholds read from the labelled cards were enough.

The next day, the restyle to match [tri-times](https://github.com/jhofman/tri-times), the "How it works" page and the deployment were done in a follow-up session with Claude Opus 5.5.

## Credits

Thanks to Nicolas Hahn ([nicolashahn/set-solver][5]) for the photos that made it possible to test this on more than one deck and table:

* `setgame1.jpeg`, `SetCards.jpeg` and `SetCards2.jpeg` in this repo are copies of photos from that project.
* The held-out evaluation (`tests/heldout.py`) and part of the charts on the "How it works" page use that project's `image-data` folder; those images are read from a clone of it, not copied here.

[1]: http://en.wikipedia.org/wiki/Set_(game)
[2]: http://norvig.com/SET.html
[5]: https://github.com/nicolashahn/set-solver
