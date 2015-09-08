# Set, Game, Match
A computer vision hack for the Set game.

[Set][1] is a great pattern matching game.
You put down a board of 12 cards, each of which has 4 attributes: shape, number, color, and pattern.
A set is any combination of 3 cards for which each attribute is the same (e.g., all ovals) or different (e.g., an oval, diamond, and squiggle).
The idea is to be fast and to identify sets as quickly as possible.

The [odds][2] that a randomly selected board of 12 cards contains at least one set is about 30 to 1, but it's tough to be certain that no set exists on any given board.

This is a hack that uses (overly) simple computer vision methods to identify cards on a board and the patterns on them, and then identifies any sets that are present.
It relies on [OpenCV 2.0][3] for vision routines, which can be [easily installed][4] with Homebrew.

Here's an example:

	$ python setgamematch.py board.jpg

`setgamematch_live.py` does the same, but uses your machine's camera in real-time.

<img src="board_labeled.jpg" width=600px/>

[1]: http://en.wikipedia.org/wiki/Set_(game)
[2]: http://norvig.com/SET.html
[3]: http://opencv.org
[4]: http://www.mobileway.net/2015/02/14/install-opencv-for-python-on-mac-os-x/