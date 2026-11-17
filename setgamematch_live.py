#!/usr/bin/env python
"""
Live version of setgamematch.py: reads frames from the default camera and
outlines any sets it sees. Press Esc to quit.

Needs a GUI build of OpenCV (pip install opencv-python, not the -headless one).
"""
import sys

import cv2

import setvision as sv
from setgamematch import annotate


def main():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        sys.stderr.write("could not open camera\n")
        return 1
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cards = sv.detect_cards(rgb)
        cv2.imshow("camera", annotate(rgb, cards, sv.find_sets(cards)))
        if cv2.waitKey(1) == 27:
            break
    cam.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
