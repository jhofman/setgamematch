#!/usr/bin/env python

import sys
import os
import cv
import cv2
import numpy as np
from collections import defaultdict


shadings = ['open', 'striped', 'solid']
colors = ['green', 'red', 'purple']
shapes = ['diamond', 'squiggle', 'oval']


class SetCard:
    def __init__(self, v=None, cnt=None):
        # stored as num, col, shading, shape
        self.v = np.array(v)
        self.cnt = cnt

    def __repr__(self):
        num = 3 if self.v[0] == 0 else self.v[0]
        s = '' if num == 1 else 's'
        return str(num) + " " + shadings[self.v[2]] + " " + colors[self.v[1]] + " " + shapes[self.v[3]] + s

    def code(self):
        return "".join(map(str, self.v))


class SetHand:
    def __init__(self, cards=[]):
        #self.cards = [v for v in cards if isinstance(v, SetCard) else SetCard(v)]
        self.cards = []
        for card in cards:
            if not isinstance(card, SetCard):
                card = SetCard(card)
            self.cards.append(card)

    def add(self, card):
        self.cards.append(v if isinstance(v, SetCard) else SetCard(v))

    def find_sets(self):
        # todo: change to dict to reference SetCard (for contour drawing)
        codes = set([card.code() for card in self.cards])

        found = set()
        sets = []
        for i, card1 in enumerate(self.cards):
            for j, card2 in enumerate(self.cards):
                if i < j:
                    card3 = SetCard(missing_card(card1, card2))

                    if card3.code() in codes:
                        s = [card1.code(), card2.code(), card3.code()]
                        s.sort()
                        s = tuple(s)
                        if s not in found:
                            found.add(s)
                            sets.append([card1, card2, card3])
                            #print card1, card2, card3

                        #print ", ".join( map(str, (card1, card2, card3)) )
                        #print card1, card2
        return sets

def is_set_match(card1, card2, card3):
    return ((card1.v + card2.v + card3.v) % 3 == 0).prod() == 1

def missing_card(card1, card2):
    return (np.array([0,0,0,0]) - ((card1.v + card2.v) % 3)) % 3


if __name__=='__main__':
    if len(sys.argv) != 2:
        sys.stderr.write('usage: %s <img>\n' % sys.argv[0])
        sys.exit(1)

    symbol_colors = np.array([[130, 165, 165], # open green
                              [60, 130, 80], # open green (hack)
                              [50, 80, 190],   # open red
                              [130, 143, 164], # open purple
                              [110, 150, 140], # striped green
                              [100, 120, 180], # striped red
                              [110, 115, 130], # striped purple
                              [75, 160, 15],   # solid green
                              [50, 30, 180],   # solid red
                              [70, 35, 60]     # solid purple
                              ])
    symbol_codes = np.array([ [0, 0], [0, 0], [0, 1], [0, 2],
                               [1, 0], [1, 1], [1, 2],
                               [2, 0], [2, 1], [2, 2]
                               ])
    symbol_extents = np.array([0.5, 0.75, 0.87])

    # read input image
    infile = sys.argv[1]
    im = cv2.imread(infile)

    # convert to grayscale
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # threshold to binary image
    # todo: change to otsu threshold
    ret,thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #0)

    # extract contours and regions
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]

    # identify cards as large areas with children but no parents
    # and other regions as candidate symbols
    card_ids = set()
    symbol_ids = set()
    for i, cnt in enumerate(contours):
        # todo: adaptive size threshold
        if cv2.contourArea(cnt) >= im.shape[0]*im.shape[1] / 5000: 
            if hierarchy[i][2] > 0 and hierarchy[i][3] == -1:
                card_ids.add(i)
            else:
                symbol_ids.add(i)

    # show original image
    cv2.imshow('display', im)

    # identify all symbol regions with their parent card
    cards = defaultdict(dict)
    for i in symbol_ids:
        parent_id = hierarchy[i][3]
 
        if parent_id in card_ids:
            # mask image
            mask = np.zeros(imgray.shape, np.uint8)
            cv2.drawContours(im, [contours[i]], -1, (0,255,0), 3)

            # compute BGR mean to identify color and shading
            mean = map(int, cv2.mean(im, mask = mask)[:3])
            err = np.abs(symbol_colors - mean).sum(1)
            shading, color = symbol_codes[err == err.min()][0]
            err_symbol = err

            # compute ratio of symbol area to bounding rectangle
            # to identify shape
            area = cv2.contourArea(contours[i])
            bx, by, bw, bh = cv2.boundingRect(contours[i])
            extent = area / (bw * bh)
            err = np.abs(symbol_extents - extent)
            shape =  (err == err.min()).nonzero()[0][0]
            err_shape = err

            cards[parent_id][i] = [color, shading, shape]

            #cv2.putText(im, repr(cards[parent_id][i]), (bx, int(by+bh/10.0)), cv2.FONT_HERSHEY_PLAIN, 2.0, (255,255,255), 5)


    # construct and label cards and add to hand
    hand = SetHand()
    for c, symbols in cards.iteritems():
        # build card object
        v = [len(symbols) % 3,] + symbols.values()[0]
        card = SetCard(v, contours[c])

        # add card to hand
        hand.add(card)

        # draw contour around card
        cv2.drawContours(im, [contours[c]], -1, (255,0,0), 3)

        # label card on original image
        bx, by, bw, bh = cv2.boundingRect(contours[c])
        cv2.putText(im, repr(card), (bx, int(by+bh/10.0)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,0), 5)


    # print sets
    for i, s in enumerate(hand.find_sets()):
        print "set #" + str(i+1) + ":",
        desc = ", ".join(map(str, s))
        cv2.putText(im, desc, (100, 100*(1+i)), cv2.FONT_HERSHEY_PLAIN, 4.0, (0,0,0), 5)
        print desc

    # save labeled image
    outfile = '%s_labeled%s' % os.path.splitext(infile)
    cv.SaveImage(outfile, cv.fromarray(im))
