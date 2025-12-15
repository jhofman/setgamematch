#!/usr/bin/env python

import sys
import os
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
        self.cards.append(card if isinstance(card, SetCard) else SetCard(card))

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
    return ((card1.v + card2.v + card3.v) % 3 == 0).all()

def missing_card(card1, card2):
    return (np.array([0,0,0,0]) - ((card1.v + card2.v) % 3)) % 3


if __name__=='__main__':
    if len(sys.argv) != 2:
        sys.stderr.write('usage: %s <img>\n' % sys.argv[0])
        sys.exit(1)

    # Color lookup table: BGR values for different symbol colors and shadings
    # Format: [shading_index, color_index] where:
    #   shading: 0=open, 1=striped, 2=solid
    #   color: 0=green, 1=red, 2=purple
    symbol_colors = np.array([[130, 165, 165], # open green
                              [60, 130, 80],    # open green (alternate)
                              [50, 80, 190],    # open red
                              [130, 143, 164],  # open purple
                              [110, 150, 140],  # striped green
                              [100, 120, 180],  # striped red
                              [110, 115, 130],  # striped purple
                              [75, 160, 15],    # solid green
                              [50, 30, 180],    # solid red
                              [70, 35, 60]      # solid purple
                              ])
    symbol_codes = np.array([[0, 0], [0, 0], [0, 1], [0, 2],
                             [1, 0], [1, 1], [1, 2],
                             [2, 0], [2, 1], [2, 2]
                             ])
    
    # Shape detection using extent (ratio of contour area to bounding rectangle area)
    # Diamond: ~0.5, Squiggle: ~0.75, Oval: ~0.87
    symbol_extents = np.array([0.5, 0.75, 0.87])

    # read input image
    infile = sys.argv[1]
    im = cv2.imread(infile)
    
    if im is None:
        sys.stderr.write('Error: Could not read image file: %s\n' % infile)
        sys.exit(1)

    # convert to grayscale
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # threshold to binary image using Otsu's method
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # extract contours and regions
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]

    # identify cards as large areas with children but no parents
    # Use adaptive threshold based on image size
    img_area = im.shape[0] * im.shape[1]
    card_min_area = img_area / 100  # Cards should be at least 1% of image
    symbol_min_area = img_area / 5000  # Symbols should be at least 0.02% of image
    
    card_ids = set()
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area >= card_min_area: 
            if hierarchy[i][2] > 0 and hierarchy[i][3] == -1:
                card_ids.add(i)
    
    if len(card_ids) == 0:
        sys.stderr.write('Warning: No cards detected in image\n')
        sys.exit(0)

    # identify all symbol regions with their parent card
    # symbols are direct children of cards that are large enough
    cards = defaultdict(dict)
    for card_id in card_ids:
        # Get all direct children of this card
        child_id = hierarchy[card_id][2]  # first child
        while child_id != -1:
            area = cv2.contourArea(contours[child_id])
            # Symbols should be large enough to filter out noise
            if area >= symbol_min_area:
                # create mask for this symbol
                mask = np.zeros(imgray.shape, np.uint8)
                cv2.drawContours(mask, [contours[child_id]], -1, 255, -1)

                # compute BGR mean to identify color and shading
                mean = list(map(int, cv2.mean(im, mask=mask)[:3]))
                err = np.abs(symbol_colors - mean).sum(1)
                best_match_idx = np.argmin(err)
                shading, color = symbol_codes[best_match_idx]

                # compute ratio of symbol area to bounding rectangle
                # to identify shape
                bx, by, bw, bh = cv2.boundingRect(contours[child_id])
                # Avoid division by zero
                if bw > 0 and bh > 0:
                    extent = area / float(bw * bh)
                    err = np.abs(symbol_extents - extent)
                    shape = np.argmin(err)
                else:
                    shape = 0  # default to diamond if we can't compute extent

                cards[card_id][child_id] = [color, shading, shape]
            
            # Move to next sibling
            child_id = hierarchy[child_id][0]


    # construct and label cards and add to hand
    hand = SetHand()
    cards_with_no_symbols = []
    
    for c, symbols in cards.items():
        # build card object
        symbol_values = list(symbols.values())
        if len(symbol_values) > 0:
            # Calculate number: 1, 2, or 3 (not 0)
            num_symbols = len(symbols)
            if num_symbols > 3:
                num_symbols = 3  # Cap at 3 for Set rules
            v = [num_symbols % 3,] + symbol_values[0]
            card = SetCard(v, contours[c])

            # add card to hand
            hand.add(card)

            # draw contour around card
            cv2.drawContours(im, [contours[c]], -1, (255, 0, 0), 3)

            # label card on original image
            bx, by, bw, bh = cv2.boundingRect(contours[c])
            cv2.putText(im, repr(card), (bx, int(by + bh / 10.0)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 0), 5)
        else:
            cards_with_no_symbols.append(c)
    
    if cards_with_no_symbols:
        sys.stderr.write(f'Warning: {len(cards_with_no_symbols)} card(s) detected with no symbols\n')

    # print sets
    for i, s in enumerate(hand.find_sets()):
        print("set #" + str(i + 1) + ":", end=" ")
        desc = ", ".join(map(str, s))
        cv2.putText(im, desc, (100, 100 * (1 + i)), cv2.FONT_HERSHEY_PLAIN, 4.0, (0, 0, 0), 5)
        print(desc)

    # save labeled image
    outfile = '%s_labeled%s' % os.path.splitext(infile)
    cv2.imwrite(outfile, im)
