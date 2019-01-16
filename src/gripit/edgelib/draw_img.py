from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import range
from builtins import int
from builtins import str
from future import standard_library
standard_library.install_aliases()
import sys
import cv2
import numpy as np
import random as rand
import edgelib.util as util
import settings
import copy


def draw_edge_list(edgelist, P):
    blank_image = copy.deepcopy(P["blank_image"])
    num_img = P["num_img"]
    path = "outputImg\\"
    
    ##Goes through every contour of edge list
    for i in range(len(edgelist)):
        print(len(edgelist))
        blank_image = copy.deepcopy(blank_image)
        #Goes through every edge of that contours
        for j in range(len(edgelist[i])-1):
            # Draws the line segments.
            color = (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
            cv2.line(blank_image, (edgelist[i][j][0], edgelist[i][j][1]), (edgelist[i][j+1][0], edgelist[i][j+1][1]), color, thickness=1)
        
        cv2.imshow("Edgelist%d" %i, blank_image)
        cv2.imwrite(str(path) + "Edgelist%d%d.png"% (num_img, i), blank_image)



def draw_merged(line_feature, P):
    img = copy.deepcopy(P["blank_image"])
    # print(line_feature[0])
    for i, e in enumerate(line_feature):
        x1 = int(e[1])
        y1 = int(e[0])
        x2 = int(e[3])
        y2 = int(e[2])
        color = (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
        cv2.line(img, (x1, y1), (x2, y2), color, 2)
    cv2.imshow("merged lines", img)


def draw_label(list_lines, i, P):
    num_img = copy.deepcopy(P["num_img"])
    img = copy.deepcopy(P["blank_image"])
    for i, e in enumerate(list_lines):
        if e[12] == 1:
            # Teal
            color = (255, 255, 0)
        # right of disc
        elif e[12] == 2:
            # Orange
            color = (0, 165, 255)
        # convex/left of curv
        elif e[12] == 3:
            # Pink
            color = (194, 89, 254)
        # convex/right
        elif e[12] == 32:
            # lavender
            color = (255, 182, 193)
            # color = (194, 89, 254)
        # concave of curv
        elif e[12] == 4:
            # Purple
            color = (128, 0, 128)
        # Remove
        elif e[12] == -1:
            # Yellow
            color = (0, 255, 255)
        else:
            # Red is a 'hole' line
            color = (0, 0, 255)
        x1 = int(e[1])
        y1 = int(e[0])
        x2 = int(e[3])
        y2 = int(e[2])
        cv2.line(img, (x1, y1), (x2, y2), color, 2)
    # cv2.imshow("Labels", img)
    cv2.imshow("Label%d" % num_img, img)
    cv2.imwrite(str(P["path"]) + "Label%d%d.png" % (num_img, i), img)


def draw_curve(list_lines, i, P):
    img = copy.deepcopy(P["blank_image"])
    num_img = copy.deepcopy(P["num_img"])
    path = copy.deepcopy(P["path"])
    for i, e in enumerate(list_lines):
        if e[10] == 12:
            # Blue is a curvature
            color = (255, 0, 0)
        elif e[10] == 13:
            # Green is a discontinuity
            color = (0, 255, 0)
        else:
            # Red is a 'hole' line
            color = (0, 0, 255)
        x1 = int(e[1])
        y1 = int(e[0])
        x2 = int(e[3])
        y2 = int(e[2])
        cv2.line(img, (x1, y1), (x2, y2), color, 2)
    # cv2.imshow("curvatures", img)
    cv2.imshow("Curvature%d" % num_img, img)
    cv2.imwrite(str(path) + "Curvature%d%d.png" % (num_img, i), img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def draw_listpair(list_pair, img):
    # blank_image = normalize_depth(img, colormap=True)

    for i in range(len(list_pair)):
        color = (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
        # line in the list of lines
        line1 = list_pair[i][0]
        line2 = list_pair[i][1]
        # print(line1, line2)
        x1 = int(line1[1])
        y1 = int(line1[0])
        x2 = int(line1[3])
        y2 = int(line1[2])

        x3 = int(line2[1])
        y3 = int(line2[0])
        x4 = int(line2[3])
        y4 = int(line2[2])

        cv2.line(img, (x1, y1), (x2, y2), color, 2)
        cv2.line(img, (x3, y3), (x4, y4), color, 2)


    # cv2.namedWindow('Line features', cv2.WINDOW_NORMAL)
    if settings.dev_mode is True:
        cv2.imshow('CONTOUR PAIRS', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
