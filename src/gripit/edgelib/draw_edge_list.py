from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
# -------------------------------------------------------------------------------
# Name:        drawedgelist
# Purpose:     Plots pixels in edge lists.
# -------------------------------------------------------------------------------
#
# Usage:    drawedgelist(edgelist, rowscols, thickness)
#
# Arguments:
#   edgelist    - Array of arrays containing edgelists
#   rowscols    - Optional 2 element vector [rows cols] specifying the size
#                 of the image from which the edges were detected. Otherwise
#                 this defaults to the bounds of the line segment points.
#
# Author: Jordan Zhu
#


from builtins import range
from builtins import str
from future import standard_library
standard_library.install_aliases()
import sys
import cv2
import numpy as np
import random as rand
import copy

def draw_edge_list(edgelist, P):
    blank_image = copy.deepcopy(P["blank_image"])
    num_img = P["num_img"]
    path = "outputImg/"
    
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


        
    
    #cv2.imshow("Edgelist", blank_image)