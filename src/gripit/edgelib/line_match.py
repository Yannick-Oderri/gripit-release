from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
#from check_overlap import check_overlap
from builtins import range
from future import standard_library
standard_library.install_aliases()
from gripit.edgelib.relative_pos import relative_pos
from gripit.edgelib.distance2d import distance2d
import copy
import random as rand
import cv2
import numpy as np
# Written 10/27/2016

# Input: LineInteresting, P(Parameters)
# Output: ListPair

def line_match(LineInteresting, list_point_final, P):
    # Constants
    delta_angle = copy.deepcopy(P["edge_pair_delta_angle"]["value"])
    min_dist = copy.deepcopy(P["edge_pair_min_dist"]["value"])
    max_dist = copy.deepcopy(P["edge_pair_max_dist"]["value"])
    size_ratio = P["edge_pair_len_diff"]["value"]/100.00
    blank_image = copy.deepcopy(P["blank_image"])
    rowsize = LineInteresting.shape[0]
    #Number of lines in this contour

    list_pair = []
    matched_lines = []
    cntr_pairs = []
    #Goes through every line
    for i in range(0, rowsize):

        #Compares it with every other line after it
        for j in range(i+1, rowsize):

            #Checking to make sure the smaller line is at least half the size of the larger line
            if(abs(LineInteresting[i, 4]-LineInteresting[j, 4]) < size_ratio*(max(LineInteresting[j, 4], LineInteresting[i, 4]))):
                
                #Checking to make sure the angles are similar, within the threshold of each other
                if abs(LineInteresting[i, 6] - LineInteresting[j, 6]) <= delta_angle or (abs(LineInteresting[i,6] - LineInteresting[j,6]) >= (180-delta_angle)):

                    #checking to make sure that they are not the same kind of lines
                    #EX: not both discontinuity rights, or discontinuity lefts, or convex
                    if(LineInteresting[i, 12] != LineInteresting[j, 12]):
                        
                        #Checking to make sure they are not too far apart or two close together
                        d = distance2d(LineInteresting[i, :], LineInteresting[j, :])
                        if min_dist < d < max_dist:
                            list_pair.append([i,j])
                            matched_lines.append([LineInteresting[i], LineInteresting[j]])
                            cntr_pairs.append([list_point_final[i], list_point_final[j]])

    return list_pair, matched_lines, cntr_pairs