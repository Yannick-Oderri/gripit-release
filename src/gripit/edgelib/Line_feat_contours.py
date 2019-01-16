from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from builtins import next
from builtins import round
from future import standard_library
standard_library.install_aliases()
from math import sqrt, atan, degrees
import numpy as np
import cv2
import copy


def calc_inf(y2, y1, x2, x1):
    return float('inf') if (y2 - y1) > 0 else float('-inf')


# This finds the points on the contour
# between the line segment
def find_star(x, y, idx, ListEdges):
    # print('x', x, 'y', y, ListEdges[idx])
    # print(ListEdges[idx][:, 0])
    sty = np.where(ListEdges[idx][:, 0] == y)
    stx = np.where(ListEdges[idx][:, 1] == x)
    # print(sty, stx)
    # Get the first element of the single-item set
    return next(iter(set(sty[0]).intersection(stx[0])))


def get_lin_index(x1, y1, img_size):
    return np.ravel_multi_index((y1, x1), img_size, order='F')

def create_line_features(ListSegments, idx, ListEdges, P):
    img_size = copy.deepcopy(P["img_size"])
    c0 = 0
    LineFeature = []
    ListPoint = []
    imageModel = P['imageModel']

    for j in range(ListSegments.shape[0] - 1):
        # print('i', i, 'j', j)
        # print('curr', ListSegments[j])
        x1, y1 = ListSegments[j].astype(int)
        x2, y2 = ListSegments[j + 1].astype(int)
        # print('x1', x1, 'y1', y1, 'x2', x2, 'y2', y2)

        # finds the positive angle of the line to the horizontal
        slope = round((y2 - y1) / (x2 - x1), 4) if ((x2 - x1) != 0) else calc_inf(y2, y1, x2, x1)
        # linear indices
        lin_ind1 = get_lin_index(x1, y1, img_size)
        lin_ind2 = get_lin_index(x2, y2, img_size)
        linelen = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        # tangent
        alpha = degrees(atan(-slope))
        alpha = 180 + alpha if alpha < 0 else alpha
        # print(alpha)

        LineFeature.append([y1, x1, y2, x2, linelen, slope, alpha, c0, lin_ind1, lin_ind2])
        c0 += 1

        a = find_star(x1, y1, idx, ListEdges)
        b = find_star(x2, y2, idx, ListEdges)
        ListPoint.append(ListEdges[idx][a:b + 1])

        # show the contour points and the line
        # height = img_size[0]
        # width = img_size[1]
        # blank_im = np.zeros((height, width, 3), np.uint8)
        # print(len(ListPoint[-1]), "ListPoint len")
        # for l,e in enumerate(ListPoint[-1]):
        #     x = int(e[1])
        #     y = int(e[0])
        #     color = (0, 255, 0)
        #     cv2.line(blank_im, (x, y), (x, y), color, thickness=1)
        # # print("x1", x1, "y1", y1, "x2", x2, "y2", y2)
        # blank_im2 = np.zeros((height, width, 3), np.uint8)
        # cv2.line(blank_im2, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
        # cv2.imshow("Current Line", blank_im2)
        # cv2.imshow("List Edges", blank_im)
        # cv2.waitKey(0)

        if LineFeature[c0 - 2][8: 10] == [lin_ind1, lin_ind2] and c0 > 2:
            del (LineFeature[c0 - 1])
            del (ListPoint[c0 - 1])
            # print('Duplicate removed')
            c0 -= 1

    len_lp = len(ListPoint)
    LPP = []
    for cnt in range(len_lp):
        LPP.append(np.ravel_multi_index((ListPoint[cnt][:, 0], ListPoint[cnt][:, 1]), img_size, order='F'))

    return np.array(LineFeature), np.array(LPP)