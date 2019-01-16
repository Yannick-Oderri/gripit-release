from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import cv2
import numpy as np
import gripit.edgelib.util as util

global window_size
global buffer_zone


def vertical_line(line):
    line[11] = 1
    # [y ; x-ts]
    return [line[0] - buffer_zone, line[1] - window_size - buffer_zone], \
           [line[0] + buffer_zone, line[1] + window_size + buffer_zone], \
           [line[2] - buffer_zone, line[3] - window_size - buffer_zone], \
           [line[2] + buffer_zone, line[3] + window_size + buffer_zone]


def horizontal_line(line):
    line[11] = 2
    # [y-ts ; x]
    return [line[0] - window_size - buffer_zone, line[1] - buffer_zone], \
           [line[0] + window_size + buffer_zone, line[1] + buffer_zone], \
           [line[2] - window_size - buffer_zone, line[3] - buffer_zone], \
           [line[2] + window_size + buffer_zone, line[3] + buffer_zone]

def get_orientation(line):
    startpt = [line[0], line[1]]
    endpt = [line[2], line[3]]
    dy = abs(line[0] - line[2])
    dx = abs(line[1] - line[3])

    if dy > dx or dy == dx:
        pt1, pt2, pt3, pt4 = vertical_line(line)
    else:
        pt1, pt2, pt3, pt4 = horizontal_line(line)
    return pt1, pt2, pt3, pt4, startpt, endpt


def create_windows(pt1, pt2, pt3, pt4, startpt, endpt):
    temp1 = np.linalg.norm(np.subtract((np.add(pt1, pt3) / 2.0), (np.add(pt2, pt4) / 2.0)))
    temp2 = np.linalg.norm(np.subtract((np.add(pt1, pt4) / 2.0), (np.add(pt2, pt3) / 2.0)))
    if temp1 > temp2:
        win_p = [startpt, endpt, pt4, pt2]
        win_n = [pt1, pt3, endpt, startpt]
    else:
        win_p = [startpt, pt4, endpt, pt2]
        win_n = [pt1, endpt, pt3, startpt]
    return win_p, win_n


def roipoly(src, poly):
    mask = np.zeros_like(src, dtype=np.uint8)
    win = util.swap_indices(poly).astype(int)

    cv2.fillConvexPoly(mask, win, 255)  # Create the ROI
    return mask


def mask_mean(src, mask):
    val_mask = src * mask
    mask_size = cv2.countNonZero(mask)
    num_nonzero = cv2.countNonZero(val_mask)
    if num_nonzero == 0 or num_nonzero / mask_size < 0.05:
        return 100000
    else:
        return sum(src[np.nonzero(val_mask)]) / num_nonzero


def line_mean(img, y, x):
    points = img[y, x]
    num_nonzero = cv2.countNonZero(points)
    # print(num_nonzero, "non zero", np.sum(points), "sum")
    return np.sum(points) / num_nonzero


def remove_points(lp, roi):
    line_mask = np.invert(np.in1d(lp, roi))
    # print(lp, "lp")
    # print(line_mask, "line mask")
    roi = roi[:][line_mask]
    cv2.imshow("roi after", roi)

def get_mean(src, lp, win_p, win_n):
    mask_p = roipoly(src, win_p)
    mask_n = roipoly(src, win_n)
    # remove_points(lp, mask_p)
    mean_p = mask_mean(src, mask_p)
    mean_n = mask_mean(src, mask_n)
    return mean_p, mean_n

# concave/convex of a curvature
def label_convexity(lp_curr, mean_p, mean_n):
    # mean_win = (mask_p + mask_n) / 2

    # print("| LP mean:", lp_curr, "| mean_P:", mean_p, "| mean_N:", mean_n)
    # if lp_curr <= mean_p and lp_curr <= mean_n:
    #     if mean_p >= mean_n:
    #         return 31
    #     elif mean_n >= mean_p:
    #         return 32
    #     else:
    #         return -1
    #     # return 31 if mask_p >= mask_n and points_p >= points_ else 32
    # elif lp_curr > mean_p or lp_curr > mean_n:
    #     return 4
    return 3 if lp_curr <= mean_p and lp_curr <= mean_n else 4


# obj on left/right side of discontinuity
def label_pose(mean_p, mean_n, count_p, count_n):
    # print("| mean_P:", mean_p, "| mean_N:", mean_n, "p count", count_p, "n count", count_n)
    if mean_p >= mean_n and count_n >= count_p:
        return 1
    elif mean_n >= mean_p and count_p >= count_n:
        return 2
    else:
        return -1
    # return 1 if mask_p >= mask_n else 2


# This removes boundary lines that aren't part of the shape
def remove_lines(src, contour, win_p, win_n):
    mask_p = roipoly(src, win_p)
    mask_n = roipoly(src, win_n)

    """
    tp = np.nonzero(mask_p)
    tn = np.nonzero(mask_n)

    pixels_p = np.squeeze(np.dstack((tp[0], tp[1])))
    pixels_n = np.squeeze(np.dstack((tn[0], tn[1])))
    """

    mask = np.zeros((src.shape[1], src.shape[0]), dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)

    # drawwwwwwwwwwwwwwwwing
    # contour2 = util.swap_indices(contour)
    # cv2.drawContours(mask2, [contour2], 0, 255, -1)
    #
    # output = mask2
    # overlay = output.copy()
    #
    # alpha = 0.3
    # win_p = util.swap_indices(win_p).astype(int)
    # win_n = util.swap_indices(win_n).astype(int)
    # cv2.fillConvexPoly(overlay, win_p,
    #                    (0, 255, 0))
    # cv2.fillConvexPoly(overlay, win_n,
    #                    (0, 0, 255))
    # cv2.addWeighted(overlay, alpha, output, 1 - alpha,
    #                 0, output)
    #
    # cv2.imshow("contour + windows", output)
    # cv2.waitKey(0)

    # tc = np.nonzero(mask)
    # contour = np.squeeze(np.dstack((tc[0], tc[1])))

    im_c = np.transpose(mask)

    # cv2.imshow("mask_p", mask_p)
    # cv2.imshow("mask_n", mask_n)
    # cv2.imshow("Contour!", im_c)
    # print(mask_p.shape, "mask shape")
    # print(im_c.shape, "contour im")
    count_p = np.count_nonzero(np.logical_and(mask_p, im_c))
    count_n = np.count_nonzero(np.logical_and(mask_n, im_c))

    """points_p = 0
    points_n = 0
    for z in range(len(pixels_p)):
        if (contour == pixels_p[z]).all(1).any():
            points_p += 1

    for z in range(len(pixels_n)):
        if (contour == pixels_n[z]).all(1).any():
            points_n += 1
    """

    return count_p, count_n



def label_curves(src, list_lines, list_point, contour):
    global window_size
    window_size = 5
    global buffer_zone
    buffer_zone = 0
    # append results of test to col 12
    col_label = np.zeros((list_lines.shape[0], 2))
    list_lines = np.hstack((list_lines, col_label))

    for i, line in enumerate(list_lines):
        pt1, pt2, pt3, pt4, startpt, endpt = get_orientation(line)
        win_p, win_n = create_windows(pt1, pt2, pt3, pt4, startpt, endpt)

        count_p, count_n = remove_lines(src, contour, win_p, win_n)
        mean_p, mean_n = get_mean(src, list_point[i], win_p, win_n)

        if line[10] == 12:
            y, x = np.unravel_index([list_point[i]], src.shape, order='F')
            mean_lp = line_mean(src, y, x)
            label = label_convexity(mean_lp, mean_p, mean_n)
            # print(label, "curv label")
        elif line[10] == 13 or line[10] == 14:
            label = label_pose(mean_p, mean_n, count_p, count_n)
            # print(label, "disc label")
        else:
            label = 0
        line[12] = label

    return list_lines
