from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import round
from builtins import int
from builtins import range
from future import standard_library
standard_library.install_aliases()
import cv2
import numpy as np
import gripit.edgelib.draw_img as di
import gripit.edgelib.util as util


def roipoly(src, poly):
    mask = np.zeros_like(src, dtype=np.uint8)
    win = util.swap_indices(poly).astype(int)

    cv2.fillConvexPoly(mask, win, 255)  # Create the ROI
    return mask

def vertical_line(line):
    # [y ; x-ts]
    return [line[0], line[1] - window_size], \
           [line[0], line[1] + window_size], \
           [line[2], line[3] - window_size], \
           [line[2], line[3] + window_size]


def horizontal_line(line):
    # [y-ts ; x]
    return [line[0] - window_size, line[1]], \
           [line[0] + window_size, line[1]], \
           [line[2] - window_size, line[3]], \
           [line[2] + window_size, line[3]]

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


def get_ordering(pt1, pt2, pt3, pt4):
    temp1 = np.linalg.norm(np.subtract((np.add(pt1, pt3) / 2.0), (np.add(pt2, pt4) / 2.0)))
    temp2 = np.linalg.norm(np.subtract((np.add(pt1, pt4) / 2.0), (np.add(pt2, pt3) / 2.0)))
    res = np.array([pt1, pt3, pt4, pt2]) if temp1 > temp2 else np.array([pt1, pt4, pt3, pt2])
    return [[int(i) for i in pt] for pt in res]

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

def getGradientMagnitude(im):
    # Get magnitude of gradient for given image.
    ddepth = cv2.CV_64F
    dx = cv2.Sobel(im, ddepth, 1, 0)
    dy = cv2.Sobel(im, ddepth, 0, 1)
    dxabs = cv2.convertScaleAbs(dx)
    dyabs = cv2.convertScaleAbs(dy)
    mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
    return mag

pairs = np.load("savePairs.npy")
contours = np.load("saveCntr.npy")

im = cv2.imread("test_im.png", -1)
mag = getGradientMagnitude(im)
mag = (((mag - mag.min()) / (mag.max() - mag.min())) * 255.9).astype(np.uint8)
blur = cv2.bilateralFilter(mag, 9, 25, 25)
mag = blur

cy = round(im.shape[0] / 2)
cx = round(im.shape[1] / 2)
P = {"path": './outputImg/',
     "img": im,
     "height": im.shape[0],
     "width": im.shape[1],
     "img_size": im.shape,
     "cx": cx,
     "cy": cy,
     "focal_length": 300,
     "thresh": 20,
     "window_size": 10,
     "min_len": 20,
     "min_dist": 10,
     "max_dist": 200,
     "delta_angle": 20
     }

# Values that depend on values in P
P2 = {"blank_image": np.zeros((P["height"], P["width"], 3), np.uint8)}

cv2.imshow("mag", mag)

# di.draw_listpair(pairs, mag)
# cv2.imshow("Line Pairs", mag)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

pc = []

window_size = 8
for k in range(len(pairs)):
    for m in range(2):
        if pairs[k][m][10] == 13:
            line = pairs[k][m]
            pt1, pt2, pt3, pt4, startpt, endpt = get_orientation(line)
            win_p, win_n = create_windows(pt1, pt2, pt3, pt4, startpt, endpt)
            roi_p = roipoly(mag, win_p)
            roi_n = roipoly(mag, win_n)

            if line[12] == 1:
                window = roi_n
            else:
                window = roi_p

            window2 = window * mag

            cv2.imshow("window", window2)
            cv2.waitKey(0)

            points = np.where(window == 255)
            magn = []

            if line[11] == 2:
                # Horizontal
                for i in range(0, points[0].shape[0], window_size * 2):
                    done = False
                    for j in range(i, i + window_size * 2 - 1):
                        # print("i = ", i, "j = ", j)
                        if j + 1 == points[0].shape[0]:
                            magn.append(0)
                            done = True
                            break
                        magn.append(mag[points[0][j]][points[1][j]] - mag[points[0][j + 1]][points[1][j + 1]])
                    if done == True:
                        break
                    magn.append(0)

                for i in range(len(magn)):
                    print(magn[i], end=" ")
                    if i % (window_size * 2) == 0:
                        print("\n")

                mask = np.zeros(len(magn), dtype=bool)
                for i in range(0, len(magn), window_size * 2):
                    col = np.array(magn[i:i + window_size * 2 - 1])
                    maxes = np.where(col == col.max())[0][-1]
                    mask[i + maxes] = True

                new_pts = np.array([points[0][np.logical_and(points[0], mask)], points[1][np.logical_and(points[1], mask)]])
                print(new_pts)

                for i in range(new_pts[0].shape[0]):
                    # print(new_pts[1][i], ",", new_pts[0][i])
                    pc.append(util.depth_to_3d(new_pts[1][i], new_pts[0][i], P))

mX = []
mY = []
mZ = []
for i in range(len(pc)):
    mX.append(pc[i][0])
    mY.append(pc[i][1])
    mZ.append(pc[i][2])
np.save("save_mX", mX)
np.save("save_mY", mY)
np.save("save_mZ", mZ)

cv2.destroyAllWindows()