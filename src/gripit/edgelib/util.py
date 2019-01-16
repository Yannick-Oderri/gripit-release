from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import range
from builtins import int
from future import standard_library
standard_library.install_aliases()
import cv2 as cv2
import numpy as np
import random as rand
from skimage import morphology
import copy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import copy


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
      

    # apply Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def normalize_depth(depthimg, colormap=None):
    # Normalize depth image to range 0-255.
    min, max, minloc, maxloc = cv2.minMaxLoc(depthimg)
    adjmap = np.zeros_like(depthimg)
    dst = cv2.convertScaleAbs(depthimg, adjmap, 255 / (max - min), -min)
    if colormap:
        return cv2.applyColorMap(dst, colormap)
    else:
        return dst


def morpho(img):
    kernel = np.ones((7, 7), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=1)
    skel = morphology.skeletonize(dilation > 0)

    return skel

def find_contours(im, mode=cv2.RETR_CCOMP):
    ####IS THERE ANY POINT TO THIS FUNCTION?#######
    
    # im = cv2.imread('circle.png')
    # error: (-215) scn == 3 || scn == 4 in function cv::ipp_cvtColor
    # imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    height = im.shape[0]
    width = im.shape[1]
    blank_image = np.zeros((height, width, 3), np.uint8)
    img = normalize_depth(im, colormap=True)
    im2, contours, hierarchy = cv2.findContours(im, mode, cv2.CHAIN_APPROX_NONE)
    contours = np.squeeze(contours)
    #draw_contours(im, contours)
        # draw_contours(blank_image, contours)
    # cv2.RETR_EXTERNAL cv2.RETR_CCOMP

    contours = np.array(contours)
    contours = np.squeeze(contours)

    return contours

def clahe(img, iter=1):
    # evenly increases the contrast of the entire image
    # ref: http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    for i in range(0, iter):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
    return img


def swap_cols(arr, frm, to):
    arr[:,[frm, to]] = arr[:,[to, frm]]


def squeeze_ndarr(arr):
    #np.squeeze(ndarr)
    temp = []
    for i in range(arr.shape[0]):
        temp.append(np.squeeze(arr[i]))
    np.copyto(arr, np.array(temp))


# Squeezes the array and swaps the columns to match Numpy's col, row ordering
def sqz_contours(contours):
    squeeze_ndarr(contours)
    # ADVANCED SLICING
    for i in range(contours.shape[0]):
        swap_cols(contours[i], 0, 1)


def draw_contours(im, contours):
    height = im.shape[0]
    width = im.shape[1]
    overlay = np.zeros((height, width, 3), np.uint8)
    output = np.zeros((height, width, 3), np.uint8)
    alpha = 0.5

    # cv2.putText(overlay, "ROI Poly: alpha={}".format(alpha), (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    for i in range(len(contours)):
        color = (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
        cv2.drawContours(im, contours, i, color, 1, 8)
        # cv2.imshow("contours", im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    # cv2.addWeighted(overlay, alpha, output, 1 - alpha,
    #                 0, output)
    cv2.imshow("contours", im)
    #cv2.waitKey(0)
    # cv2.destroyAllWindows()


def swap_indices(arr):
    res = []
    for i, e in enumerate(arr):
        res.append([arr[i][1], arr[i][0]])
    return np.array(res)


def create_img(mat):
    blank_image = np.zeros((mat.shape[0], mat.shape[1], 3), np.uint8)
    mask = np.array(mat * 255, dtype=np.uint8)
    masked = np.ma.masked_where(mask <= 0, mask)

    return mask

#Passes in a depth image and turns it into a pointcloud
def depth_to_PC(P):
    # img is the depth image, blank_image is for the pointcloud
    ###Need to edit for mouseY and mouseX(crop it later) -- done
    new_blank_image = copy.deepcopy(P["blank_image"])

    x_val = []
    y_val = []
    z_val = []

    for y_coord in range(len(new_blank_image)):
        for x_coord in range(len(new_blank_image[0])):
            x, y, z = depth_to_3d(x_coord, y_coord, P)
            """z = img[yCoord][xCoord]
                                                x = (xCoord - cx) * z / f
                                                y = (yCoord - cy) * z / f"""
            # print(y, x)
            # print("blank_imageyx", blank_image[yCoord][xCoord])
            new_blank_image[y_coord][x_coord] = (x, y, z)
            """newX.append(int(x))
                                                newY.append(int(y))
                                                newZ.append(int(z))"""
            if (x_coord % 10 == 0 and y_coord % 10 == 0):
                x_val.append(int(x))
                y_val.append(int(y))
                z_val.append(int(z))

    # create3dPlot in plot_3d.py (xVal, yVal, zVal)
    np.save("saveX", x_val)
    np.save("saveY", y_val)
    np.save("saveZ", z_val)

    return new_blank_image


# DepthTo3d
def depth_to_3d(x, y, P):
    cx = P["cx"]
    cy = P["cy"]
    f = P["focal_length"]
    z = copy.deepcopy(P["img"][y][x])
    x = (x - cx) * (z / (f))
    y = (y - cy) * (z / (f))
    return x, y, z

def create3dPlot(xVal, yVal, zVal):
    newFig = plt.figure()
    ax = newFig.add_subplot(111, projection='3d')
    x = xVal
    y = yVal
    z = zVal
    ax.scatter(x, y, z, c="r", marker="o")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.savefig('foo1.png')
    plt.close(newFig)

def fixHoles(img, gradImg, backgroundVal):
    """prox = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),           (0, 1),
                        (1, -1), (1, 0), (1, 1)]
                for y in range(len(img)):
                    for x in range(len(img[0])):
                        if(img[y][x] == 0):
                            total = 0
                            totalNear = 0
                            for eachProx in range(len(prox)):
                                for upTen in range(10):
                                    newY = y + prox[eachProx][0]*upTen
                                    newX = x + prox[eachProx][1]*upTen
                                    if(img[newY][newX] != 0):
                                        total += img[newY][newX]
                                        totalNear += 1
                            img[y][x] = total//totalNear
                return img"""
    for y in range(len(img)):
        for x in range(len(img[0])):
            if(img[y][x] == 0):
                print(gradImg[y][x], backgroundVal, "grad and background")
                gradImg[y][x] = backgroundVal
    return gradImg


def fixHoles2(img):
    prox = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1), (1, 0), (1, 1)]
    for y in range(len(img)):
        for x in range(len(img[0])):
            if(img[y][x] == 0):
                total = 0
                totalNear = 0
                for eachProx in range(len(prox)):
                    for upTen in range(10):
                        newY = y + prox[eachProx][0]*upTen
                        newX = x + prox[eachProx][1]*upTen
                        if(img[newY][newX] != 0):
                            total += img[newY][newX]
                            totalNear += 1
                img[y][x] = total//totalNear
    return img
