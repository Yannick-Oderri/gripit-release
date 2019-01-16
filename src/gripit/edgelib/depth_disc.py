from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import cv2 as cv2
import numpy as np
import gripit.edgelib.util as util



def depth_discont(depth_im, imageModel, sigma=0.33):
    # Depth discontinuity
    # depthimg = util.normalize_depth(depth_im)
    # dimg2 = clahe(depthimg, iter=2)
    #depth_im = util.fixHoles2(depth_im)
    dimg2 = util.auto_canny(depth_im, sigma)
    cv2.imshow("Before skel", dimg2)
    skel2 = util.morpho(dimg2)

    cv2.imshow("discontinuity", dimg2)
    #util.showimg(dimg2, "Discontinuity")
    #cnt2 = util.find_contours(util.create_img(skel2), cv2.RETR_EXTERNAL)

    return skel2