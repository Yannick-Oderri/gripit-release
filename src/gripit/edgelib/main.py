from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
# deprivated

from builtins import round
from builtins import range
from builtins import int
from future import standard_library
standard_library.install_aliases()
import cv2
import numpy as np
import scipy.io as sio
import settings
import matplotlib.pyplot as plt
import Line_feat_contours as lfc
import classify_curves as cc
import label_curves as lc
import merge_lines as merge_lines
import util as util
from edge_detect import edge_detect
from line_match import line_match
from line_seg import line_seg
from crop_image import crop_image
import draw_img as draw
import random as rand
import copy
import logging as log

np.set_printoptions(threshold=np.nan)

if __name__ == '__main__':
    context = settings.EdgeProcessingDetectContext.initializeContext(settings.ExecutionMode.DEVELOPMENT_1)
    settings.init()

    settings.dev_mode = False

    for num_img in [4]:
        #Crops the image according to the user's mouse clicks
        #First click is top left, second click is bottom right
        mouse_X, mouse_Y = crop_image(num_img)


        # Read in depth image, -1 means w/ alpha channel.
        # This keeps in the holes with no depth data as just black.

        depth_im = 'img/learn%d.png'%num_img
        old_img = cv2.imread(depth_im, -1)

        #crops the depth image
        img = old_img[mouse_Y[0]:mouse_Y[1], mouse_X[0]:mouse_X[1]]

        final_img = util.normalize_depth(img, colormap=cv2.COLORMAP_BONE)
        cy = round(img.shape[0] / 2)
        cx = round(img.shape[1] / 2)

        #USE COPY.DEEPCOPY if you don't want to edit variables passed in
        #AKA THE IMAGES, DON'T DO THIS
        P = {"path": 'outputImg\\',
        "num_img": num_img,
        "img": img,
        "height": img.shape[0],
        "width":img.shape[1],
        "img_size":img.shape,
        "mouse_X": mouse_X,
        "mouse_Y": mouse_Y,
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

        #Values that depend on values in P
        P2 = {"blank_image": np.zeros((P["height"], P["width"], 3), np.uint8)}

        #adds all these new values to P
        P.update(P2)

        #Creates point cloud map
        point_cloud = util.depth_to_PC(P)
        if context.ShowPointCloudWnd:
            cv2.imshow("Point cloud", point_cloud)

    
        # Open a copy of the depth image
        # to change the contrast on the full-sized image
        img2 = cv2.imread(depth_im, -1)
        img2 = util.normalize_depth(img2)
        img2 = util.clahe(img2, iter=2)
        
        # crops the image
        img2 = img2[mouse_Y[0]:mouse_Y[1], mouse_X[0]:mouse_X[1]]
        P["img2"] = img2

        # *********************************** SECTION 1 *****************************************

        # FIND DEPTH / CURVATURE DISCONTINUITIES.
        curve_disc, depth_disc, edgelist = edge_detect(P)

        #CREATES LINE SEGMENTS
        seglist = line_seg(edgelist, tol=5)
        if context.ShowEdgeListWnd:
            draw.draw_edge_list(seglist, P)

        line_pairs = []
        cntr_pairs = []

        img_size = copy.deepcopy(P["img_size"])
        height = img_size[0]
        width = img_size[1]
        blank_im = np.zeros((height, width, 3), np.uint8)
        print("img size", img_size)


        window_size = 3
        def roipoly(src, poly):
            mask = np.zeros_like(src, dtype=np.uint8)
            win = util.swap_indices(poly)
            cv2.fillConvexPoly(mask, win, 255)  # Create the ROI
            res = src * mask
            # cv2.imshow("roi", res)
            # cv2.waitKey(0)
            return res


        def get_orientation(line, window_size):
            dy = abs(line[0] - line[2])
            dx = abs(line[1] - line[3])
            # Vertical or horizontal line test
            if dy > dx or dy == dx:
                pt1 = [line[0], line[1] - window_size]
                pt2 = [line[0], line[1] + window_size]
                pt3 = [line[2], line[3] - window_size]
                pt4 = [line[2], line[3] + window_size]
                return pt1, pt2, pt3, pt4
            else:
                pt1 = [line[0] - window_size, line[1]]
                pt2 = [line[0] + window_size, line[1]]
                pt3 = [line[2] - window_size, line[3]]
                pt4 = [line[2] + window_size, line[3]]
                return pt1, pt2, pt3, pt4


        def get_ordering(pt1, pt2, pt3, pt4):
            temp1 = np.linalg.norm(np.subtract((np.add(pt1, pt3) / 2.0), (np.add(pt2, pt4) / 2.0)))
            temp2 = np.linalg.norm(np.subtract((np.add(pt1, pt4) / 2.0), (np.add(pt2, pt3) / 2.0)))
            res = np.array([pt1, pt3, pt4, pt2]) if temp1 > temp2 else np.array([pt1, pt4, pt3, pt2])
            return [[int(i) for i in pt] for pt in res]


        def grad_dir(img):
            # compute x and y derivatives
            # OpenCV's Sobel operator gives better results than numpy gradient
            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)
            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)

            # calculate gradient direction angles
            # phase needs 64-bit input
            angle = cv2.phase(sobelx, sobely)

            # truncates number
            gradir = np.fix(180 + angle)

            return gradir


        # ******* SECTION 2 *******
        # SEGMENT AND LABEL THE CURVATURE LINES (CONVEX/CONCAVE).
        for j in range(seglist.shape[0]):
            #First round of creating line features- creating these features first to use in line merge
            #Line features created:
            line_feature, list_point = lfc.create_line_features(seglist[j], j, edgelist, P)

            #Merge lines that are next to each other and have a similar slope
            line_new, list_point_new, line_merged = merge_lines.merge_lines(line_feature, list_point, P)

            #Draw the contour with the merged lines
            if settings.dev_mode is True:
                draw.draw_merged(line_new, P)

            #Classify curves as either discontinuity or curvature
            line_new = cc.classify_curves(curve_disc, depth_disc, line_new, list_point_new, P)
            
            #Draws curves with different colors according to what kind of discontinuity that line is
            #KEY: Green is depth discontinuity
            #KEY: Blue is curvature discontinuity
            if settings.dev_mode is True:
                draw.draw_curve(line_new, j, P)

            #Label curves further
            #Curvature - convex or concave
            #Depth - Right or Left or Doesn't belong to this contour at all 
            line_new = lc.label_curves(img, line_new, list_point_new, edgelist[j])
    
            #Draws curves with different colors according to what kind of discontinuity that line is
            #KEY: Curvature-
            #Convex: Pink
            #Concave: Purple
            #KEY: Depth-
            #Left: Blue
            #Right: Orange
            #Does not belong to this contour: Yellow
            if settings.dev_mode is True:
                draw.draw_label(line_new, j, P)

            # START PAIRING THE LINES
            # Delete lines that are concave OR less than the minimum length OR shouldn't be part of that contour (it belongs to the object in front of it or next to it)
            delete_these = np.where(np.logical_or(line_new[:, 12] == 4, line_new[:, 12] == -1, line_new[:, 4] < P["min_len"]))
            line_final = np.delete(line_new, delete_these, axis=0)
            list_point_final = np.delete(list_point_new, delete_these, axis=0)
            
            #Starts pairing lines that passed minimum requirements
            list_pair, matched_lines, matched_cntrs = line_match(line_final, list_point_final, P)


            cv2.imwrite("test_im.png", img)
            for k in range(len(matched_lines)):
                line_pairs.append(matched_lines[k])
                for m in range(2):
                    if line_pairs[k][m][10] == 13:
                        del matched_cntrs[k][m]     ## (fixme) This keeps crashing

                        pt1, pt2, pt3, pt4 = get_orientation(line_pairs[k][m], window_size)
                        win = np.array(get_ordering(pt1, pt2, pt3, pt4))
                        print("window: ", win)
                        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)
                        abs_sobel64f = np.absolute(sobely)
                        sobel_8u = np.uint8(abs_sobel64f)
                        roi = roipoly(sobel_8u, win)
                        cv2.imshow("Line window", img)
                        roi2 = roi[win[0][0]:win[2][0], win[0][1]:win[2][1]]
                        print(roi2.shape)
                        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)
                        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)

                        cv2.imshow("Image", roi2)
                        print(roi2[0][0])
                        print(np.where(roi2 == 255))

                        # mag = []
                        # for i in range(roi2.shape[1]):
                        #     for j in range(roi2.shape[0] - 1):

            # np.save("save_matched_lines", np.array(matched_lines))
            # np.save("save_contours", np.array(matched_cntrs))


            # print("matched lines", matched_lines)
            # print("Matched Shape", matched_cntrs)
            # print("matched cntrs", matched_cntrs)

            matched_cntrs = np.squeeze(np.array(matched_cntrs))

            # Draws the contour points (for debugging)
            for i in range(matched_cntrs.shape[0]):
                mc = matched_cntrs.flatten()
                y, x = np.unravel_index([mc[i]], img_size, order='F')

                # x = np.squeeze(x)
                # y = np.squeeze(y)
                # print("x ", x)
                # print("y ", y)
                cntr_pairs.append([x, y])
                for j in range(len(y)):
                    x0 = int(x[j])
                    y0 = int(y[j])
                    # print(x0, ", ", y0)
                    color = (0, 255, 0)
                    cv2.line(final_img, (x0, y0), (x0, y0), color, thickness=1)  # (fixme) disable countour printing

            #Draws the pairs that were found
            #Same colors are paired together
            draw.draw_listpair(matched_lines, final_img) #(fixme) remove this code


        # Contour pairs (debugging)
        if context.ShowContourPairs:
            cv2.imshow("Contour pairs", blank_im)
        
        ## --- Print Line Pairs Code
        line_pairs_colors = []
        for i in range(len(line_pairs)):
            line_pairs_colors.append((rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255)))
        
        def _onLineSelectChanged(val):
            lineselect = 0
            timg = copy.deepcopy(final_img)
            if val == 0:
                log.info("Displaying all lines\n")
                lineselect = range(len(line_pairs))
            else:
                log.info("Displaying line: %d" % val)
                lineselect = [val-1]

            for i in lineselect:
                color = line_pairs_colors[i]
                # line in the list of lines
                line1 = line_pairs[i][0]
                line2 = line_pairs[i][1]
                # print(line1, line2)
                x1 = int(line1[1])
                y1 = int(line1[0])
                x2 = int(line1[3])
                y2 = int(line1[2])

                x3 = int(line2[1])
                y3 = int(line2[0])
                x4 = int(line2[3])
                y4 = int(line2[2])

                cv2.line(timg, (x1, y1), (x2, y2), color, 2)
                cv2.line(timg, (x3, y3), (x4, y4), color, 2) 
            cv2.imshow("Line Pairs", final_img)


        #Final drawing of all the pairs that were found
        if context.ShowLinePairsWnd:
            _onLineSelectChanged(0)
            cv2.createTrackbar("Line Select", "Line Pairs", 0, len(line_pairs), _onLineSelectChanged)

        ## ---- End Line Pair Printing
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # ******* SECTION 3 *******


        # Gets them from the point cloud
        cntr_pc = []
        for i in range(0, len(cntr_pairs)):
            # line1 = point_cloud[cntr_pairs[i][1][:], cntr_pairs[i][0][:]]
            # line2 = point_cloud[cntr_pairs[i + 1][1][:], cntr_pairs[i + 1][0][:]]
            # cntr_pc.append([line1, line2])
            # cntr_pc.append(util.depth_to_3d(cntr_pairs[i][0][:], cntr_pairs[i][1][:], P))
            for j in range(cntr_pairs[i][0].shape[0]):
                cntr_pc.append(util.depth_to_3d(cntr_pairs[i][0][j], cntr_pairs[i][1][j], P))

        # print("cntr pc: ", cntr_pc)
        np.save("saveCntr", np.array(cntr_pc))
        np.save("savePairs", np.array(line_pairs))

        print(line_pairs)
        pairs_3d = []
        for i in range(len(line_pairs)):
            line1 = line_pairs[i][0]
            line2 = line_pairs[i][1]

            pairs_3d.append(util.depth_to_3d(int(line1[1]), int(line1[0]), P))
            pairs_3d.append(util.depth_to_3d(int(line1[3]), int(line1[2]), P))
            pairs_3d.append(util.depth_to_3d(int(line2[1]), int(line2[0]), P))
            pairs_3d.append(util.depth_to_3d(int(line2[3]), int(line2[2]), P))
            # pairs_3d.append(point_cloud[int(line1[0])][int(line1[1])])
            # pairs_3d.append(point_cloud[int(line1[2])][int(line1[3])])
            # pairs_3d.append(point_cloud[int(line2[0])][int(line2[1])])
            # pairs_3d.append(point_cloud[int(line2[2])][int(line2[3])])
        # print("Pairs 3d", pairs_3d)
        np.save("save_pairs", np.array(pairs_3d))

        # for i in range(len(line_pairs)):
        #     if line_pairs[i][10] == 13:
        #         del

