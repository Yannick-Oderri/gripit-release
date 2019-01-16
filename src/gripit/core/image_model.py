
from collections import OrderedDict
from enum import Enum
import cv2
import logging as log
from pathlib import Path
import numpy as np
import copy
import random
import pdb
from PyQt5.QtCore import QPointF as Point2D

import gripit.core.config as config
import gripit.edgelib.util as util
import gripit.edgelib.Line_feat_contours as lfc
import gripit.edgelib.classify_curves as cc
import gripit.edgelib.label_curves as lc
import gripit.edgelib.merge_lines as merge_lines
from gripit.edgelib.line_seg import line_seg
from gripit.edgelib.line_match import line_match
from gripit.edgelib.edge_detect import edge_detect
from gripit.edgelib.edge_detect import create_rgb_img
import gripit.core.ynk_utils as helper
from gripit.core.edge_model import EdgeModel, EdgePair

class CropMethod(Enum):
    USERCROP = 0
    AUTOCROP = 1


class ImageElement(object):


    def __init__(self, _context, cImage, dImage, name):
        self.context = _context
        self.name = name
        self.P = None

        # load rgb image
        self.rgbImage = cImage
        # load depth image
        self.depthImage = dImage

        width, height, channels = self.rgbImage.shape
        # initialize crop window as initial size of image
        self.crop_rect = (0, 0, width, height)

        self.pointCloud = None
        self.lineData = None
        self.auxiliary_images = None
        self.initializeAttribute(_context, config.SCENE_CONFIG)

    def initializeAttribute(self, context, defaultConfig):
        tempDict = {}
        self._imageAttributes = copy.deepcopy(defaultConfig)
        for key in defaultConfig:
            attribute = defaultConfig[key]
            if attribute['type'] == "UI_GROUP":
                for key in attribute['value']:
                    tempDict[key] = attribute['value'][key]

        for key in tempDict:
            tempDict[key]['hidden'] = True
            self._imageAttributes[key] = tempDict[key]

    # Crop Image
    def cropImage(self, left=None, top=None, right=None, bottom=None, cropMethod = CropMethod.USERCROP):
        if cropMethod == CropMethod.AUTOCROP:
            left = self.crop_rect[0]
            top = self.crop_rect[1]
            right = self.crop_rect[2]
            bottom = self.crop_rect[3]

        log.info("Cropping Image: %s" % self.name)
        # Crops image using numpy slicing
        _trgbImageCropped = self.rgbImage[top:bottom, left:right]
        _tdepthImageCropped = self.depthImage[top:bottom, left:right]
        self.rgbImageCropped = _trgbImageCropped
        self.depthImageCropped = _tdepthImageCropped
        _tdepthImageClahe = util.normalize_depth(copy.deepcopy(self.depthImage))
        _tdepthImageClahe = util.clahe(_tdepthImageClahe, iter=2)
        self.depthImageClahed = _tdepthImageClahe[top:bottom, left:right]
        self.crop_rect = (left, top, right, bottom)
        self.setAttribute("crop_rectangle", self.crop_rect)

    def isProcessed(self):
        return self.lineData != None

    def getLineData(self):
        if self.lineData == None:
            P = self.generatePObject()
            # Find depth / curvature discontinuities
            curve_disc, depth_disc, edgelist = edge_detect(P)
            # Generate line segments of contours
            seglist = line_seg(edgelist, tol=P["sgmnt_tolerance"]["value"])
            self.lineData = self._processLineSegments(seglist, edgelist, curve_disc, depth_disc, P)
            return self.lineData
        else:
            return self.lineData

    def _processLineSegments(self, seglist, edgelist, curve_disc, depth_disc, P):
        edgePairList = []
        line_pairs = []
        cntr_pairs = []
        point_group = []
        img = P['img']

        blank_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        blank_image1 = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        results = copy.deepcopy(self.getCroppedRGBImage()) / 2

        # ******* SECTION 2 *******
        # SEGMENT AND LABEL THE CURVATURE LINES (CONVEX/CONCAVE).
        for j in range(seglist.shape[0]):
            # First round of creating line features- creating these features first to use in line merge
            # Line features created:
            line_feature, list_point = lfc.create_line_features(seglist[j], j, edgelist, P)

            # Merge lines that are next to fdeach other and have a similar slope
            line_new, list_point_new, line_merged = merge_lines.merge_lines(line_feature, list_point, P)

            # Classify curves as either discontinuity or curvature
            line_new = cc.classify_curves(curve_disc, depth_disc, line_new, list_point_new, P)

            # Draws curves with different colors according to what kind of discontinuity that line is
            # KEY: Green is depth discontinuity
            # KEY: Blue is curvature discontinuity
            # if settings.dev_mode is True:
            # draw.draw_curve(line_new, j, P)

            # Label curves further
            # Curvature - convex or concave
            # Depth - Right or Left or Doesn't belong to this contour at all
            line_new = lc.label_curves(img, line_new, list_point_new, edgelist[j])

            # Draws curves with different colors according to what kind of discontinuity that line is
            # KEY: Curvature-
            # Convex: Pink
            # Concave: Purple
            # KEY: Depth-
            # Left: Blue
            # Right: Orange
            # Does not belong to this contour: Yellow
            # if settings.dev_mode is True:
            # draw.draw_label(line_new, j, P)

            # START PAIRING THE LINES
            # Delete lines that are concave OR less than the minimum length OR
            # shouldn't be part of that contour (it belongs to the object in front of
            # it or next to it)
            delete_these = np.where(np.logical_or(line_new[:, 12] == 4, line_new[
                                  :, 12] == -1, line_new[:, 4] < P["min_sgmnt_len"]["value"]))
            for idx in range(len(list_point_new)):
                if len(list_point_new[idx]) < 16:
                    delete_these = (np.append(delete_these[0], idx),)
            # delete_these.sort()
            print("Delete These")
            print(delete_these)
            line_final = np.delete(line_new, delete_these, axis=0)
            list_point_final = np.delete(list_point_new, delete_these, axis=0)
            for tline in line_new:
                randC = random.uniform(0, 1)
                randB = random.uniform(0, 1)
                randA = random.uniform(0, 1)
                col = (0, 255, 0)
                if tline[12] == 1:
                    col = (0, 0, 255)
                elif tline[12] == 2:
                    col = (255, 0, 0)
                cv2.line(blank_image, (int(tline[1]), int(tline[0])), (int(tline[3]), int(tline[2])), col, 1, 8) 

            for tline in line_final:        
                randC = random.uniform(0, 1)
                randB = random.uniform(0, 1)
                randA = random.uniform(0, 1)
                col = (0, 255, 0)
                if tline[12] == 1:
                    col = (0, 0, 255)
                elif tline[12] == 2:
                    col = (255, 0, 0)
                cv2.line(blank_image1, (int(tline[1]), int(tline[0])), (int(tline[3]), int(tline[2])), col, 1, 8) 



            # Starts pairing lines that passed minimum requirements
            list_pair, matched_lines, matched_cntrs = line_match(line_final, list_point_final, P)
            depthImage = self.getCroppedDepthImage()
            for k in range(len(matched_lines)):
                edgePair = None
                pairs =  []
                points = []
                col = np.random.rand(3).flatten() * 255
                for m in range(2):
                    pointList = []
                    mc = matched_cntrs[k][m].flatten()
                    for ind in mc:
                        y, x = np.unravel_index(ind, P['img_size'], order='F')
                        if depthImage[y][x] != 0:
                            pointList.append((x,y))                        
                    kwds = {
                        'lineFeatureArray': matched_lines[k][m],
                        'imageModel': P['imageModel'],
                        'pointList': pointList,
                        'ID': "edge-{}-{}-{}".format(j, k, m),
                        'edgeAttributes':{ # This should be added withing edge object... its all defined in lineFeatureArray
                            'edge_clasification': int(matched_lines[k][m][10]),
                            'object_direction': matched_lines[k][m][12],
                            'old_edgePointList': pointList, # This creates
                            'points_shifted': False # have the points been shifted
                        }
                    }
                    edge = EdgeModel(**kwds)
                    edge.setAttribute("edge_clasification", int(matched_lines[k][m][10]))
                    edge.setAttribute("object_direction", matched_lines[k][m][12])
                    edge.setAttribute("old_edgePointList", pointList)
                    if len(pointList) > 10:
                        pairs.append(edge)
                    # print to resulting image
                    cv2.line(results, (int(edge.x1()), int(edge.y1())), (int(edge.x2()), int(edge.y2())), np.array((int(col[0]), int(col[1]), int(col[2]))), 3, cv2.LINE_AA)

                if len(pairs) != 2:
                    continue
                edgePair = EdgePair(P["imageModel"], pairs[0], pairs[1], "ep-{}-{}".format(j, k))
                edgePair.setRenderColor(col)
                ## Shift edge
                P["imageModel"].context.processEdge(P["imageModel"], edgePair)
                edgePairList.append(edgePair)

                # Deprecated
                line_pairs.append((matched_lines[k], "line-{}".format(k), np.random.rand(1, 1, 3).flatten() * 255))
                

            matched_cntrs = np.squeeze(np.array(matched_cntrs))
            # Draws the contour points (for debugging)
            for i in range(matched_cntrs.shape[0]):
                mc = matched_cntrs.flatten()
                y, x = np.unravel_index([mc[i]], P['img_size'], order='F')
                x = np.squeeze(x)
                y = np.squeeze(y)
                cntr_pairs.append([x, y])
                color = np.random.rand(1, 1, 3).flatten()

                for j in range(y.size):
                    x0 = int(x.item(j))
                    y0 = int(y.item(j))
                    point_group.append((x0, y0, color))

            # Draws the pairs that were found
            # Same colors are paired together
            # draw.draw_listpair(matched_lines, final_img) #(fixme) remove this code

        # Debugging images
        curve_disc_img = create_rgb_img(curve_disc)
        depth_disc_img = create_rgb_img(depth_disc)
        self.addAuxiliaryImage("curve_discontinuity", curve_disc_img)
        self.addAuxiliaryImage("depth_discontinuity", depth_disc_img)
        self.addAuxiliaryImage("segments_found", blank_image)
        self.addAuxiliaryImage("segments_used", blank_image1)
        self.addAuxiliaryImage("final_results", results)

        curve_disc_img = cv2.putText(curve_disc_img, "Curve Discontinuity", (20, curve_disc.shape[0] - 20), cv2.FONT_HERSHEY_PLAIN, 1, (255))
        depth_disc_img = cv2.putText(depth_disc_img, "Depth Discontinuity", (20, depth_disc.shape[0] - 20), cv2.FONT_HERSHEY_PLAIN, 1, (255) )
        segment_before_img = cv2.putText(blank_image, "Segments Found", (20, depth_disc.shape[0] - 20), cv2.FONT_HERSHEY_PLAIN, 1, (255) )
        segment_final_img = cv2.putText(blank_image1, "Segments Final", (20, depth_disc.shape[0] - 20), cv2.FONT_HERSHEY_PLAIN, 1, (255) )

        timg = np.hstack((curve_disc_img, depth_disc_img))
        timg2 = np.hstack((segment_before_img, segment_final_img))
        timg = np.vstack((timg, timg2))

        timg = cv2.resize(timg, (curve_disc_img.shape[1], curve_disc_img.shape[0]))


        processed_images = (
            timg, 
            segment_final_img, 
            segment_before_img, 
            curve_disc_img, 
            depth_disc_img,
            )
        cv2.imshow("Segments ", blank_image)
        cv2.imshow("Segments Final", blank_image1)
        return {
            "edge_pairs": edgePairList,
            "countour_points": point_group,
            "segment_list": seglist,
            "processed_images": processed_images
        }


    def addAuxiliaryImage(self, imageName, data):
        if self.auxiliary_images is None:
            self.auxiliary_images = {}

        self.auxiliary_images[imageName] = data

    def saveAuxiliaryImage(self, imageName):
        imgData = None
        try:
            imgData = self.auxiliary_images[imageName]
        except KeyError as err:
            log.warn("Auxiliary image {} not found. \n Process Image First".format(imageName))
            return

        def __saveImage(path, data):
            cv2.imwrite(str(path), data)

        self.context.saveFiletoStore("{}/{}.png".format(self.name, imageName), imgData, __saveImage)

    def getPointCloudFromCrop(self):
        if self.pointCloud is None:
            helper = self.context.getHelperClass()
            self.pointCloud = helper.processPointCloud(self.context, self)
            # self.pointCloud.saveToFile("pc_{}.txt".format(self.name))
        return self.pointCloud

    def getCroppedRGBImage(self):
        """Returns cropped RGB image
        """
        if self.rgbImageCropped is None:
            raise RuntimeError("Image not cropped: %s" % self.name)
        return self.rgbImageCropped

    def getCroppedDepthImage(self, colorMap=None):
        """Returns cropped Depth image
        """
        if self.depthImageCropped is None:
            raise RuntimeError("Image not cropped: %s" % self.name)
        if colorMap is None:
            return self.depthImageCropped
        else:
            return util.normalize_depth(self.depthImageCropped, colorMap)

    def getBaseRGBImage(self):
        return self.rgbImage

    def getBaseDepthImage(self):
        return self.depthImage

    def getCroppingRectangle(self):
        return self.crop_rect

    # Provides Point cloud
    def getPointCloud(self):
        return self.getPointCloudFromCrop()

    def generatePObject(self):
        """
        Generates an key value array which was used for processing prior to the use of class sturctures
        """
        if self.P is None:
            P = {
                "path": Path('./outputImg/'),
                "num_img": 1,  # arbitrary number...
                "img": self.depthImageCropped,
                "img2": self.depthImageClahed,
                "blank_image": np.zeros((self.depthImageCropped.shape[0], self.depthImageCropped.shape[1], 3), np.uint8),
                "height": self.depthImageCropped.shape[0],
                "width": self.depthImageCropped.shape[1],
                "img_size": self.depthImageCropped.shape,
                "mouse_X": (self.crop_rect[0], self.crop_rect[1]),
                "mouse_Y": (self.crop_rect[2], self.crop_rect[3]),
                "cx": round(self.depthImageCropped.shape[0] / 2),
                "cy": round(self.depthImageCropped.shape[1] / 2),
                "cropped_center_x": (self.depthImage.shape[1]/2) - self.crop_rect[0],
                "cropped_center_y": (self.depthImage.shape[0]/2) - self.crop_rect[1],
                "focal_length": self.context.focal_length,
                "thresh": 1,
                "window_size": 10,
                "min_len": 20,
                "min_dist": 10,
                "max_dist": 200,
                "delta_angle": 20,
                "imageModel": self,
                "context": self.context
            }

            self.P = P
            self.P.update(**self._imageAttributes)
        return self.P

    
    def getAttribute(self, key):
        return self._imageAttributes[key]["value"]

    def setAttribute(self, key, value):
        if key in self._imageAttributes:
            self._imageAttributes[key]["value"] = value
        else:
            self._imageAttributes[key] = {"value":value}

    def hasAttribute(self, key):
        return key in self._imageAttributes


    def deleteCache(self):
        del self.pointCloud
        del self.lineData
        self.pointCloud = None
        self.lineData = None

    def getName(self):
        return self.name
