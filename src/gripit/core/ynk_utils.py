from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import open
from builtins import range
from builtins import dict
from builtins import int
from future import standard_library
standard_library.install_aliases()
from builtins import object
from gripit.core.edge_model import EdgeAttributes, EdgeModel, EdgePair
from gripit.core.point_cloud_model import PointCloudModel
from gripit.core.point_cloud_render_element import SceneElement
import pyqtgraph.opengl as gl
from PyQt5.QtCore import QPointF as Point2D
from PyQt5.QtCore import QLineF
from PyQt5 import QtGui
import math
import cv2 as cv2
import numpy as np
import random as rand
from skimage import morphology
import copy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import copy
import logging as log
import gripit.edgelib.util
import scipy
from skimage.measure import LineModel, ransac
from gripit.edgelib.plane_model_nd import PlaneModelND
from scipy.spatial import distance
import os
import gripit.deps.pypcd.pypcd as pcd


def processFace(imageModel, edgePair):
    depthImage = imageModel.getCroppedDepthImage()
    pointCloud = imageModel.getPointCloudFromCrop().getPointCloudData()

    pointsOfInterest = []
    for m in range(2):
        edge = edgePair[m]
        edgePoints = edge.getEdgePointCloudIndexes()
        for point in edgePoints:
            index = point[1] * depthImage.shape[1] + point[0] % depthImage.shape[1]
            pointCloudPoint = pointCloud[index]
            pointsOfInterest.append(pointCloudPoint)

    if len(pointsOfInterest) < 3:
        log.info("Not Enough Points to process face.")

    model_robustP, inliersP = ransac(np.asarray(pointsOfInterest).squeeze(), PlaneModelND,  min_samples=3, residual_threshold=1, max_trials=20)
    return (model_robustP.params[0], model_robustP.params[1], model_robustP.eigenvectors)



def shiftLineOnObject(imageModel, edgeModel):
    """
    Shift edge onto object.
    Imagemodel: Working Image
    EdgeModel: Current Edge that is being shift
    """
    if edgeModel.getAttribute("edge_clasification") != EdgeAttributes.EDGE_TYPE_DD1:
        log.info("Edge Object {} has a edge classification of {} will not be shfited:".format(edgeModel.getID(), edgeModel.getAttribute("edge_clasification")))
        return -1

    if edgeModel.getAttribute("object_direction") == EdgeAttributes.OBJECT_CENTERED:
        log.info("Edge Object {} is centered: will not be shifted.".format(edgeModel.getID()))
        return 0

    if edgeModel.getAttribute("points_shifted") == True:
        log.info("Edge Object {} is already shifted.".format(edgeModel.getID()))
        return 0


    gradientBorder = 1
    depthImage = imageModel.getCroppedDepthImage()
    # Get Line Orientation
    startPos = (edgeModel.x1(), edgeModel.y1())
    endPos = (edgeModel.x2(), edgeModel.y2())
    edgeLength = edgeModel.length()
    pointList = edgeModel.getEdgePointCloudIndexes()
    pointCloud = imageModel.getPointCloudFromCrop()


    perpendicularSampleLength = 25
    perpendicularVector = edgeModel.normalVector().unitVector()

    if edgeModel.getAttribute("object_direction") == 2:
        perpendicularVector.setAngle(perpendicularVector.angle() + 180)
    else:
        perpendicularVector.setAngle(perpendicularVector.angle() + 180)
        # perpendicularVector.setP2(Point2D(-1*perpendicularVector.dx(), -1*perpendicularVector.dy()))
    
    perpendicularVector1 = QLineF(0, 0, perpendicularVector.dx() * perpendicularSampleLength, 
        perpendicularVector.dy() * perpendicularSampleLength)


    gradientData = []
    renderGradient = []

    # append start and end Positons to be shifted also
    # pointList.append(startPos)
    # pointList.append(endPos)

    # Obtain a sample values of all pixels on the line and add the array to gradientdata
    initialValues = 5
    for i in  range(len(pointList)):
        point = pointList[i]
        sampleLine = QLineF(perpendicularVector1)             
        sampleLine.translate(point[0] - perpendicularVector.dx()*initialValues, point[1] - perpendicularVector.dy()*initialValues)
        sampleData = sampleLineFromImage(depthImage, sampleLine, perpendicularSampleLength)
        sampleData = fixLineSample(sampleData)

        gradientData.append(sampleData)
    ## Shift values
    newPointList = []
    for index in range(len(gradientData)):
        x, y = pointList[index]
        lineSample = gradientData[index]
        grad = np.diff(lineSample)

        # print("Sample data\n")
        # print(gradientData[index])
        # print("Gradient\n")
        # print(grad)
        # print("\n")
        absoluteGrad = abs(grad)
        absoluteGrad = np.gradient(absoluteGrad)        
        renderGradient.append(np.diff(absoluteGrad)/np.amax(np.diff(absoluteGrad)))

        deltaIndex = np.argmin(absoluteGrad) - initialValues + 3

        # maxDelta = 0
        # deltaIndex = 5
        # thresHold = 200        
        # for i in range(1, len(gradientData[index])):
        #     delta = abs(gradientData[index][i] - gradientData[index][i - 1])
        #     if delta > thresHold and delta > maxDelta:
        #         maxDelta = delta
        #         deltaIndex = i
        if deltaIndex > 0:
            x = int(x + perpendicularVector.dx() * deltaIndex)
            y = int(y + perpendicularVector.dy() * deltaIndex)
        finalIndex = (x, y)
        newPointList.append(finalIndex)

    # edgeModel.setAttribute("old_edgePointList", pointList)
    # pointListSize = len(newPointList) - 2
    edgeModel.setAttribute("perpendicularVector", perpendicularVector)
    edgeModel.setAttribute("perpendicularVectorOffset", initialValues)
    edgeModel.setAttribute("line_shift_gradient", renderGradient)
    edgeModel.setAttribute("oldPointList", pointList)
    edgeModel.setEdgePoint(newPointList)
    edgeModel.setAttribute("points_shifted", True)


def fixLineSample(sampleData):
    minimumDistance = 100

    if sampleData[0] < minimumDistance:
        for j in range(1, len(sampleData)):
            if sampleData[j] > minimumDistance:
                sampleData[0] = sampleData[j]
                break

    for i in range(len(sampleData)-1):
        if sampleData[i] <= minimumDistance:
            sampleData[i] = sampleData[i - 1]

    return sampleData

def fitLinetoEdgeModel(imageModel, edgeModel):
    pointList = np.array(edgeModel.getPointCloudPoints())
    # print("{} - PointList Length {}".format(edgeModel.getID(), len(pointList)))
    sampleNumber = int(len(pointList) * .20)
    if sampleNumber > 20:
        sampleNumber = 20    
    direction = None

    try:
        model_robust, inliers = ransac(pointList, LineModel, min_samples=2,
                               residual_threshold=1, max_trials=1000)
        origin, direction = model_robust.params

        startPos = origin
    except ValueError as err:
        log.warn("Exception Thrown for model robust again!!!")
        direction = None
    
    pointList = edgeModel.getEdgePointCloudIndexes()
    startPoint = getKNearestPoint(pointList[0:sampleNumber], imageModel.getPointCloudFromCrop())
    endPoint = getKNearestPoint(pointList[-1*sampleNumber:-1], imageModel.getPointCloudFromCrop())
    startPoint = imageModel.getPointCloudFromCrop().getPointfromXYCoordinate(startPoint[0], startPoint[1])
    endPoint = imageModel.getPointCloudFromCrop().getPointfromXYCoordinate(endPoint[0], endPoint[1])

    dst = distance.euclidean(startPoint,endPoint)
    if direction is not None:
        if np.dot(endPoint - startPoint, direction) < 0:
            direction = direction * -1

        endPos = startPoint + (direction * dst)
    else:
        endPos = endPoint

    edgeModel.setAttribute("startPos3D", startPoint)
    edgeModel.setAttribute("endPos3D", endPos)
    # edgeModel.setP1(Point2D(startPos[0], startPos[1]))
    # edgeModel.setP2(Point2D(endPos[0], endPos[1]))


def getKNearestPoint(sampleGroup, pointCloud):
    thresHold = 4
    # throwaway 
    data = np.empty((0,3))
    for sample in sampleGroup:
        data = np.vstack((data, np.array(pointCloud.getPointfromXYCoordinate(sample[0], sample[1]))))
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    t_remove = abs(data - mean) < (1.8 * std)
    t_remove_2 = []

    for i in range(len(t_remove)):
        row = t_remove[i]
        if False in row:
            t_remove_2.append(i) 

    data = np.delete(data,t_remove_2, axis=0)

    newSampleGroup = np.delete(sampleGroup, t_remove_2, axis=0)
    mean = np.mean(data, axis=0)
    index = 0
    if len(newSampleGroup) > 2:
        index = scipy.spatial.KDTree(data).query(mean)[1]
        return newSampleGroup[index]

    return sampleGroup[0]

    
def sampleLineFromImage(cvImage, sampleLine, interpolationSize):
    x = np.linspace(sampleLine.x1(), sampleLine.x2(), interpolationSize)
    y = np.linspace(sampleLine.y1(), sampleLine.y2(), interpolationSize)

    # Extract values along line
    lineValue = scipy.ndimage.map_coordinates(cvImage, np.vstack((y,x)))

    return np.array(lineValue, dtype=float)


def processPointCloud(context, imageModel):
    depthImg = imageModel.getCroppedDepthImage()
    pointCloudData = np.zeros((len(depthImg)*len(depthImg[0]), 3))
    for y_coord in range(len(depthImg)):
        for x_coord in range(len(depthImg[0])):
            x, y, z, = depthToPointCloud(context, x_coord, y_coord, imageModel)
            pointCloudData[(y_coord*len(depthImg[1]))+x_coord%len(depthImg[1])] = (x, y, z)

    return PointCloudModel(pointCloudData=pointCloudData, imageModel=imageModel, context=context)



def depthToPointCloud(context, x_cord, y_cord, imageModel):
    depthImg = imageModel.getCroppedDepthImage()
    scaleFactor = 20
    cx = len(depthImg[0])/2
    cy = len(depthImg)/2
    f = 550 #context.focal_length
    z = depthImg[y_cord][x_cord]
    x = ((x_cord - cx) * (z / (f)))/scaleFactor
    y = ((y_cord - cy) * (z / (f)))/scaleFactor
    z = ((-1 * z)/scaleFactor) + 50

    point = QtGui.QVector3D(x, y, z)

    tr = QtGui.QMatrix4x4()
    tr.translate( 0.0, 0.0, 0.0)
    tr.rotate(50, 1, 0, 0)
    # center = params['center']
    # tr.translate(-center.x(), -center.y(), -center.z())

    inv, result = tr.inverted()

    point = inv * (point)
    return -point.x(), point.y(), point.z()

def ddepthtTo3DPoint(context, x_cord, y_cord, imageModel):
    depthImg = imageModel.getCroppedDepthImage()
    params = SceneElement.getSceneParameters(imageModel)


    coords = "xyz"
    cameraPos = []
    # center = Point3D(0, 0, 0)
    for i in range(len(coords)):
        value = imageModel.getAttribute("camera_location_{}".format(coords[i]))
        cameraPos.append(value)

    cx = len(depthImg[0])/2
    cy = len(depthImg)/2

    focalLength = params["focal_length"]
    scaleFactor = focalLength*9
    z = depthImg[y_cord][x_cord]/25


    x = (x_cord-cx)*(z/scaleFactor) 
    y = -1*(y_cord-cy)*(z/scaleFactor)
    z = -z

    # x = math.sin((fov/2)*math.pi/180) * z_cord * 20 #nearClip/z_cord * proj_x
    # y = math.sin((fov/2)*math.pi/180) * z_cord * 20 #nearClip/z_cord * proj_y
    # z = -z_cord

    point = QtGui.QVector3D(x, y, z)

    tr = QtGui.QMatrix4x4()
    tr.translate( 0.0, 0.0, -params['distance'])
    tr.rotate(params['elevation']-90, 1, 0, 0)
    tr.rotate(params['azimuth']+90, 0, 0, -1)
    # center = params['center']
    # tr.translate(-center.x(), -center.y(), -center.z())

    inv, result = tr.inverted()

    point = inv * (point)
    return point.x(), point.y(), point.z()


def depthtTo3DPoint(context, x_cord, y_cord, imageModel):
    depthImg = imageModel.getCroppedDepthImage()
    params = SceneElement.getSceneParameters(imageModel)
    nMax = np.amax(depthImg)

    coords = "xyz"
    cameraPos = []
    # center = Point3D(0, 0, 0)
    for i in range(len(coords)):
        value = imageModel.getAttribute("camera_location_{}".format(coords[i]))
        cameraPos.append(value)

    cx = len(depthImg[0])/2
    cy = len(depthImg)/2

    focalLength = params["focal_length"]
    scaleFactor = focalLength
    z = depthImg[y_cord][x_cord]
    z = (z / nMax) * params['distance']    


    x = (x_cord-cx)*(z/scaleFactor) 
    y = -1*(y_cord-cy)*(z/scaleFactor)
    z = -z

    # x = math.sin((fov/2)*math.pi/180) * z_cord * 20 #nearClip/z_cord * proj_x
    # y = math.sin((fov/2)*math.pi/180) * z_cord * 20 #nearClip/z_cord * proj_y
    # z = -z_cord

    point = QtGui.QVector3D(x, y, z)

    tr = QtGui.QMatrix4x4()
    tr.translate( 0.0, 0.0, -params['distance'])
    tr.rotate(params['elevation']-90, 1, 0, 0)
    tr.rotate(params['azimuth']+90, 0, 0, -1)
    # center = params['center']
    # tr.translate(-center.x(), -center.y(), -center.z())

    inv, result = tr.inverted()

    point = inv * (point)
    return point.x(), point.y(), point.z()

def pointToImg(context, point, imageModel):
    scaleFactor = 40
    depthImg = imageModel.getCroppedDepthImage()
    params = SceneElement.getSceneParameters(imageModel)

    pos = QVector3D(point[0], point[1], point[2])

    coords = "xyz"
    cameraPos = []
    # center = Point3D(0, 0, 0)
    for i in range(len(coords)):
        value = imageModel.getAttribute("camera_location_{}".format(coords[i]))
        cameraPos.append(value)

    cx = len(depthImg[0])/2
    cy = len(depthImg)/2

    focalLength = params["focal_length"]
    z = depthImg[y_cord][x_cord]


    x = (x_cord-cx)#*(z/focalLength) #((x_cord - cx) * (z / (focalLength)))
    y = -1*(y_cord-cx)#*(z/focalLength) #((y_cord - cy) * (z / (focalLength)))
    z = -1*((z))/(scaleFactor/3)

    # x = math.sin((fov/2)*math.pi/180) * z_cord * 20 #nearClip/z_cord * proj_x
    # y = math.sin((fov/2)*math.pi/180) * z_cord * 20 #nearClip/z_cord * proj_y
    # z = -z_cord

    point = QtGui.QVector3D(x, y, z)


    tr = QtGui.QMatrix4x4()
    tr.translate( 0.0, 0.0, -params['distance'])
    tr.rotate(params['elevation']-90, 1, 0, 0)
    tr.rotate(params['azimuth']+90, 0, 0, -1)
    # center = params['center']
    # tr.translate(-center.x(), -center.y(), -center.z())

    inv, result = tr.inverted()

    point = inv * (point)
    return point.x(), point.y(), point.z()

## Code taking directly from main.py main function.... Why was it defined there... IDK
def roipoly(src, poly):
    mask = np.zeros_like(src, dtype=np.uint8)
    win = util.swap_indices(poly)
    cv2.fillConvexPoly(mask, win, 255)  # Create the ROI
    res = src * mask
    # cv2.imshow("roi", res)
    # cv2.waitKey(0)
    return res


def getOrientation(line, window_size):
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


def getOrdering(pt1, pt2, pt3, pt4):
    temp1 = np.linalg.norm(np.subtract((np.add(pt1, pt3) / 2.0), (np.add(pt2, pt4) / 2.0)))
    temp2 = np.linalg.norm(np.subtract((np.add(pt1, pt4) / 2.0), (np.add(pt2, pt3) / 2.0)))
    res = np.array([pt1, pt3, pt4, pt2]) if temp1 > temp2 else np.array([pt1, pt4, pt3, pt2])
    return [[int(i) for i in pt] for pt in res]


def gradDir(img):
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

class ObjParser(object):
    """This defines a generalized parse dispatcher; all parse functions
    reside in subclasses."""

    def parseFile(self, file_path):
        self.vertexList = []
        self.faceIndecies = []
        self.materialDefinition = {}
        self.selectedMaterialName = None
        self.colors = []
        for line in open(file_path, 'r'):
            self.parseLine(line, dir=os.path.dirname(file_path))

        return {
            'vertexList': self.vertexList, 
            'faceIndecies': self.faceIndecies, 
            'colors':self.colors
        }

    def parseLine(self, line, dir):
        """Determine what type of line we are and dispatch
        appropriately."""
        if line.startswith('#'):
            return

        values = line.split()
        if len(values) < 2:
            return

        line_type = values[0]
        args = values[1:]
        i = 0
        for arg in args:
            if dir != '' and ('mtllib' in line or 'map_Kd' in line):
                args[i] = dir + '/' + arg
            else:
                args[i] = arg
            i += 1

        if hasattr(self, "parse_{}".format(line_type)):
            parse_function = getattr(self, 'parse_%s' % line_type)
            parse_function(args)
        else:
            return

    def parse_mtllib(self, args):
        materialParser = MaterialParser()
        self.materialDefinition = dict(self.materialDefinition, **materialParser.parseFile(args[0]))

    def parse_usemtl(self, args):
        materialName = args[0]
        if materialName in self.materialDefinition.keys():
            self.selectedMaterialName = materialName
        else:
            raise RuntimeError("Material {} not defined".format(materialName))

    def parse_v(self, args):
        vector = []
        for arg in args:
            vectorElement = float(arg)
            vector.append(vectorElement)
        self.vertexList.append(vector)

    def parse_f(self, args):
        face = []
        for arg in args:
            attributes = arg.split('/')
            face.append(int(attributes[0]) - 1)
        self.faceIndecies.append(face)
        self.colors.append(self.materialDefinition[self.selectedMaterialName]["Kd"])

    def getMeshData(self):
        # print("Mesh Data ")
        # print(self.vertexList)
        # print(self.faceIndecies)
        # print(self.colors)
        mesh_data = gl.MeshData(vertexes=np.array(self.vertexList)*1, faces=np.array(self.faceIndecies), faceColors=np.array(self.colors))
        return mesh_data

class MaterialParser(object):
    """This defines a generalized parse dispatcher; all parse functions
    reside in subclasses."""

    def parseFile(self, file_path):
        self.materialDefinitions = {}
        self.currentMaterialName = None
        for line in open(file_path, 'r'):
            self.parseLine(line, dir=os.path.dirname(file_path))

        return self.materialDefinitions

    def parseLine(self, line, dir):
        """Determine what type of line we are and dispatch
        appropriately."""
        if line.startswith('#'):
            return

        values = line.split()
        if len(values) < 2:
            return

        line_type = values[0]
        args = values[1:]
        i = 0

        if hasattr(self, "parse_{}".format(line_type)):
            parse_function = getattr(self, 'parse_%s' % line_type)
            parse_function(args)
        elif line_type in ("Ka", "Kd", "Ks", "Ke", "Ni", "d", "illum"):
            values = []
            for arg in args:
                val = float(arg)
                values.append(val)
            self.materialDefinitions[self.currentMaterialName][line_type] = values
        else:
            return
    def parse_newmtl(self, args):
        self.currentMaterialName = args[0]
        self.materialDefinitions[self.currentMaterialName] = {}

    def parse_Kd(self, args):
        values = []
        for arg in args:
            val = float(arg)
            values.append(val)
        values.append(1.0)
        self.materialDefinitions[self.currentMaterialName]["Kd"] = values

def getOBJParser():
    return ObjParser()

def loadPCD(fileName):
    pc = pcd.PointCloud.from_path(fileName)
    width = pc.width
    height = pc.height
    imgd = np.reshape(pc.pc_data["rgb"], (height, width, 1))
    img = imgd.copy()
    img = img.view(np.uint8)
    img = np.delete(img, 3, 2)

    x = pc.pc_data['x'].copy()
    y = pc.pc_data['y'].copy()
    z = pc.pc_data['z'].copy()

    depth = np.asarray((z/ np.nanmax(z)) * 2000, np.uint16)
    depth = np.reshape(depth, (height, width))
    # pdb.set_trace()
    # depth = util.
    # depth = cv2.
    return img, depth, (x, y, z)

def rotateImage(image, angle):
   (h, w) = image.shape[:2]
   center = (w / 2, h / 2)
   M = cv2.getRotationMatrix2D(center,angle,1.0)
   rotated_image = cv2.warpAffine(image, M, (w,h))
   return rotated_image

