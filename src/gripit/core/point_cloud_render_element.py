from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import super
from builtins import range
from builtins import str
from builtins import int
from future import standard_library
standard_library.install_aliases()
import pyqtgraph as pg
import PyQt5
from PyQt5 import QtCore
from PyQt5.QtGui import QVector3D
from pyqtgraph.opengl import GLViewWidget, GLScatterPlotItem
import pyqtgraph.opengl as gl 
from pyqtgraph import Transform3D
from gripit.core.grabbermodel import GrabberElement
import numpy as np
import logging as log
from enum import Enum
import math
import cv2
from copy import deepcopy
from OpenGL.GL import *


GRABBER_MODEL_NAME = "test_grabber_l"


class RenderGroup(Enum):    
    Fixed = 0
    Variable = 1
    Sprite = 2


class SceneElement(GLViewWidget):
    """docstring for SceneElement"""
    def __init__(self):
        # super(GLViewWidget, self).__init__(parent=None)
        GLViewWidget.__init__(self, parent=None)

        self.pointCloud = None
        self.grabberElement = None
        self.EdgePairElements = None
        self.selectedGrabber = None
        # self.pointCloudView = None
        self.renderItems = {}
        self.noRepeatKeys = [
            QtCore.Qt.Key_Right, 
            QtCore.Qt.Key_Left, 
            QtCore.Qt.Key_Up, 
            QtCore.Qt.Key_Down, 
            QtCore.Qt.Key_PageUp, 
            QtCore.Qt.Key_PageDown,
            QtCore.Qt.Key_C,
            QtCore.Qt.Key_R]


    def setData(self, **kwds):        
        args = ('pointCloud', 'imageModel', 'edgePairList', 'context')
        for k in kwds.keys():
            if k not in args:
                raise Exception('Invalid keyword argument: %s (allowed arguments are %s)' % (k, str(args)))

        for arg in args:
            if arg in kwds:
                setattr(self, arg, kwds[arg])

        self.opts = SceneElement.getSceneParameters(self.imageModel)
        self.initializeScene()        
        self.updateRenderAttributes()
        self.update()

    def update(self):
        GLViewWidget.update(self)

    @staticmethod
    def getSceneParameters(imageModel):
        coords = "xyz"
        pos = []
        center = QVector3D(0, 0, 0)
        for i in range(len(coords)):
            value = imageModel.getAttribute("camera_location_{}".format(coords[i]))
            pos.append(value)
        
        dist = math.sqrt((pos[0]-center[0])**2 + (pos[1]-center[1])**2 + (pos[2]-center[2])**2)
        elevation = math.asin((pos[2]-center[2]) / dist)
        azimuth = math.acos((pos[0]-center[0]) / (dist * math.cos(elevation)))
        if math.asin(pos[2] / (dist * math.cos(elevation))) < 0:
            azimuth = azimuth + math.pi

        focalLength = imageModel.getAttribute("camera_focal_length")
        camera_sensor_hori = imageModel.getAttribute("camera_sensor_size_x")
        camera_sensor_vert = imageModel.getAttribute("camera_sensor_size_x")
        hFOV = 2*math.atan(camera_sensor_hori/(2*focalLength))
        vFOV = 2 * math.atan(camera_sensor_vert/(2*focalLength)) # not used

        opts = {
            'focal_length': focalLength,
            'center': center,  ## will always appear at the center of the widget
            'distance': dist,         ## distance of camera from center
            'fov':  hFOV * 180/math.pi,               ## horizontal field of view in degrees
            'elevation':  elevation * 180/math.pi,         ## camera's angle of elevation in degrees
            'azimuth': azimuth * 180/math.pi,            ## camera's azimuthal angle in degrees 
            'clip_near': 0.0001*dist,
            'clip_far' : 1000 * dist, 
            'viewport': None,         ## glViewport params; None == whole widget
            'bgcolor': (0.10, 0, 0, 0)
        }
        # print(opts)
        return opts

    def getGripperSprite(self):
        self.pointCloudView.hide()
        self.pointCloudView.setVisible(False)
        # self.removeItem(self.pointCloudView)
        self.paintGL()
        self.setBackgroundColor('k')
        glClear( GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT)
        glFlush()
        data = self.renderToArray(size=(self.width(), self.height()))

        # img = self.readQImage()

        width = self.width()
        height = self.height()

        # ptr = img.bits()
        # ptr.setsize(img.byteCount())
        arr = deepcopy(np.array(data).transpose(1, 0, 2))#.reshape(height, width, 4)  #  Copies the data
        cv2.imshow("renderbuffer", arr)
        self.pointCloudView.show()

    def getRenderedImage(self):
        self.gridElement.hide()
        self.gridElement.setVisible(False)
        # self.removeItem(self.pointCloudView)
        self.paintGL()
        self.setBackgroundColor('k')
        glClear( GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT)
        glFlush()
        data = self.renderToArray(size=(self.width(), self.height()))

        # img = self.readQImage()

        width = self.width()
        height = self.height()

        # ptr = img.bits()
        # ptr.setsize(img.byteCount())
        arr = deepcopy(np.array(data).transpose(1, 0, 2))#.reshape(height, width, 4)  #  Copies the data
        self.gridElement.show()
        return arr

    def initializeScene(self):
        # setup backgroud color
        # self.setBackgroundColor(pg.mkColor(0.10))

        for renderGroup in RenderGroup:
            if renderGroup in self.renderItems.keys():
                for item in self.renderItems[renderGroup]:
                    self.removeItem(item)
                    self.renderItems[renderGroup].remove(item)
            else:
                self.renderItems[renderGroup] = []
        self.gridElement = gl.GLGridItem()
        self.addItem(self.gridElement, RenderGroup.Fixed)

        self.pointCloudView = gl.GLScatterPlotItem(pxMode=True)
        # self.pointCloudView.scale(-1,1,1, local=False)
        # self.pointCloudView.translate(
        #     self.getContext().camera_translate_x,
        #     self.getContext().camera_translate_y,
        #     self.getContext().camera_translate_z, local=False)
        # self.pointCloudView.rotate(angle=self.getContext().camera_pitch, x=-1, y=0, z=0, local=False)
        self.addItem(self.pointCloudView, RenderGroup.Fixed)




    def processFace(self, edgePair):
        """
        Using ransac, calculates the normals for the currently selected edge pairs and displays a vectors
        corresponding the orientatoin of the face.
        """
        self.currentEdgePair = edgePair
        # if self.grabberElement is not None:
        #     self.removeItem(self.grabberElement)
        self.grabberElement = GrabberElement(self.context)
        self.addItem(self.grabberElement, group=RenderGroup.Variable)

        self.grabberElement.show()
        length = 10

        position = edgePair.getCenterPoint3D()
        eigenVectors = edgePair.getOrientation()
        # make eigen vector into 4x4 matrix for transform3d
        eigenVectors = np.insert(eigenVectors, 3, 0, axis=0)
        eigenVectors = np.insert(eigenVectors, 3, 0, axis=1)
        eigenVectors = eigenVectors.T
        eigenVectors = eigenVectors.flatten()
        # set eigen[3][3] to 1
        eigenVectors[15] = 1
        transformation  = Transform3D(eigenVectors)
        self.grabberElement.applyTransform(transformation, local=False)
        self.grabberElement.translate(position[0], position[1], position[2], local=False)
        self.grabberElement.applyTransform(self.pointCloudView.transform(), local=False)
        self.grabberElement.update()

        axisItem = gl.GLAxisItem(size=QVector3D(length, length, length))
        axisItem.applyTransform(self.grabberElement.transform(), local=False)
        self.addItem(axisItem, RenderGroup.Variable)
        self.grabberElement.setOpenGrabber(edgePair.getDistanceBetweenEdgePair())
        return


    def moveGrabber(self, dx, dy):
        if self.currentEdgePair is None or self.selectedGrabber is None:
            return

        cPos = self.cameraPosition()        
        cVec = self.opts['center'] - cPos
        dist = cVec.length()
        xDist = dist * 2. * np.tan(0.5 * self.opts['fov'] * np.pi / 180.)  ## approx. width of view at distance of center point
        xScale = xDist / self.width()

        translationVector = QVector3D(dx, dy, 0) * xScale * 100
        if translationVector.length() is not 0:
            self.selectedGrabber.showDirectionArrow(3)
        else:
            self.selectedGrabber.showDirectionArrow(0)

        transformation, res = self.projectionMatrix().inverted()
        translationVector = transformation * translationVector

        orientation = self.currentEdgePair.getOrientation()
        xAxis = QVector3D(orientation[0][0], orientation[0][1], orientation[0][2])
        translationMag = QVector3D.dotProduct(translationVector, xAxis)/ xAxis.length()
        translationMag = translationMag * xScale * 200
        grabberTransform = self.grabberElement.transform()
        grabberTransform.translate(translationMag, 0, 0)
        grabberPos = grabberTransform.column(3).toVector3D()

        max = 0
        for i in range(2):
            edge = self.currentEdgePair[i]
            startPos = edge.getStartPos3D()
            startPos = QVector3D(startPos[0], startPos[1], startPos[2])
            endPos = edge.getEndPos3D()
            endPos = QVector3D(endPos[0], endPos[1], endPos[2])
            distV = endPos - startPos
            if distV.length() > max:
                max = distV.length()
        max = max/2
        edgePairCenter = self.currentEdgePair.getCenterPoint3D()
        edgePairCenter = QVector3D(edgePairCenter[0], edgePairCenter[1], edgePairCenter[2])

        difference = grabberPos - edgePairCenter
        if max < difference.length():
            col = edgePairCenter.toVector4D()
            col.setW(1)
            grabberTransform.setColumn(3, col)
            if translationMag > 0:
                self.selectedGrabber.showDirectionArrow(1)
            else:
                max = -max
                self.selectedGrabber.showDirectionArrow(2)                

            grabberTransform.translate(max, 0, 0)
            self.grabberElement.setTransform(grabberTransform)
        else:
            self.grabberElement.translate(translationMag, 0, 0, local=True)
        self.grabberElement.update()
        self.paintGL()
        
    def positionGrabber(self, transformation): 
    # Grabber is positioned in process face for now
        pass


    def updateRenderAttributes(self):
        pointCloudData = None
        glLineList = [] # This may need to be reimplemented
        try:
            pointCloudData = self.pointCloud.getPointCloudData();
        except Exception:
            log.info("No pointcloud data available")
            return

        rgbData = self.imageModel.getCroppedRGBImage()

        rgbData = rgbData.reshape(rgbData.shape[0]*rgbData.shape[1], 3)
        color = np.insert(rgbData, 3, 255, axis=1)/255

        # color = np.full((len(pointCloudData), 4), (0.1, 0.1, 0.8, 0.8))
        size = np.full((len(pointCloudData), 1), 1.8).flatten()

        for edgePair in self.edgePairList:
            glLineList.append(self.getGLlineFromEdgePair(edgePair))
            params = self.imageModel.context.processFace(self.imageModel, edgePair)
            norm = params[2][:,2]
            if norm[2] < -math.cos(math.pi*0.7) : 
                norm = norm * -1
            glLine = self.toGlLine(edgePair.getCenterPoint3D(), params[0] + norm*3, col=(0,1,0,1))
            # glLine.applyTransform(self.grabberElement.transform(), local=True)
            glLineList.append(glLine)
            # glLineList.append(self.toGlLine(params[0], params[0] + params[2][1]*30))
            # glLineList.append(self.toGlLine(params[0], params[0] + params[2][0]*30))
            for m in range(2):
                edge = edgePair[m]
                glLineList.append(self.getGLlineFromEdge(edge))

                edgePointList = edge.getEdgePointCloudIndexes()

                if edge.getAttribute("points_shifted"):
                    oldPointList = edge.getAttribute("oldPointList")
                    perpendicularVector = edge.getAttribute("perpendicularVector")
                    offset = edge.getAttribute("perpendicularVectorOffset")
                    lineGradientVectorList = edge.getAttribute("line_shift_gradient")
                    transitionColor = (0.2, 0.2, 0.7, 0.7)
                    for i in range(len(lineGradientVectorList)):
                        point = oldPointList[i]
                        lineGradient = lineGradientVectorList[i]
                        for j in range(len(lineGradient)):
                            gradientValue = lineGradient[j]
                            gradPoint = (int(point[0] + perpendicularVector.dx()*(j - offset + 1)), int(point[1] + perpendicularVector.dy()*(j - offset + 1)))
                            index = self.pointCloud.getIndexfromXYCoordinate(gradPoint[0], gradPoint[1])
                            currentColor = color[index]
                            newColor = []

                            for k in range(len(currentColor)):
                                newColor.append(currentColor[k] * (1.0 - float(gradientValue)) + transitionColor[k] * float(gradientValue))                  
                            color[index] = np.array(newColor)
                            size[index] = size[index] + 1.1 * float(gradientValue)

                for point in edge.getEdgePointCloudIndexes():
                    indexx = self.pointCloud.getIndexfromXYCoordinate(point[0], point[1])
                    tcol = edge.getRenderColor()
                    # color[indexx] = (tcol.redF(), tcol.greenF(), tcol.blueF(), 0.8)
                    color[indexx] = np.array((0.7, 0.7, 0.7, 0.8))
                    size[indexx] = (2.2)

        self.pointCloudView.setData(pos=pointCloudData, size=size, color=color, pxMode=True)

        # Add edgePair Lines to render object
        for glLine in glLineList:
            self.addItem(glLine, group=RenderGroup.Variable)


    def addItem(self, item, group=RenderGroup.Variable):
        self.renderItems[group].append(item)
        GLViewWidget.addItem(self, item)

    def removeItem(self, item):
        GLViewWidget.removeItem(self, item)
        return
        for renderGroup in self.renderItems:
            for renderItem in renderGroup:
                if renderItem == item:
                    renderGroup.remove(renderItem)
    
    def getGLlineFromEdgePair(self, edgePair):
        lineWidth = 2
        startPos = edgePair.getCenterPoint3D()
        endPos = startPos + edgePair._getBisectingVector() * 3
        tcol = (1,0,0,1) #edgePair.getRenderColor()

        glLine = gl.GLLinePlotItem(pos=np.vstack([
            np.asarray([startPos]),
            np.asarray([endPos])
            ]), 
            color=tcol, mode="lines", width=lineWidth)
        glLine.setTransform(self.pointCloudView.transform())

        return glLine

    def getGLlineFromEdge(self, edge):
        lineWidth = 2
        startPos = edge.getStartPos3D()
        endPos = edge.getEndPos3D()
        tcol = edge.getRenderColor()

        glLine = gl.GLLinePlotItem(pos=np.vstack([
            np.asarray([startPos]),
            np.asarray([endPos])
            ]), 
            color=(tcol.redF(), tcol.greenF(), tcol.blueF(), 0.8), mode="lines", width=lineWidth)
        glLine.setTransform(self.pointCloudView.transform())

        return glLine

    def toGlLine(self, startPos, endPos, col=(1,1,1,1)):
        lineWidth = 2

        glLine = gl.GLLinePlotItem(pos=np.vstack([
            np.asarray([startPos]),
            np.asarray([endPos])
            ]), 
            color=col, mode="lines", width=lineWidth)
        glLine.setTransform(self.pointCloudView.transform())

        return glLine


    def removeAllItems(self):
        for group in adself.renderItems.keys():
            for item in self.renderItems[group]:
                self.removeItem(item)
                renderItem[group].remove(item)
        self.update()



    def projectionMatrix(self, region=None):
        # Xw = (Xnd + 1) * width/2 + X
        if region is None:
            region = (0, 0, self.width(), self.height())
        
        x0, y0, w, h = self.getViewport()
        dist = self.opts['distance']
        fov = self.opts['fov']
        nearClip = dist * 0.001 if self.opts['clip_near'] is None else self.opts['clip_near']
        farClip = dist * 1000 if  self.opts['clip_far'] is None else self.opts['clip_far']

        r = nearClip * np.tan(fov * 0.5 * np.pi / 180.)
        t = r * h / w

        # convert screen coordinates (region) to normalized device coordinates
        # Xnd = (Xw - X0) * 2/width - 1
        ## Note that X0 and width in these equations must be the values used in viewport
        left  = r * ((region[0]-x0) * (2.0/w) - 1)
        right = r * ((region[0]+region[2]-x0) * (2.0/w) - 1)
        bottom = t * ((region[1]-y0) * (2.0/h) - 1)
        top    = t * ((region[1]+region[3]-y0) * (2.0/h) - 1)

        tr = PyQt5.QtGui.QMatrix4x4()
        tr.frustum(left, right, bottom, top, nearClip, farClip)
        return tr
    

    def evalKeyState(self):
        speed = 2.0
        if len(self.keysPressed) > 0:
            for key in self.keysPressed:
                if key == QtCore.Qt.Key_Right:
                    self.orbit(azim=-speed, elev=0)
                elif key == QtCore.Qt.Key_Left:
                    self.orbit(azim=speed, elev=0)
                elif key == QtCore.Qt.Key_Up:
                    self.orbit(azim=0, elev=-speed)
                elif key == QtCore.Qt.Key_Down:
                    self.orbit(azim=0, elev=speed)
                elif key == QtCore.Qt.Key_PageUp:
                    pass
                elif key == QtCore.Qt.Key_PageDown:
                    pass
                elif key == QtCore.Qt.Key_C:
                    params = SceneElement.getSceneParameters(self.imageModel)
                    self.setCameraPosition(distance=params['distance'], elevation=params['elevation'], azimuth=params['azimuth'])
                elif key == QtCore.Qt.Key_R:
                    self.getGripperSprite()
                self.keyTimer.start(16)
        else:
            self.keyTimer.stop()

    def mousePressEvent(self, ev):
        regionSize = 5
        mousePos = ev.pos()
        self.mousePos1 = mousePos
        region = (mousePos.x(), mousePos.y(), regionSize, regionSize)

        items = self.itemsAt(region)
        for item in items:
            print(item)
            if isinstance(item, gl.GLMeshItem) and item.parentItem() == self.grabberElement:
                self.selectedGrabber = self.grabberElement
                return
        print("GLWidet mousePRess")
        super(GLViewWidget, self).mousePressEvent(ev)
        GLViewWidget.mousePressEvent(self, ev)


    def mouseReleaseEvent(self, ev):
        print("release")
        if self.selectedGrabber is not None:
            self.selectedGrabber.showDirectionArrow(0)
            self.selectedGrabber.update()
            self.paintGL()
        self.selectedGrabber = None
   
    def mouseMoveEvent(self, ev):
        diff = ev.pos() - self.mousePos1
        if self.selectedGrabber is not None and ev.buttons() == QtCore.Qt.LeftButton:
            self.moveGrabber(-diff.x(), diff.y())
            self.mousePos1 = ev.pos() # Ensure mousePos is set here
        else:
            GLViewWidget.mouseMoveEvent(self, ev)
            super(GLViewWidget, self).mouseMoveEvent(ev)

    def getContext(self):
        return self.context

