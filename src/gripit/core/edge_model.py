from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import super
from builtins import int
from builtins import str
from future import standard_library
standard_library.install_aliases()
from builtins import object
from PyQt5.QtCore import QPointF, QLineF
import PyQt5.QtGui as QtGui
from enum import Enum
import numpy as np
from OpenGL.GL import *
from OpenGL.arrays import vbo
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.opengl import shaders
import math

class EdgeAttributes(object):
    BJECT_LEFT = 1
    OBJECT_RIGHT = 2
    OBJECT_CENTERED = -1
    EDGE_TYPE_DD1 = 13
    EDGE_TYPE_DD2 = 14
    EDGE_TYPE_CD = 12

class LineFeatureParameters(Enum):
    Y1 = 0
    X1 = 1
    Y2 = 2
    X2 = 3
    LENGTH = 4
    SLOPE = 5
    ALPHA = 6
    INDEX = 7
    START_POINT = 8
    END_POINT = 9
    CLASSIFICATION = 10
    OBJ_LOCATION = 12

class EdgePair(object):
    def __init__(self, imageModel, Edge1, Edge2, id):
        self._edges = (Edge1, Edge2)
        self.ID = id
        self.imageModel = imageModel
        self.center3D = None
        self.orientation = None
        self.isValid = True

    def setRenderColor(self, color):
        for edge in self._edges:
            edge.setRenderColor(color)

    def getRenderColor(self):
        return self._edges[0].getRenderColor()

    def setRenderWidth(self, width):
        for edge in self._edges:
            edge.setRenderWidth(width)

    def getID(self):
        return self.ID

    def isValid(self):
        return self._isValid

    def __getitem__(self, index):
        return self._edges[index]

    def getCenterPoint3D(self):
        if self.center3D is None:
            self.getOrientation() # this method sets center 3d variable
        return self.center3D

    def getOrientation(self):
        # Usees ransac to get normal and center vector
        self.orientation = np.empty((3,3))
        params = None
        try:
            params = self.imageModel.context.processFace(self.imageModel, self)
        finally:
            if params is None:
                self._isValid = False
                raise ValueError("EdgePair {} is Invalid.".format(self.getID()))

        self.center3D = params[0]

        direction = np.array(self._getBisectingVector())

        normalVector = np.array(params[2])[:,2] # Param is in column major and should be transposed
        normalVector = self._normalizeVector(normalVector)

        sPos1 = self._edges[0].getStartPos3D()
        sPos2 = self._edges[1].getStartPos3D()
        vect = sPos2 - sPos1

        if np.arccos(np.clip(np.dot(vect, direction), -1.0, 1.0)) > math.pi/2:
            direction = -direction

        # angle = np.arccos(np.clip(np.dot(normalVector, direction), -1.0, 1.0))
        # if np.sum(np.sign(direction) == np.sign(normalVector)) < 2 and normalVector[2] < 0.15:
        #     normalVector = -normalVector

        latitude = np.cross(normalVector, direction) #params[2][1]
        self.orientation[0] = np.array(direction)
        self.orientation[1] = np.array(latitude)
        self.orientation[2] = np.array(normalVector)
        
        return self.orientation

    def _normalizeVector(self, vect):
        vect = vect/math.sqrt(math.pow(vect[0],2) + math.pow(vect[1],2) + math.pow(vect[2], 2))
        return vect


    def _getBisectingVector(self):
        edge1 = np.array(self._normalizeVector(self._edges[0].getEndPos3D() - self._edges[0].getStartPos3D()))
        edge2 = np.array(self._normalizeVector(self._edges[1].getEndPos3D() - self._edges[1].getStartPos3D()))
        edge2_dir = np.dot(edge2, edge1)
        edge1 = self._normalizeVector(edge1)
        edge2 = self._normalizeVector(edge2 * edge2_dir)

        vect = (edge2 + edge1)/2        
        return self._normalizeVector(vect)

    def getNormalVector(self):
        try:
            self.normalVector
        except AttributeError:
            orientation = self.getOrientation()
            self.normalVector = orientation[2][2]

    def getDistanceBetweenEdgePair(self):
        # define a plane from one line
        edge2 = np.array((self._edges[0].getStartPos3D(), self._edges[0].getEndPos3D()))
        edge1 = np.array((self._edges[1].getStartPos3D(), self._edges[1].getEndPos3D()))

        orientation = self.getOrientation()
        norm = orientation[1]

        center1 = (edge1[0] + edge1[1])/2
        center2 = (edge2[0] + edge2[1])/2

        tvect = edge1[0] - edge2[0] #self.getCenterPoint3D()
        dist = np.dot(tvect, norm)


        return abs(dist)


class EdgeModel(QLineF):
    imageModel = None
    pointList = None
    ID = None
    edgeAttributes = {}
    # Direction of object
    _objectLocation = None
    #_pen = None

    def __init__(self, **kwds):
        super(QLineF, self).__init__()

        self.imageModel = None
        self.pointList = None
        self.ID = None

        # initialize data
        self.setData(**kwds)

        self.pen = QtGui.QPen()
        self.setRenderColor((255, 255, 255))
        self.setRenderWidth(2)

    def setData(self, **kwds):
        """
        Initialize Edge Model:
        Args
                imageModel: The Parent ImageModel used to process line data
                startPos: Point2D Starting Position of edge
                endPos: Point2D Ending Positoin of edge
                pointList: List of indexs which are attributed to edge
                ID: line identification
        """
        args = ("ID", "lineFeatureArray", "imageModel", "pointList", "edgeAttributes")
        for k in kwds.keys():
            if k not in args:
                raise Exception("Invalid keyword argument: {} (allowed arguments are {})".format(k, str(args)))

        for arg in args:
            if arg in kwds:
                setattr(self, arg, kwds[arg])

        self.setP1(QPointF(self.lineFeatureArray[LineFeatureParameters.X1.value], 
            self.lineFeatureArray[LineFeatureParameters.Y1.value]))
        self.setP2(QPointF(self.lineFeatureArray[LineFeatureParameters.X2.value], 
            self.lineFeatureArray[LineFeatureParameters.Y2.value]))


    def setRenderColor(self, color):
        """
        Sets display color for rendering edge
        """
        self.pen.setColor(QtGui.QColor(color[0], color[1], color[2]))

    def getRenderColor(self):
        return self.pen.color()

    def setRenderWidth(self, width):
        """
        Sets Display  witdh
        """
        self.pen.setWidth(width)

    def getRenderData(self):
        return self.pen

    def setEdgePoint(self, pointList):
        self.pointList = pointList

    def getImageModel(self):
        return self.imageModel

    def getEdgePointCloudIndexes(self):
        return self.pointList

    def getPointCloudPoints(self):
        try:
            self.pointCloudPoints
        except AttributeError:
            self.pointCloudPoints = []
            for point in self.pointList:
                point3D = self.imageModel.getPointCloud().getPointfromXYCoordinate(point[0], point[1])
                self.pointCloudPoints.append(point3D)
        finally:
            return self.pointCloudPoints

    def getStartPos3D(self):
        ## Temporary 
        if self.hasAttribute("startPos3D"):
            return self.getAttribute("startPos3D")
        return self.imageModel.getPointCloud().getPointfromXYCoordinate(self.x1(), self.y1())
        

    def getEndPos3D(self):
        if self.hasAttribute("endPos3D"):
            return self.getAttribute("endPos3D")
        return self.imageModel.getPointCloud().getPointfromXYCoordinate(self.x2(), self.y2())


    def getStartIndex(self):
        return self.imageModel.getPointCloud().getIndexfromXYCoordinate(self.x1(), self.y1())

    def getEndIndex(self):
        return self.imageModel.getPointCloud().getIndexfromXYCoordinate(self.x2(), self.y2())

    def getStartPos(self):
        return QPointF(self.x1(), self.y1())

    def getEndPos(self):
        return QPointF(self.x2(), self.y2())

    def getID(self):
        return self.ID

    def getAttribute(self, key):
        return self.edgeAttributes[key]

    def setAttribute(self, key, value):
        self.edgeAttributes[key] = value

    def hasAttribute(self, key):
        return key in self.edgeAttributes

    # TODO
    def paint(self):
        if self.pos is None:
            return
        self.setupGLState()
        
        glEnableClientState(GL_VERTEX_ARRAY)

        try:
            glVertexPointerf(self.pos)
            
            if isinstance(self.color, np.ndarray):
                glEnableClientState(GL_COLOR_ARRAY)
                glColorPointerf(self.color)
            else:
                if isinstance(self.color, QtGui.QColor):
                    glColor4f(*fn.glColor(self.color))
                else:
                    glColor4f(*self.color)
            glLineWidth(self.width)
            #glPointSize(self.width)
            
            if self.antialias:
                glEnable(GL_LINE_SMOOTH)
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
                
            if self.mode == 'line_strip':
                glDrawArrays(GL_LINE_STRIP, 0, int(self.pos.size / self.pos.shape[-1]))
            elif self.mode == 'lines':
                glDrawArrays(GL_LINES, 0, int(self.pos.size / self.pos.shape[-1]))
            else:
                raise Exception("Unknown line mode '%s'. (must be 'lines' or 'line_strip')" % self.mode)
                
        finally:
            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)

