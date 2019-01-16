from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
import pyqtgraph.opengl as gl 
from pyqtgraph import Transform3D
from PyQt5.QtGui import QVector3D
import numpy as np
import logging as log

class GrabberElement(GLGraphicsItem):
    """docstring for GrabberElement"""
    MAX_WIDTH = 5
    def __init__(self, context, parentItem=None):
        # super(gl.GLGraphicsItem, self).__init__(parentItem)
        GLGraphicsItem.__init__(self)

        self.context = context
        self.leftGrabber = None
        self.rightGrabber = None
        self.upArrow = None
        self.forwardArrow = None
        self.currentOpenningWidth = 0

        self.initGrabberModel("grabber{}", "grabber_arrow{}", "grabber_base")
        self.setOpenGrabber(2)
        self.showDirectionArrow(0)

    def initGrabberModel(self, grabberMeshName, arrowName, baseMeshName):
        # initialize grabber Element
        grabberRMeshData = self.context.loadModel(grabberMeshName.format("_r"))
        grabberLMeshData = self.context.loadModel(grabberMeshName.format("_l"))
        forwardArrowMeshData = self.context.loadModel(arrowName.format("_up"))
        upArrowMeshData = self.context.loadModel(arrowName.format("_forward"))
        grabberBaseMeshData = self.context.loadModel(baseMeshName)

        grabberRMeshArguments = {
            "meshdata": grabberRMeshData,
            "color": (0.75, 0.75, 0.75, 0.85),
            "edgeColor": (0.4, 0.4, 0.4, 1.0),
            "drawEdge": True,
            "drawFace": True,
            "shader": 'shaded',
            "glOptions": 'opaque'
        }

        grabberLMeshArguments = {
            "meshdata": grabberLMeshData,
            "color": (0.75, 0.75, 0.75, 0.85),
            "edgeColor": (0.4, 0.4, 0.4, 1.0),
            "drawEdge": True,
            "drawFace": True,
            "shader": 'shaded',
            "glOptions": 'opaque'
        }

        baseArguments = {
            "meshdata": grabberBaseMeshData,
            "color": (0.75, 0.75, 0.75, 0.85),
            "edgeColor": (0.4, 0.4, 0.4, 1.0),
            "drawEdge": True,
            "drawFace": True,
            "shader": 'shaded',
            "glOptions": 'opaque'
        }

        forwardArrowArguments = {
            "meshdata": forwardArrowMeshData,
            "color": (0.75, 0.75, 0.75, 0.3),
            "edgeColor": (0.4, 0.4, 0.4, 1.0),
            "drawEdge": True,
            "drawFace": True,
            "shader": 'balloon',
            "glOptions": 'additive'
        }

        backwardArrowArguments = {
            "meshdata": forwardArrowMeshData,
            "color": (0.75, 0.75, 0.75, 0.3),
            "edgeColor": (0.4, 0.4, 0.4, 1.0),
            "drawEdge": True,
            "drawFace": True,
            "shader": 'balloon',
            "glOptions": 'additive'
        }

        upArrowArguments = {
            "meshdata": upArrowMeshData,
            "color": (0.75, 0.75, 0.75, 0.3),
            "edgeColor": (0.4, 0.4, 0.4, 1.0),
            "drawEdge": True,
            "drawFace": True,
            "shader": 'balloon',
            "glOptions": 'additive'
        }

        mirrorTrans = np.identity(4)
        mirrorTrans[1][1] = -1

        self.rightGrabber = gl.GLMeshItem(**grabberRMeshArguments)
        self.leftGrabber = gl.GLMeshItem(**grabberLMeshArguments)
        self.leftGrabber.applyTransform(Transform3D(mirrorTrans.flatten()), local=True)
        self.rightGrabber.applyTransform(Transform3D(mirrorTrans.flatten()), local=True)
        self.forwardArrow = gl.GLMeshItem(**forwardArrowArguments)
        self.backwardArrow = gl.GLMeshItem(**backwardArrowArguments)
        mirrorTrans[1][1] = 1
        mirrorTrans[0][0] = -1
        self.backwardArrow.applyTransform(Transform3D(mirrorTrans.flatten()), local=True)
        self.upArrow = gl.GLMeshItem(**upArrowArguments)
        self.grabberBase = gl.GLMeshItem(**baseArguments)

        self.rightGrabber.setParentItem(self)
        self.leftGrabber.setParentItem(self)
        self.forwardArrow.setParentItem(self)
        self.backwardArrow.setParentItem(self)
        self.upArrow.setParentItem(self)
        self.grabberBase.setParentItem(self)

    def setOpenGrabber(self, width):
        if width > self.MAX_WIDTH:
            log.warning("Grabber opened beyond maximum width.")
        twidth = width
        width = width - self.currentOpenningWidth
        dr_translate  = width / 2
        dl_translate = width / 2
        self.currentOpenningWidth = twidth
        self.rightGrabber.translate(0, -dr_translate, 0, local=True)
        self.leftGrabber.translate(0, dl_translate, 0, local=True)

    def showDirectionArrow(self, state=0):
        if state == 0:
            self.forwardArrow.hide()
            self.backwardArrow.hide()
        elif state == 2:
            self.forwardArrow.show()
            self.backwardArrow.hide()
        elif state == 1:
            self.forwardArrow.hide()
            self.backwardArrow.show()
        elif state == 3:
            self.forwardArrow.show()
            self.backwardArrow.show()
