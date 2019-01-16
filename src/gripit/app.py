# -*- coding: utf-8 -*-
"""
GripIt - UCF, Edge Base Gripper Implementation
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import super
from builtins import range
from builtins import int
from builtins import str
from future import standard_library
standard_library.install_aliases()

__author__ = 'Yannick Roberts'
__license__ = 'MIT'
__version__ = '1.1.20'


# Application Imports 
import sys, math, collections, math, copy
import logging as log
import PyQt5.QtWidgets as QtWdgt
import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import cv2


from gripit.core.context import EdgeProcessingDetectContext, ExecutionMode
from gripit.core.point_cloud_render_element import SceneElement
from gripit.core.ros_image import ROSImage, OnTopicReived
from gripit.gui.ImageViewerQt import ImageViewerQt
from gripit.gui.FrameLayout import FrameLayout as CollapsableBox
from gripit.gui.QLabelSlider import QLabelSlider


# Set this number of the image index that needs to be procesed
# (fixme) add file loading if necessary

IMAGE_NUM = 0
EXECUTION_MODE = ExecutionMode.DEVELOPMENT_ROS
DATA_STORE = "real"

class App(QtGui.QWidget):
    edgeProcessorContext = None
    currentImageModel = None    

    # Application Initializatoin
    def __init__(self, app):
        super().__init__()
        global IMAGE_NUM
        self.title = 'GripIt'
        self.left = 10
        self.top = 100
        self.width = 1020
        self.height = 640
        self.isProcesed = False
        self.processRuns = 0
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.initArg(app)
        self.initEdgeProcessorContext(app)
        self.initUI()
        if (self.imageSelectBox.count() == 0) and (self.getContext().getMode() != ExecutionMode.DEVELOPMENT_ROS):
            log.warning("Image datastore is empty.")
            exit()
        # load selected image else load first image in list
        IMAGE_NUM = 0 if IMAGE_NUM == 0 else self.imageSelectBox.findData(str(IMAGE_NUM))
        if IMAGE_NUM is -1: # If entered data is not present
            log.warning("Invalid index entered.")
            IMAGE_NUM = 0
        self.imageSelectBox.currentIndexChanged.emit(IMAGE_NUM)

        # self.initSidebarAttrbutes(self.currentImageModel, self.sideBarLayout)
        # self.displayImage()

    def initArg(self, app):
        global IMAGE_NUM
        global EXECUTION_MODE
        global DATA_STORE
        args = app.arguments()
        #EXECUTION_MODE = ExecutionMode.DEVELOPMENT_ROS
        DATA_STORE = "ros"
        #return
        for i in range(len(args)):
            if args[i] == "-n":                
                IMAGE_NUM = args[i + 1]
            elif args[i] == "-s":
                DATA_STORE = args[i + 1]
            elif args[i] == "-m":
                if args[i+1] == "user":
                    EXECUTION_MODE = ExecutionMode.USER
                elif args[i+1] == "developer":
                    EXECUTION_MODE = ExecutionMode.DEVELOPMENT
            elif args[i] == "--ros":
                EXECUTION_MODE = ExecutionMode.DEVELOPMENT_ROS
                DATA_STORE = "ros"

        

    def initUI(self):
        """Initialize Program UI
        """
        log.info("Initializing program ui")
        self.tabWidget = QtGui.QTabWidget()        
        self.tabWidget.setTabsClosable(False)
        self.tabWidget.setMovable(True)
        self.tabWidget.setMinimumWidth(self.width-280)
        # self.tabWidget.tabCloseRequested.connect(self.close_window_tab)
        appLayout = QtGui.QGridLayout(self)
        appLayout.setSpacing(4)

        self.sideBarTabWidget = QtGui.QTabWidget()
        sideBarWdgt = QtGui.QWidget()

        scroll = QtGui.QScrollArea()
        # scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.sideBarLayout = QtWdgt.QVBoxLayout()
        sideBarWdgt.setLayout(self.sideBarLayout)
        self.sideBarTabWidget.setMinimumWidth(320)
        self.sideBarTabWidget.setMinimumHeight(600)
        scroll.setWidget(sideBarWdgt)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(450)
        wrapper, self.imageSelectBox = self.initImageListComboBox()
        # self.sideBarLayout.addWidget(wrapper)

        tLayout = QtWdgt.QVBoxLayout()
        tWdgt = QtGui.QWidget()
        tWdgt.setLayout(tLayout)
        tLayout.addWidget(wrapper)
        tLayout.addWidget(scroll)
        self.sideBarTabWidget.addTab(tWdgt, "Parameters")

        
        self.imageWidget = QtGui.QWidget()
        self.tabWidget.addTab(self.imageWidget, "Base Image")
        baseImageLayout = QtWdgt.QHBoxLayout(self.imageWidget)
        self.baseImage = ImageViewerQt()
        appLayout.addWidget(self.sideBarTabWidget, 0,0, 1, 1)
        appLayout.addWidget(self.tabWidget, 0, 1, 1, 1)
        baseImageLayout.addWidget(self.baseImage)
        self.loadTabUI(self)


        self.show()

    # Returns a UI combobox which has a list of avvialable image in datastore
    def initImageListComboBox(self):
        imageDescriptors = self.edgeProcessorContext.listAvailableImages()
        comboBox = QtWdgt.QComboBox()
        for descriptor in imageDescriptors:
            # do not add depth map to color listings
            if ("depth" in descriptor[0]) and (self.getContext().getMode() == ExecutionMode.DEVELOPMENT_ROS):
                continue
            comboBox.addItem(descriptor[0], descriptor[1])     
        def _imageSelectedEvent(val): # Image has been selected
            index = self.imageSelectBox.itemData(val)   
            self.loadImageFile(index)
        comboBox.currentIndexChanged.connect(_imageSelectedEvent)

        qtComponentWrapper = QtWdgt.QGroupBox("Images")
        qtComponentLayout = QtWdgt.QGridLayout()
        qtComponentWrapper.setLayout(qtComponentLayout)
        qtComponentLayout.addWidget(comboBox, 0, 0, 1, 2)

        # # If ROS, add depth option        
        if (self.getContext().getMode() == ExecutionMode.DEVELOPMENT_ROS):
            dcomboBox = QtWdgt.QComboBox()
            self.dImageSelectBox = dcomboBox
            for descriptor in imageDescriptors:
                if "depth" in descriptor[0]:
                    dcomboBox.addItem(descriptor[0], descriptor[1])

            #def _imageSelectedEvent(val): # Image has been selected
            #    index = self.imageSelectBox.itemData(val)
            #    self.loadImageFile(index)
            #dcomboBox.currentIndexChanged.connect(_imageSelectedEvent)
            qtComponentLayout.addWidget(dcomboBox, 1, 0, 1, 2)

        saveButton = QtGui.QPushButton("Save")
        qtComponentLayout.addWidget(saveButton, 2, 0, 1, 1)
        def _saveImageParamsEvent(val):
            self.storeImageParameters(self.currentImageModel)
        saveButton.clicked.connect(_saveImageParamsEvent)

        resetButton = QtGui.QPushButton("Reset")
        def _resetImageParamsEvent(val):
            self.resetImageParameters(self.currentImageModel)
        resetButton.clicked.connect(_resetImageParamsEvent)
        qtComponentLayout.addWidget(resetButton, 2, 1, 1, 1)

        # Add process button
        self.processImgBtn = QtGui.QPushButton("Process")
        self.processImgBtn.clicked.connect(self.processImage)
        qtComponentLayout.addWidget(self.processImgBtn, 3, 0, 1, 2)

        return qtComponentWrapper, comboBox


    def initEdgeProcessorContext(self, app):
        log.info("Initializing program Context")
        self.edgeProcessorContext = EdgeProcessingDetectContext.initializeContext(
            dataStore=DATA_STORE,
            _mode=EXECUTION_MODE)

    # event which loads ImageModel
    def loadImageFile(self, imageNumber):        
        # remove all sidebar parameters except imageselect
        self.clearLayout(self.sideBarLayout)

        #get image numbers
        imageId = self.imageSelectBox.currentText()
        if self.getContext().getMode() == ExecutionMode.DEVELOPMENT_ROS:
            dimgId = self.dImageSelectBox.currentText()
            imageId = (imageId, dimgId)

        self.currentImageModel = self.edgeProcessorContext.loadImage(imageId)
        self.initSidebarAttrbutes(self.currentImageModel, self.sideBarLayout)
        
        if self.edgeProcessorContext.getMode() == ExecutionMode.DEVELOPMENT_ROS:
            self.displayROSImage(self.currentImageModel, 0)   # select rgb as default image 
        else:
            self.displayImage()


    def storeImageParameters(self, imageModel):
        params = imageModel._imageAttributes
        self.getContext().saveImageModelParameters(params, imageModel.getName())

        # save auxiliary images
        if imageModel.isProcessed():
            # try to incorporate pointcloud to image storage

            pointCloudImage = self.glWidget.getRenderedImage()
            imageModel.addAuxiliaryImage("point_cloud", pointCloudImage)


            auxImages = imageModel.auxiliary_images
            for imageName in auxImages:
                imageModel.saveAuxiliaryImage(imageName)

    def resetImageParameters(self, imageModel):
        self.getContext().resetImageModelParameters(imageModel)
        self.imageSelectBox.currentIndexChanged.emit(self.imageSelectBox.currentIndex())

    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()
            elif child.layout() is not None:
                self.clearLayout(child.layout())

    def initSidebarAttrbutes(self, currentImageModel, parentLayout):
        
        attributes = copy.deepcopy(currentImageModel._imageAttributes)
        for key in attributes:
            item = attributes[key]
            if item['hidden']: continue
            component = self.initImageAttributeUIComponents(key, item, currentImageModel)
            parentLayout.addWidget(component)
        # add stretch to window
        parentLayout.addStretch(1)
        # parentLayout.setSizeConstraint(parentLayout.SetMinAndMaxSize)

    def initImageAttributeUIComponents(self, key, item, imageModel):
        # qtComponentWrapper = CollapsableBox(item['label'])
        # qtComponentLayout = qtComponentWrapper.getContentLayout()
        qtComponentWrapper = QtWdgt.QGroupBox(item['label'])
        qtComponentLayout = QtWdgt.QVBoxLayout()
        qtComponentWrapper.setLayout(qtComponentLayout)
        qtComponent = None
        attributeType = item['type']
        # component callback
        def _componentCallback(val):
            qtComponentWrapper.setTitle("{}:{}".format(item['label'], str(val)))
            imageModel.setAttribute(key, val)
        # If component is an integer
        if attributeType == 'INT':
            def _componentCallback(val):
                if val == '':
                    val = 0
                qtComponentWrapper.setTitle("{}:{}".format(item['label'], str(val)))
                imageModel.setAttribute(key, int(val))
            #define numeric text validator
            onlyInt = QtGui.QIntValidator()
            qtComponent = QtWdgt.QLineEdit()
            qtComponent.setValidator(onlyInt)
            qtComponent.setText("{}".format(item['value']))
            qtComponent.textEdited.connect(_componentCallback)
        elif attributeType == 'INT_RANGE':
            def _componentCallback(val):
                imageModel.setAttribute(key, int(val))
                qtComponentWrapper.setTitle("{}:{}".format(item['label'], str(val)))
            qtComponent = QtGui.QSlider(1)
            qtComponent.setRange(item['min'], item['max'])
            # qtComponent.setMaximum(item['max'])
            # qtComponent._setTickPosition(QtWdgt.QSlider.NoTicks)            
            qtComponent.setTickInterval(int((item['max']-item['min'])/10))
            qtComponent.setValue(item['value'])
            qtComponent.valueChanged.connect(_componentCallback)
        elif attributeType == 'REAL':
            def _componentCallback(val):
                if val == '':
                    val = 0.0
                elif val[-1] == '.':
                    val = val + '0'
                try:
                    float(val)
                    qtComponentWrapper.setTitle("{}:{}".format(item['label'], str(val)))
                except ValueError:
                    log.warning("Non-numeric value added")
                    return
                imageModel.setAttribute(key, float(val))
            #define numeric text validator
            onlyInt = QtGui.QDoubleValidator()
            qtComponent = QtWdgt.QLineEdit()
            qtComponent.setValidator(onlyInt)
            qtComponent.setText("{}".format(item['value']))
            qtComponent.textEdited.connect(_componentCallback)
            qtComponent.textEdited.emit(qtComponent.text())
        elif attributeType == 'STRING':
            qtComponent = QtWdgt.QText()
        elif attributeType == 'UI_GROUP':
            qtComponentWrapper = CollapsableBox(item['label'])
            qtComponentLayout = qtComponentWrapper.getContentLayout()
            # qtComponent = QtWdgt.QGroupBox(item['label'])
            # qtComponent.setLayout(qtComponentLayout)    
            for k in item['value']:
                i = item['value'][k]
                # if i['hidden'] == True: continue
                imageModel.setAttribute(k, i['value'])
                qtComponent = self.initImageAttributeUIComponents(k, i, imageModel)                
                # qtComponent.setEnabled(True)
                qtComponentLayout.addWidget(qtComponent)
            # qtComponentWrapper.addStretch(1)
            return qtComponentWrapper
        else:
            raise RuntimeError("GUI Componenet not defined.")

        qtComponent.setEnabled(True)
        # Add button to interface
        qtComponentLayout.addWidget(qtComponent)
        qtComponentLayout.addStretch(1)
        # parentLayout.addWidget(qtComponentWrapper) could be deleted

        return qtComponentWrapper

    def displayImage(self):
        if self.currentImageModel is None:
            raise RuntimeError("Unable to display image")
        cv2Img = self.currentImageModel.getBaseRGBImage()
        timage = QtGui.QImage(cv2Img.data, cv2Img.shape[1], cv2Img.shape[0], 3 * cv2Img.shape[1], QtGui.QImage.Format_RGB888)
        self.baseImage.setImage(timage)

        if self.currentImageModel.hasAttribute("crop_rectangle"):
            crop_rect = self.currentImageModel.getAttribute("crop_rectangle")
            self.baseImage.setCropRectangle(crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3])

    def displayROSImage(self, imageModel, index):
        if self.currentImageModel is None:
            raise RuntimeError("Unable to display image")
        
        try:
            self.rosImg.unregisterReceiveSignal()
        except AttributeError:
            self.rosImg = None


        if index == 0:
            self.rosImg = imageModel.getBaseRGBImage()
        else:
            self.rosImg = imageModel.getBaseDepthImage()
        signal = OnTopicReived()

        def _updateImage(cv2Img):
            timage = None
            if index == 0:
                timage = QtGui.QImage(cv2Img.data, cv2Img.shape[1], cv2Img.shape[0], 3 * cv2Img.shape[1], QtGui.QImage.Format_RGB888)
            else:
                print("TODO")
                pass
                #timage = QtGui.QImage(cv2Img.data, cv2Img.shape[1], cv2Img.shape[0], 3*)
            self.baseImage.setImage(timage)

            if imageModel.hasAttribute("crop_rectangle") and self.isProcesed == False:
                crop_rect = imageModel.getAttribute("crop_rectangle")
                self.baseImage.setCropRectangle(crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3])

        signal.receivedSignal.connect(_updateImage)
        self.rosImg.registerReceiveSignal(signal)


    def clearImageCache(self, imageModel):
        imageModel.deleteCache()

    def getTabView(self, name):
        pass

    def loadTabUI(self, hidden=True):
        self.tabItems = collections.OrderedDict()

        # init contour tab
        debugTab = QtGui.QWidget()
        self.contourTabLayout = QtGui.QGridLayout(debugTab)
        self.tabItems["Debug"] = debugTab


        # init edge view tab
        edgeView = QtGui.QWidget()
        self.edgeViewLayout = QtGui.QGridLayout(edgeView)
        self.tabItems["Edge Pairs"] = edgeView


        # init point cloud view tab
        pointCloudView = QtGui.QWidget()
        self.pointCloudViewLayout = QtGui.QGridLayout(pointCloudView)
        self.tabItems["Point Cloud"] = pointCloudView
        self.loadPointCloudWidget(self.pointCloudViewLayout)


        # Display Tabs
        if hidden == False:
            for name in self.tabItems:
                item = self.tabItems[name]
                self.tabWidget.addTab(item, name)

    def displayTab(self, name):
        if name not in self.tabItems.keys():
            raise RuntimeError("Tab name '{}' not defined.".format(name))
        tab = self.tabItems[name]
        if tab.parent() == self.tabWidget:
            return
        else:
            self.tabWidget.addTab(tab, name)

    def hideTab(self, name):
        if name not in self.tabItems.keys():
            raise RuntimeError("Tab name '{}' not defined.".format(name))
        tab = self.tabItems[name]
        self.tabWidget.removeTab(tab, name)

    def processImage(self):        
        currentlySelectedTabIndex = 0
        # ensure image has been selected
        if self.currentImageModel is None:
            raise RuntimeError("Unable to process Image: Image not loaded.")

        if self.isProcesed == True:
            log.info("Reprocessing Image.")
            self.clearImageCache(self.currentImageModel)
            currentlySelectedTabIndex = self.tabWidget.currentIndex()        

        log.info("Processing image: {}".format(self.currentImageModel.name))
        self.processRuns = self.processRuns + 1
        # crop image 
        trect = self.baseImage.getCropRectangle()
        self.currentImageModel.cropImage(trect.left(), trect.top(), trect.right(), trect.bottom())
        

        # routine for profiling 
        # cProfile.runctx("self.currentImageModel.getLineData()", globals(), locals())

        # point cloud
        pointCloud = self.currentImageModel.getPointCloudFromCrop()

        self.displayEdgeSegmentViewWidget(self.currentImageModel, self.contourTabLayout, self.isProcesed)

        self.displayEdgeInfoTab(self.currentImageModel, self.edgeViewLayout, self.isProcesed)

        self.displayEdgeAsPointCloud(self.currentImageModel)

        

        # itnitalize views by manually selecting first items
        if len(self.segmentList) > 0:
            self.segmentItemSelected(0)
            self.lineItemSelected(0)

        self.tabWidget.setCurrentIndex(currentlySelectedTabIndex)
        self.isProcesed = True
        self.processImgBtn.setText("Update Parameters")
        return

    def displayEdgeSegmentViewWidget(self, imageModel, segmentViewWrapper, retainIndex = False):
        try:
            self.segmentList
        except AttributeError:
            self.segmentList = {}
        else:
            del self.segmentList
            self.segmentList = {}

        segmentListImageViewIndex = 0
        try:
            self.segmentListImageView
        except AttributeError:
            self.segmentListImageView = pg.ImageView(name="ImageView")
            segmentViewWrapper.addWidget(self.segmentListImageView, 0,1,10,10)
        else:
            if retainIndex:
                segmentListImageViewIndex = self.segmentListImageView.currentIndex
            del self.segmentListImageView
            self.segmentListImageView = pg.ImageView(name="ImageView")
            segmentViewWrapper.addWidget(self.segmentListImageView, 0,1,10,10)

        segmentSelectBoxIndex = 0
        try:
            self.segmentSelectBox
        except AttributeError:
            self.segmentSelectBox = QtWdgt.QComboBox()
            segmentViewWrapper.addWidget(self.segmentSelectBox, 0,0,1,1)
        else:
            if retainIndex: # initialize segmentslect box to 0 on parameter update
                segmentSelectBoxIndex = 0 #self.segmentSelectBox.currentIndex
            self.segmentSelectBox.currentIndexChanged.disconnect()
            segmentViewWrapper.removeWidget(self.segmentSelectBox)
            self.segmentSelectBox.deleteLater()
            del self.segmentSelectBox
            self.segmentSelectBox = QtWdgt.QComboBox()
            segmentViewWrapper.addWidget(self.segmentSelectBox, 0,0,1,1)

        rgbImage = imageModel.getCroppedRGBImage()
        depthImage = cv2.cvtColor(imageModel.getCroppedDepthImage(), cv2.COLOR_GRAY2RGB)



        # Obtain Line data from image object (data is cached after first run)
        lineData = imageModel.getLineData()
        processed_images = lineData["processed_images"]

        timage_list = [rgbImage, depthImage]
        #timage_list.append(processed_images)
        timg = np.stack(timage_list)
        self.segmentListImageView.setImage(timg)
        self.segmentSelectBox.addItem("Show All")
        self.segmentSelectBox.addItem("None")        
        # Create Graphic object for each line (this is displayed on image)
        for s in range(len(lineData["segment_list"])):
            segments = lineData["segment_list"][s]
            segmentGroup = []
            segmentGroupName = "group-{}".format(s)
            for index in range(len(segments)-1):
                segmentName = "seg-{}-{}".format(s, index)
                startPos = QtCore.QPointF(segments[index][0], segments[index][1])
                endPos = QtCore.QPointF(segments[index+1][0], segments[index+1][1])
                lineSegment = QtCore.QLineF(startPos, endPos)
                color = np.random.rand(1, 1, 3).flatten() * 256
                pen = QtGui.QPen()
                pen.setColor(QtGui.QColor(color[0], color[1], color[2]))
                lineWdgt = QtWdgt.QGraphicsLineItem()                
                lineWdgt.setLine(lineSegment)
                lineWdgt.setPen(pen)
                # Add line widget to view item
                self.segmentListImageView.getView().addItem(lineWdgt)
                # hide line widget
                lineWdgt.hide()
                segmentGroup.append((segmentName, lineWdgt))
            self.segmentList[segmentGroupName] = segmentGroup
            self.segmentSelectBox.addItem(segmentGroupName)

        self.segmentSelectBox.currentIndexChanged.connect(self.segmentItemSelected)

        if retainIndex == True:
            self.segmentListImageView.setCurrentIndex(segmentListImageViewIndex)
            # self.segmentSelectBox.setCurrentIndex(segmentSelectBoxIndex)
        else:
            self.displayTab("Debug")

    def displayEdgeInfoTab(self, imageModel, viewWrapper, retainIndex):
        self.lineViews = {}
        self.currentLineItemSelected = None

        lineDataViewIndex = 0
        try:
            self.lineDataView
        except AttributeError:
            self.lineDataView = pg.ImageView(name="ImageView")
        else:
            if retainIndex:
                lineDataViewIndex = self.lineDataView.currentIndex
            viewWrapper.removeWidget(self.lineDataView)
            self.lineDataView.deleteLater()
            del self.lineDataView
            self.lineDataView = pg.ImageView(name="ImageView")

        lineSelectBoxIndex = 0
        try:
            self.lineSelectBox
        except AttributeError:
            self.lineSelectBox = QtWdgt.QComboBox()
            self.lineSelectBox.addItem("All Line Pairs")
        else:
            self.lineSelectBox.currentIndexChanged.disconnect()
            if retainIndex:
                lineSelectBoxIndex = self.lineSelectBox.currentIndex()
            viewWrapper.removeWidget(self.lineSelectBox)
            self.lineSelectBox.deleteLater()
            del self.lineSelectBox
            self.lineSelectBox = QtWdgt.QComboBox()
            self.lineSelectBox.addItem("All Line Pairs")


        rgbImage = imageModel.getCroppedRGBImage()
        depthImage = cv2.cvtColor(imageModel.getCroppedDepthImage(), cv2.COLOR_GRAY2RGB)
        timg = np.stack((rgbImage, depthImage))


        self.lineDataView.setImage(timg)

        viewWrapper.addWidget(self.lineSelectBox, 0,0,1,1)
        viewWrapper.addWidget(self.lineDataView, 0,1,10,10)        

        # Obtain Line data from image object (data is cached after first run)
        lineData = self.currentImageModel.getLineData()

        # Create Graphic object for each line (this is displayed on image)
        for edgePair in lineData["edge_pairs"]:
            lineViewPair = []
            for index in range(2):
                edge = edgePair[index]
                lineWdgt = QtWdgt.QGraphicsLineItem()
                lineWdgt.setLine(edge)
                lineWdgt.setPen(edge.getRenderData())
                self.lineDataView.addItem(lineWdgt)
                lineWdgt.hide()
                lineViewPair.append(lineWdgt)
            self.lineViews[edgePair.getID()] = (edgePair, lineViewPair)
            self.lineSelectBox.addItem(edgePair.getID())

        self.lineSelectBox.currentIndexChanged.connect(self.lineItemSelected)

        # Display points that are part of a line

        # Checkbox for displaying points
        displayContourPoints = QtWdgt.QCheckBox("Display Edge Points")
        viewWrapper.addWidget(displayContourPoints, 1,0,1,1)        
        self.shiftEdgeBtn = QtGui.QPushButton("Shift Edges")
        # self.shiftEdgeBtn.clicked.connect(self.shiftEdge)
        self.shiftEdgeBtn.hide()
        viewWrapper.addWidget(self.shiftEdgeBtn, 2,0,1,1)


        self.processFaceBtn = QtGui.QPushButton("Grip Pair")
        self.processFaceBtn.clicked.connect(self.processFace)
        self.processFaceBtn.hide()
        viewWrapper.addWidget(self.processFaceBtn, 3,0,1,1)

        self.pointViewItems = []
        self.showEdgePoints = False

        def _showEdgePoints(enabled):
            if enabled == 2:
                self.showEdgePoints = True
            else:
                self.showEdgePoints = False
            self.lineItemSelected(self.lineSelectBox.currentIndex())

        displayContourPoints.stateChanged.connect(_showEdgePoints)

        if retainIndex == True:
            self.lineDataView.setCurrentIndex(lineDataViewIndex)
        else:
            self.displayTab("Edge Pairs")

    def lineItemSelected(self, itemKey):
        strVal = self.lineSelectBox.itemText(itemKey)

        if strVal == "All Line Pairs": # Display all items
            self.displayEdgesOnImage(self.currentImageModel, showPoints=self.showEdgePoints)
            self.displayEdgeAsPointCloud(self.currentImageModel)
            self.currentEdgePair = None            
            self.shiftEdgeBtn.hide()
            self.processFaceBtn.hide()
        else:
            edgePair = self.lineViews[strVal][0]
            self.currentEdgePair = edgePair
            self.displayEdgeAsPointCloud(self.currentImageModel, edgePair)
            self.displayEdgesOnImage(self.currentImageModel, edgePair, showPoints=self.showEdgePoints)
            # self.shiftEdgeBtn.show()
            self.processFaceBtn.show()

    def segmentItemSelected(self, itemKey):
        strVal = self.segmentSelectBox.itemText(itemKey)
        segmentGroup = []
        if strVal not in ("None", "Show All"):
            segmentGroup = self.segmentList[strVal]
        elif strVal == "Show All":
            for name in self.segmentList:
                segmentGroup = segmentGroup + self.segmentList[name] 

        vb = self.segmentListImageView.getView()
        # ensure no segments are in the image veiw
        for key in self.segmentList:
            for segmentItem in self.segmentList[key]:
                segmentItem[1].hide()                

        # draw all line segments
        for segmentItem in segmentGroup:
            segmentItem[1].show()

    def processFace(self):
        """
        Using ransac, calculates the normals for the currently selected edge pairs and displays a vectors
        corresponding the orientatoin of the face.
        """
        self.glWidget.processFace(self.currentEdgePair)
        return
        length = 10
        width = 2
        params = self.edgeProcessorContext.processFace(self.currentImageModel, self.currentEdgePair)

        position = params[0]
        eigenVectors  = params[2].astype(float)
        direction = np.asarray((eigenVectors[0][0], eigenVectors[0][1], eigenVectors[0][2]))
        latitude = np.asarray((eigenVectors[1][0], eigenVectors[1][1], eigenVectors[1][2]))
        normal = np.asarray((eigenVectors[2][0], eigenVectors[2][1], eigenVectors[2][2]))
        posFinal = np.vstack([position, normal])
        normalLine = gl.GLLinePlotItem(pos=np.vstack([
                np.asarray([position[0], position[1], position[2]]),
                np.asarray([normal[0]*length+position[0], normal[1]*length+position[1], normal[2]*length+position[2]])
                ]), 
                color=(1,1,1,1), mode="lines", width=width)
        normalLine.setTransform(self.pointCloudViewModel.transform())
        directionLine = gl.GLLinePlotItem(pos=np.vstack([
                np.asarray([position[0], position[1], position[2]]),
                np.asarray([direction[0]*length+position[0], direction[1]*length+position[1], direction[2]*length+position[2]])
                ]), 
                color=(1,0,0,1), mode="lines", width=width)
        directionLine.setTransform(self.pointCloudViewModel.transform())
        latitudeLine = gl.GLLinePlotItem(pos=np.vstack([
                np.asarray([position[0], position[1], position[2]]),
                np.asarray([latitude[0]*length+position[0], latitude[1]*length+position[1], latitude[2]*length+position[2]])
                ]), 
                color=(0,1,0,1), mode="lines", width=width)
        latitudeLine.setTransform(self.pointCloudViewModel.transform())


        # self.glWidget.addItem(axisItem)
        self.glWidget.addItem(normalLine)
        self.glWidget.addItem(latitudeLine)
        self.glWidget.addItem(directionLine)

    def displayEdgesOnImage(self, imageModel, edgePair = None, showPoints = False):
        """
        Renders the currently selected edgepair on the processed image
        """
        vb = self.lineDataView.getView()
        lineData = imageModel.getLineData()
        edgePairList = None
        if edgePair == None:
            edgePairList = lineData["edge_pairs"]
        else:
            edgePairList = (edgePair,)


        for key in self.lineViews:
            self.lineViews[key][1][0].hide()
            self.lineViews[key][1][1].hide()

        for pv in self.pointViewItems:            
            vb.removeItem(pv)

        self.pointViewItems = []

        for edgePair in edgePairList:
            wdgPair = []
            for m in range(2):
                edge = edgePair[m]
                lineWdgt = self.lineViews[edgePair.getID()][1][m]
                lineWdgt.setLine(edge)
                lineWdgt.setPen(edge.getRenderData())
                if showPoints == False:
                    lineWdgt.show()                    
                    continue
                else:
                    lineWdgt.hide()                    

                pointList = edge.getEdgePointCloudIndexes()
                size = 1
                if showPoints == True:                
                    for point in pointList:
                        pointView = QtWdgt.QGraphicsRectItem()
                        pointView.setRect(point[0]-size, point[1]-size, size, size)
                        pointView.setPen(edge.getRenderData())
                        # pointView.show()
                        vb.addItem(pointView)
                        self.pointViewItems.append(pointView)

    def displayEdgeAsPointCloud(self, imageModel, edgePair=None):
        log.info("Rendering Point Cloud.")

        if edgePair is None:
            lineData = imageModel.getLineData()
            edgePairList = lineData["edge_pairs"]
        else:
            edgePairList = (edgePair,)
        pointCloud = imageModel.getPointCloud()


        self.glWidget.setData(
            imageModel=imageModel,
            edgePairList=edgePairList,
            pointCloud=pointCloud,
            context=self.getContext())
        
        self.displayTab("Point Cloud")

    def loadPointCloudWidget(self, viewWrapper):
        """ Loads widget which will be used to display point cloud """
        try:
            self.glWidget
        except AttributeError:
            self.glWidget = SceneElement()
        else:
            self.glWidget = SceneElement()

        viewWrapper.addWidget(self.glWidget, 0, 0, 10, 10)

    def getContext(self):
        return self.edgeProcessorContext

# applicatoin access point
def main():
    app = QtGui.QApplication(sys.argv)
    ex = App(app)
    sys.exit(app.exec_())

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    ex = App(app)
    sys.exit(app.exec_())
