from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from builtins import open
from builtins import str
from builtins import object
from future import standard_library
standard_library.install_aliases()
from pathlib import Path
import logging as log
import collections, os, json, re
from enum import Enum
import pyqtgraph as pg
import cv2
import rospy
from .ros_image import *
from .ros_image_model import ROSImageElement

# local imports
from .image_model import ImageElement
import gripit.core.ynk_utils as util

# settings.py
pg.setConfigOptions(imageAxisOrder='row-major')
__CONTEXT = None


class ExecutionMode(Enum):
    """
    DEVELOPER_MODE > 5
    """
    # Sets development level
    DEVELOPMENT = 5
    DEVELOPMENT_1 = 5
    DEVELOPMENT_2 = 6
    DEVELOPMENT_BLENDER = 1
    DEVELOPMENT_REAL = 2
    DEVELOPMENT_ROS = 3


class EdgeProcessingDetectContext(object):

    def __init__(self, dataStorePath, _mode):  # (fixme) setup mode switching
        log.basicConfig(level=log.NOTSET)
        self.__mode = _mode
        self.dataStorePath = dataStorePath
        dataStoreConfig = dataStorePath / "config.json"
        cwd = os.getcwd()
        if not dataStoreConfig.is_file():
            raise IOError("'{}' does not contain a configuration file.".format(cwd))

        with open(str(dataStoreConfig)) as config_file:
            self.dataStoreConfig = json.load(config_file)
        # THis is not necessary anymore ## initialize datastore

        if self.__mode == ExecutionMode.DEVELOPMENT_BLENDER:
            pass
        else: # else default mode
            self.modelPath = "./GripIt/resources/models/"
            self.outPath = "./outFile/"
            self.rgbNamePrefix = "clearn{}"
            self.depthNamePrefix = "learn{}"
            self.imgExt = "png"
            self.modelExt = "obj"
            # self.__mode = ExecutionMode.DEVELOPMENT_1
            self.focal_length = 500
            self.thresh = 20
            self.window_size = 10
            self.min_len = 20
            self.min_dist = 10
            self.max_dist = 200
            self.delta_angle = 20
            self.camera_pitch = 50
            self.camera_translate_x = 0
            self.camera_translate_y = 0
            self.camera_translate_z = 50

    def getColorImageConfig(self):
        return self.dataStoreConfig["colorImage"]

    def getRGBImagePath(self):
        return self.dataStorePath / self.dataStoreConfig["colorImage"]["path"]

    def getRGBImageName(self, index):
        return self.dataStoreConfig["colorImage"]["nameTemplate"].format(index)

    def getDepthImagePath(self):
        return self.dataStorePath / self.dataStoreConfig["depthImage"]["path"]

    def getDepthImageName(self, index):
        return self.dataStoreConfig["depthImage"]["nameTemplate"].format(index)

    def listAvailableImages(self):
        print("EXECUTING RUNTIME: {}".format(self.__mode))
        if self.__mode == ExecutionMode.DEVELOPMENT_ROS:
            return self.listAvailableROSImages()
        else:
            return self.listAvailableLocalImages()
    
    def listAvailableLocalImages(self):
        colorImgConfig = self.getColorImageConfig()
        fileExt = colorImgConfig["imgExt"]
        imageNameTemplate = colorImgConfig["nameTemplate"]
        prefix = imageNameTemplate.find("{")
        suffix = len(imageNameTemplate) - (imageNameTemplate.find("}"))

        images = list(self.getRGBImagePath().glob("*.{}".format(fileExt)))
        results = []
        for image in images:
            name = Path(image).stem
            index = name[prefix:len(name)]
            results.append((name, index))
        results.sort()
        return results

    # Finds avaialble images to be presented on image drop-down
    def listAvailableROSImages(self):
        targetMessages = ("sensor_msgs/CompressedImage", 
                            "sensor_msgs/Image",
                            "sensor_msgs/compressed")

        # get image topics
        topics = rospy.get_published_topics()
        images = []
        for topic in topics:
            for trgtMsg in targetMessages:
                if trgtMsg == topic[1]:
                    images.append(topic[0])

        index = 0
        results = []
        for image in images:
            # name = Path(image).stem            
            results.append((image, index))
            index = index + 1
        results.sort()
        return results

    def getDataStoreConfig(self):
        return self.dataStoreConfig

    def loadImage(self, index):
        if self.__mode == ExecutionMode.DEVELOPMENT_ROS:
            return self.loadROSImage(index[0], index[1])
        else:
            return self.loadLocalImage(index)

    def loadLocalImage(self, index):
        colorImgConfig = self.getColorImageConfig()
        imgExt = colorImgConfig["imgExt"]
        rgb_img_path = self.getRGBImagePath() / (self.getRGBImageName(index) + "." + imgExt)
        log.info("Loading Color Image: {}".format(rgb_img_path))

        # Check if image exists
        if not rgb_img_path.is_file():
            raise IOError("Image file: '%s' does not exist." % rgb_img_path)

        rgbImage = self.loadColorImage(rgb_img_path)

        depth_img_path = Path(self.getDepthImagePath() / (self.getDepthImageName(index) + "." + imgExt))
        log.info("Loading Depth Image: {}".format(depth_img_path))
        # Check if depth image is available
        if not depth_img_path.is_file():
            raise IOError("Depth image file: '%s' does not exist." % _img_path)
        depthImage = self.loadDepthImage(depth_img_path)

        try:
            # (fixme) cache image?
            currentImage = ImageElement(self, rgbImage, depthImage, self.getRGBImageName(index))
            ## Load external parameters if available
            parameterFilePath = self.dataStorePath / (self.dataStoreConfig["database"]["path"] + currentImage.getName())
            if parameterFilePath.is_file():
                with open(str(parameterFilePath)) as config_file:
                    config = json.load(config_file, object_pairs_hook=collections.OrderedDict)
                    currentImage.initializeAttribute(self, config)
        except IOError as err:
            raise err

        return currentImage


    def loadROSImage(self, rgbName, depthName):
        colorImgConfig = self.getColorImageConfig()
        rgb_img_path = rgbName
        log.info("Connecting to ROS topic: {}".format(rgb_img_path))

        rgbImage = ROSImage(rgb_img_path, 0)
        rgbImage.subscribe()
        
        depth_img_path = depthName
        log.info("Connecting to ROS topic: {}".format(depth_img_path))
        
        depth_img = ROSImage(depth_img_path, 1)
        depth_img.subscribe()
        
        try:
            # (fixme) cache image?
            currentImage = ROSImageElement(self, rgbImage, depth_img, str(rgb_img_path))
        except IOError as err:
            raise err

        return currentImage



    def loadColorImage(self, path):
        ext = Path(path).suffix
        img = None
        if ext == ".png":
            img = cv2.imread(str(path))
        elif ext == ".pcd":
            img,dimg,cloud = util.loadPCD(str(path))

        if "rotate" in self.getColorImageConfig():
            img = util.rotateImage(img, self.getColorImageConfig()["rotate"])

        return img

    def loadDepthImage(self, path):
        ext = Path(path).suffix
        dimg = None
        if ext == ".png":
            dimg = cv2.imread(str(path), -1)
        elif ext == ".pcd":
            img, dimg, cloud = util.loadPCD(str(path))

        if "rotate" in self.getColorImageConfig():
            dimg = util.rotateImage(dimg, self.getColorImageConfig()["rotate"])

        return dimg

    def loadModel(self, name):
        model_path = Path(self.modelPath + name + "." + self.modelExt)
        log.info("Loading Model: {}".format(model_path))
        model_data = None
        # check if file exists
        if not model_path.is_file():
            raise IOError("File does not exist.")

        try:
            parser = util.getOBJParser()
            parser.parseFile("{}".format(model_path))
            model_data = parser.getMeshData()
        except IOError as err:
            raise err
        else:
            return model_data

        return model_data


    def saveFiletoStore(self, path, data, callback):
        # Saves file 
        dataPath = None
        try:
            root = self.dataStorePath / Path(self.dataStoreConfig["outputDir"])            
            dataPath = root / path
        except KeyError as err:
            log.warn("Store config element, 'outputDir', not defined. \n Directory 'output' Created.")
            root = Path(self.dataStorePath / "out")
            if not root.is_dir():
                os.makedirs(str(root))
            dataPath = root / path
        try:
            if not dataPath.parent.is_dir():
                os.makedirs(str(dataPath.parent))
        except OSError as err:
            if not dataPath.parent.is_dir():
                raise err

        callback(dataPath, data)

    def processEdge(self, imageModel, edgePair):
        for i in range(2):
            util.shiftLineOnObject(imageModel, edgePair[i])
            util.fitLinetoEdgeModel(imageModel, edgePair[i])

    def saveImageModelParameters(self, params, name): # pass in image model instead of name
        # TODO if database using somthing other than folders to store aprameters
        filePath = self.dataStorePath / (self.dataStoreConfig["database"]["path"] + name)
        fileWriter = open(str(filePath), "w")
        try:
            content = json.dumps(params, separators=(',',':'))
            fileWriter.write(content)
        except IOError as err:
            raise err
        finally:
            fileWriter.close()

    def resetImageModelParameters(self, imageModel):
        name = imageModel.getName()
        filePath = self.dataStorePath / (self.dataStoreConfig["database"]["path"] + name)
        if filePath.is_file():
            os.remove(str(filePath))

    def processFace(self, imageModel, edgePair):
        return util.processFace(imageModel, edgePair)


    def getHelperClass(self):
        return util

    def getMode(self):
        """
        Description: Returns current mode of context
        """
        return self.__mode

    @staticmethod
    def initializeContext(dataStore, _mode):
        """
        Description:
            Initialize the program context which hold global information about running process
        Args: 
            dataStore: name of datastore to load images and dependent data
            _mode: staging mode to run code
        """
        dataStorePath = Path("./data/{}/".format(dataStore))
        log.warn(_mode)
        try:

            if not dataStorePath.is_dir():
                raise NotADirectoryError("Datastore '{}' could not be located.".format(dataStore))  
            __CONTEXT
        except NameError:
            log.info("Initializing Edge Processing Context")

            __CONTEXT = EdgeProcessingDetectContext(dataStorePath, _mode)
            return __CONTEXT
        else:
            return __CONTEXT

    @staticmethod
    def getContext():
        """ Returns application context        
        """

        try:
            __CONTEXT
        except NameError:            
            #return EdgeProcessingDetectContext.initializeContext()
            raise NameError("not available.")
        else:
            return __CONTEXT

if __name__ == '__main__':
    getContext()