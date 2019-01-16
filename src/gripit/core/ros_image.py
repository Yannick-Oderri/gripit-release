import rospy
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import threading
import copy
import PyQt5.QtCore as QtCore
from skimage import img_as_uint

class OnTopicReived(QtCore.QObject):
    receivedSignal = QtCore.pyqtSignal(np.ndarray)
ROS_DEPTH_IMAGE = 1
ROS_COLOR_IMAGE = 0

class ROSImage:
    def __init__(self, topic, imgType=ROS_COLOR_IMAGE):
        self.topic = str(topic)
        self.lock = threading.RLock()
        self.OnReceived = None
        self.bridge = CvBridge()
        self.data = np.array([])
        self.shape = (480, 640, 3)
        self.imgType = imgType
        rospy.init_node("test", anonymous=True)

    def _callback(self, ros_msg_img):
        frame = None
        if self.imgType == ROS_COLOR_IMAGE:
            frame = self._callbackColor(ros_msg_img)
        else:
            frame = self._callbackDepth(ros_msg_img)


        if self.data.shape[0] <= 1:
            self.data = np.zeros(frame.shape, frame.dtype)

        self.lock.acquire()
        self._copyFrame(frame)
        self.lock.release()

        if self.OnReceived != None:
            self.OnReceived.receivedSignal.emit(self.getImageData())
        
        #if self.imgType == ROS_DEPTH_IMAGE:
        #    cv2.imshow(self.topic, self.getImageData())


    def _callbackColor(self, ros_msg_img):
        # Use cv_bridge() to convert the ROS image to OpenCV format
        #print("RGB Callback")        
        try:
            frame = self.bridge.imgmsg_to_cv2(ros_msg_img, "bgr8")
        except CvBridgeError, e:
            print e   
        # Convert the image to a Numpy array since most cv2 functions
        # require Numpy arrays.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.array(frame, dtype=np.uint8)

                    
    def _callbackDepth(self, ros_image):
        # Use cv_bridge() to convert the ROS image to OpenCV format
        #print("Depth Callback")
        try:
            # The depth image is a single-channel float32 image
            depth_image = self.bridge.imgmsg_to_cv2(ros_image, "32FC1")
        except CvBridgeError, e:
            print e

        # Convert the depth image to a Numpy array since most cv2 functions
        # require Numpy arrays.
        depth_array = np.array(depth_image, dtype=np.float32)
        # Normalize the depth image to fall between 0 (black) and 1 (white)
        cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
        if self.imgType == ROS_DEPTH_IMAGE:
            cv2.imshow(self.topic, depth_array)
        #depth_array = depth_array.astype(np.uint16)
        depth_array = img_as_uint(depth_array)
        max = np.amax(depth_array)
        min = np.amin(depth_array)
        cv2.convertScaleAbs(depth_array, depth_array, 6500./(max-min), -min)
        #cv2.normalize(depth_array, depth_array, 0, 20000, cv2.NORM_MINMAX)
        #depth_array = depth_array.astype(np.u)
        #depth_array[ depth_array > 1400 ] = 0
        depth_array = depth_array/50
        # depth_array[ depth_array > 1400 ] = 0
        return depth_array

    def pauseTopic(self):
        pass # TODO

    def getCurrentFrame(self):
        data = self.getImageData()
        frame = np.zeros(data.shape, data.dtype)
        np.copyto(frame, data)
        
        if self.imgType != ROS_COLOR_IMAGE:
            cv2.imwrite("/tmp/depthWrite.png", frame)
        
        return frame


    def _copyFrame(self, frame):
        np.copyto(self.data, frame)


    def subscribe(self, cb=None):
        print("Subscribing to topic: {}".format(self.topic))

        if cb == None:
            rospy.Subscriber(self.topic, Image, self._callback)
        else:
            rospy.Subscriber(self.topic, Image, cb)


    def registerReceiveSignal(self, signal):
        self.OnReceived = signal

    def unregisterReceiveSignal(self):
        self.registerReceiveSignal(None)
    
    def getImageData(self):
        data = None
        self.lock.acquire()
        data = self.data
        self.lock.release()

        return data