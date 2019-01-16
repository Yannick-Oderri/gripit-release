from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import int
from builtins import str
from future import standard_library
standard_library.install_aliases()
from builtins import object
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
from enum import Enum


class PointCloudFilter(Enum):
    ALL = 0
    NEGATE = 1
    ONLY = 2


class PointCloudModel(object):
    """ Point cloud representation """

    def __init__(self, **kwds):
        """ All Keyword arguments are sent to setData """
        self.pointCloudData = None
        self.imageModel = None
        self.renderProperties = {
            'pointCloud': {
                'color': (0.1, 0.1, 1, 0.4),
                'size': 2.0
            }
        }
        self._setValues(**kwds)

    def _setValues(self, **kwds):

        args = ('pointCloudData', 'imageModel', 'renderProperties', 'context')
        for k in kwds.keys():
            if k not in args:
                raise Exception('Invalid keyword argument: %s (allowed arguments are %s)' % (k, str(args)))

        for arg in args:
            if arg in kwds:
                setattr(self, arg, kwds[arg])
        self.update()

    def update(self):
        pass

    def getPointCloudData(self):
        return self.pointCloudData

    def getFilteredCloudData(self, indexList, filterType=None):
        if filterType is None:
            return self.getPointCloudData()

    def getPoint(self, index):
        return self.pointCloudData[index]

    def getPointfromXYCoordinate(self, x, y):
        index = self.getIndexfromXYCoordinate(x, y)
        return self.getPoint(index)

    def getIndexfromXYCoordinate(self, x, y):
        shape = self.imageModel.getCroppedDepthImage().shape
        indexx = int(y * shape[1] + x % shape[1])
        return indexx

    def saveToFile(self, outName):
        handle = self.context.getFileWriter(outName)
        for point in self.pointCloudData:
            handle.write("[{}, {}, {}]\n".format(point[0], point[1], point[2]))
        handle.close()
