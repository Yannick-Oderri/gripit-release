from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import super
from builtins import str
from future import standard_library
standard_library.install_aliases()
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QSlider, QWidget, QVBoxLayout, QHBoxLayout, QLabel
__author__ = 'Andres'


class QLabelSlider(QWidget):
    def __init__(self, sliderOrientation=None):
        super(QLabelSlider, self).__init__()
        self._slider = QSlider(sliderOrientation)
        self.setLayout(QVBoxLayout())
        self._labelTicksWidget = QWidget(self)
        self._labelTicksWidget.setLayout(QHBoxLayout())
        self._labelTicksWidget.layout().setContentsMargins(0, 0, 0, 0)

        self.layout().addWidget(self._slider)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(self._labelTicksWidget)

    def setTickLabels(self, listWithLabels):
        return
        lengthOfList = len(listWithLabels)
        for index, label in enumerate(listWithLabels):
            label.setFont(QtGui.QFont("Sans", 10, QtGui.QFont.Bold))
            label.setContentsMargins(0, 0, 0, 0)
            if index > lengthOfList/3:
                label.setAlignment(QtCore.Qt.AlignCenter)
            if index > 2*lengthOfList/3:
                label.setAlignment(QtCore.Qt.AlignRight)
            self._labelTicksWidget.layout().addWidget(label)

    def setRange(self, mini, maxi):
        self._slider.setRange(mini, maxi)
        self._labels = [QLabel(str(mini)), QLabel(str(mini)), QLabel(str(maxi))]
        self.setTickLabels(self._labels)

    def setPageStep(self, value):
        self._slider.setPageStep(value)

    def setTickInterval(self, value):
        self._slider.setTickInterval(value)

    def setTickPosition(self, position):
        self._slider.setTickPosition(position)

    def setValue(self, value):
        self._labels[1].setText(str(value))
        self._slider.setValue(value)

    def onValueChangedCall(self, function):
        def _valueChanged(val):
            self._labels[1].setText(str(val))
            function(val)
        self._slider.valueChanged.connect(_valueChanged)