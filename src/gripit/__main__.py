
import PyQt5.QtGui as QtGui
import sys
from app import App

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    ex = App(app)
    sys.exit(app.exec_())