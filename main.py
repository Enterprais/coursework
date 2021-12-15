import Window as win
import sys
from PyQt5 import (QtWidgets)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = win.MainWindow()
    window.show()
    app.exec_()


