
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QFileDialog, QLabel, QMessageBox, QPushButton
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont
import cv2
import os
from trainedModel import *
import datetime

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        # Setting up the GUI
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowIcon(QIcon('icon.png'))
        MainWindow.resize(1920, 1080)
        MainWindow.setLayoutDirection(QtCore.Qt.LeftToRight)
        # other action buttons
        self.Start = QPushButton(MainWindow)
        self.Start.setGeometry(QtCore.QRect(168, 900, 121, 30))
        self.Start.setObjectName("Start")
        self.Start.setStyleSheet(
            "QPushButton::hover""{""background-color:lightgreen;""}""QPushButton""{""border:None;border-style: outset; border-radius:5px; background-color:rgb(255, 170, 127)""}")

        self.Exit = QPushButton(MainWindow)
        self.Exit.setGeometry(QtCore.QRect(1700, 20, 100, 25))
        self.Exit.setStyleSheet(
            "QPushButton::hover""{""background-color:rgb(255,0,0);""}""QPushButton""{""border:None;border-style: outset; border-radius:5px; background-color:rgb(230, 0, 0)""}")
        self.Exit.setObjectName("Exit")

        self.OriginalIamge = QLabel(MainWindow)
        self.OriginalIamge.setGeometry(QtCore.QRect(100, 100, 830, 371))
        self.OriginalIamge.setBaseSize(QtCore.QSize(0, 0))
        self.OriginalIamge.setAutoFillBackground(True)
        self.OriginalIamge.setStyleSheet("border: 2px solid rgb(0,0,0); border-radius: 5px")  # Stylesheet
        self.OriginalIamge.setLineWidth(3)
        self.OriginalIamge.setTextFormat(QtCore.Qt.PlainText)
        self.OriginalIamge.setAlignment(QtCore.Qt.AlignCenter)
        self.OriginalIamge.setOpenExternalLinks(False)
        self.OriginalIamge.setObjectName("OriginalIamge")

        self.DetectedImage = QLabel(MainWindow)
        self.DetectedImage.setGeometry(QtCore.QRect(940, 475, 830, 371))
        self.DetectedImage.setBaseSize(QtCore.QSize(0, 0))
        self.DetectedImage.setAutoFillBackground(True)
        self.DetectedImage.setStyleSheet("border: 2px solid rgb(0,0,0); border-radius: 5px")  # Stylesheet
        self.DetectedImage.setLineWidth(3)
        self.DetectedImage.setTextFormat(QtCore.Qt.PlainText)
        self.DetectedImage.setAlignment(QtCore.Qt.AlignCenter)
        self.DetectedImage.setOpenExternalLinks(False)
        self.DetectedImage.setObjectName("DetectedImage")

        # Mendatory settings button
        self.Browse = QPushButton(MainWindow)
        self.Browse.setGeometry(QtCore.QRect(20, 50, 130, 25))
        self.Browse.setStyleSheet(
            "QPushButton::hover""{""background-color:lightgreen;""}""QPushButton""{""border:None;border-style: outset; border-radius:5px; background-color:rgb(255, 170, 127)""}")
        font = QFont()
        font.setBold(False)
        font.setWeight(50)
        self.Browse.setFont(font)
        self.Browse.setAutoFillBackground(False)
        # self.Browse.setAlignment(QtCore.Qt.AlignCenter)
        self.Browse.setObjectName("Browse")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Defining the clicking actions of GUI buttons
        self.Start.clicked.connect(self.start_scanning)
        self.Browse.clicked.connect(self.open)
        self.Exit.clicked.connect(self.exit_gui)

        # required variables
        self.filename = None


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate(
            "MainWindow", "CBCT Teeth image scanner"))
        self.Exit.setText(_translate("MainWindow", "Exit"))
        self.Browse.setText(
            _translate("MainWindow", "Browse"))
        self.Start.setText(_translate("MainWindow", "Scan"))
        self.OriginalIamge.setText(_translate(
            "MainWindow", "Original Image will be shown here!"))
        self.DetectedImage.setText(_translate(
            "MainWindow", "Image with detections, will be shown here!"))


    # to browse the image to be scanned
    def open(self):
        self.fileName, _ = QFileDialog.getOpenFileName(MainWindow, "Open File", QtCore.QDir.currentPath())
        if self.fileName:
            image = cv2.imread(str(self.fileName))
            # print(image)
            if image.shape[0]==0:
                QMessageBox.information(MainWindow, "Image Viewer", "Cannot load %s." % self.fileName)
                return
            self.displayTo(image,self.OriginalIamge)
    

    # to run the detection model
    def start_scanning(self):
        if self.fileName==None:
            # messsage box
            msg = QMessageBox()
            msg.setWindowTitle("Alert!")
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please upload an image first\n" +
                        "then try again..")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return

        self.DetectedImage.setText("Please wait, we're processing your X-ray...")
        self.DetectedImage.setAlignment(QtCore.Qt.AlignCenter)
        QApplication.processEvents()

        # load the pre-trained model
        try:
            model = load_saved_model()
        except Exception as e:
            print("Could not load the pre-trained model, becuase {}".format(e))
            return

        # run detection
        try:
            detection_results, original_image = make_prediction(model, self.fileName)
        except:
            print("some erro occurred while running the detection...\n" + "try again...")
            return

        bbox = detection_results[0]['rois']

        # mark the detections on the original
        img_with_detection = make_visualization(original_image, detection_results)

        self.displayTo(img_with_detection[250:550, 105:718],self.DetectedImage)


    # To display the image image in GUI, with or without detection
    def displayTo(self, image, frame):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (827, 367))
        height, width, channel = image.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(image, width, height, step, QImage.Format_RGB888)
        frame.setPixmap(QPixmap.fromImage(qImg))
        frame.setAlignment(QtCore.Qt.AlignHCenter)
        QApplication.processEvents()


    def return_cpu_temp(self):
        os.popen("cd")
        os.popen("cd ../..")
        temp = str(int(os.popen(
            "cat /sys/devices/virtual/thermal/thermal_zone6/temp").read()) // 1000) + " ℃"
        return temp


    # To open Baumer viewer to configure and change, camera settings
    def open_buammer_settings(self):
        print("Please wait.............")
        try:
            sudoPass = "pi"
            command = "/opt/baumer-camera-explorer/bin/bexplorer"
            os.popen("sudo -S %s" % (command), 'w').write(sudoPass)
        except Exception as e:
            # messsage box
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Exitself.Exit)
            msg.setText(str(e))
            msg.setWindowTitle("Alert!")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()


    # To close the GUI
    def exit_gui(self):
        print("Thanks for using me!")
        sys.exit()

    # calculate date and time
    def get_date_time(self):
        date = datetime.datetime.now()
        return date.strftime("%d-%m-%Y %H:%M:%S")


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QDialog()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())