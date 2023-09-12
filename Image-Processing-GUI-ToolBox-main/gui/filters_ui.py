# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'filters_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_filterDialog(object):
    def setupUi(self, filterDialog):
        filterDialog.setObjectName("filterDialog")
        filterDialog.resize(1141, 461)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(filterDialog.sizePolicy().hasHeightForWidth())
        filterDialog.setSizePolicy(sizePolicy)
        filterDialog.setMinimumSize(QtCore.QSize(1141, 461))
        filterDialog.setMaximumSize(QtCore.QSize(1141, 461))
        self.frame = QtWidgets.QFrame(filterDialog)
        self.frame.setGeometry(QtCore.QRect(10, 10, 1121, 51))
        self.frame.setStyleSheet("QFrame#frame{\n"
"border-radius: 4px;\n"
"border: 2px solid rgb(229, 229, 229);\n"
"}")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label_4 = QtWidgets.QLabel(self.frame)
        self.label_4.setGeometry(QtCore.QRect(10, 0, 401, 41))
        self.label_4.setStyleSheet("color: #FFF;\n"
"font: 75 20pt \"Berlin Sans FB Demi\";")
        self.label_4.setTextFormat(QtCore.Qt.PlainText)
        self.label_4.setObjectName("label_4")
        self.close_button = QtWidgets.QPushButton(self.frame)
        self.close_button.setGeometry(QtCore.QRect(1090, 20, 17, 17))
        self.close_button.setMinimumSize(QtCore.QSize(16, 16))
        self.close_button.setMaximumSize(QtCore.QSize(17, 17))
        self.close_button.setStyleSheet("QPushButton {\n"
"    border: none;\n"
"    border-radius: 8px;\n"
"    background-color: rgb(255, 0, 0);\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: rgba(255, 0, 0, 150);\n"
"}")
        self.close_button.setText("")
        self.close_button.setObjectName("close_button")
        self.minimize_button = QtWidgets.QPushButton(self.frame)
        self.minimize_button.setGeometry(QtCore.QRect(1060, 20, 17, 17))
        self.minimize_button.setMinimumSize(QtCore.QSize(16, 16))
        self.minimize_button.setMaximumSize(QtCore.QSize(17, 17))
        self.minimize_button.setStyleSheet("QPushButton {\n"
"    border: none;\n"
"    border-radius: 8px;\n"
"    background-color: rgb(255, 170, 0);\n"
"}\n"
"QPushButton:hover {\n"
"background-color: rgba(255, 170, 0, 150);\n"
"}")
        self.minimize_button.setText("")
        self.minimize_button.setObjectName("minimize_button")
        self.frame_2 = QtWidgets.QFrame(filterDialog)
        self.frame_2.setGeometry(QtCore.QRect(10, 70, 1121, 381))
        self.frame_2.setStyleSheet("QFrame#frame_2{\n"
"border-radius: 4px;\n"
"border: 2px solid rgb(229, 229, 229);\n"
"}")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.tabs = QtWidgets.QTabWidget(self.frame_2)
        self.tabs.setGeometry(QtCore.QRect(10, 10, 541, 361))
        self.tabs.setWhatsThis("")
        self.tabs.setObjectName("tabs")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.slider1 = QtWidgets.QSlider(self.tab)
        self.slider1.setGeometry(QtCore.QRect(220, 40, 31, 221))
        self.slider1.setMinimum(1)
        self.slider1.setMaximum(31)
        self.slider1.setSingleStep(1)
        self.slider1.setTracking(True)
        self.slider1.setOrientation(QtCore.Qt.Vertical)
        self.slider1.setObjectName("slider1")
        self.label = QtWidgets.QLabel(self.tab)
        self.label.setGeometry(QtCore.QRect(200, 270, 71, 21))
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setObjectName("label")
        self.slider1_counter = QtWidgets.QLabel(self.tab)
        self.slider1_counter.setGeometry(QtCore.QRect(230, 10, 31, 31))
        self.slider1_counter.setTextFormat(QtCore.Qt.AutoText)
        self.slider1_counter.setObjectName("slider1_counter")
        self.dx_checkBox = QtWidgets.QCheckBox(self.tab)
        self.dx_checkBox.setGeometry(QtCore.QRect(120, 110, 81, 20))
        self.dx_checkBox.setChecked(True)
        self.dx_checkBox.setObjectName("dx_checkBox")
        self.dy_checkBox = QtWidgets.QCheckBox(self.tab)
        self.dy_checkBox.setGeometry(QtCore.QRect(120, 150, 81, 20))
        self.dy_checkBox.setObjectName("dy_checkBox")
        self.apply_button = QtWidgets.QPushButton(self.tab)
        self.apply_button.setEnabled(True)
        self.apply_button.setGeometry(QtCore.QRect(420, 290, 93, 28))
        self.apply_button.setObjectName("apply_button")
        self.tabs.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.label_2 = QtWidgets.QLabel(self.tab_2)
        self.label_2.setGeometry(QtCore.QRect(200, 270, 71, 21))
        self.label_2.setTextFormat(QtCore.Qt.AutoText)
        self.label_2.setObjectName("label_2")
        self.slider2_counter = QtWidgets.QLabel(self.tab_2)
        self.slider2_counter.setGeometry(QtCore.QRect(230, 10, 31, 31))
        self.slider2_counter.setTextFormat(QtCore.Qt.AutoText)
        self.slider2_counter.setObjectName("slider2_counter")
        self.slider2 = QtWidgets.QSlider(self.tab_2)
        self.slider2.setGeometry(QtCore.QRect(220, 40, 31, 221))
        self.slider2.setMinimum(1)
        self.slider2.setMaximum(31)
        self.slider2.setSingleStep(1)
        self.slider2.setOrientation(QtCore.Qt.Vertical)
        self.slider2.setObjectName("slider2")
        self.apply_button_2 = QtWidgets.QPushButton(self.tab_2)
        self.apply_button_2.setEnabled(True)
        self.apply_button_2.setGeometry(QtCore.QRect(420, 290, 93, 28))
        self.apply_button_2.setObjectName("apply_button_2")
        self.tabs.addTab(self.tab_2, "")
        self.image_label = QtWidgets.QLabel(self.frame_2)
        self.image_label.setGeometry(QtCore.QRect(560, 30, 550, 340))
        self.image_label.setFrameShape(QtWidgets.QFrame.Box)
        self.image_label.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.image_label.setText("")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setObjectName("image_label")

        self.retranslateUi(filterDialog)
        self.tabs.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(filterDialog)

    def retranslateUi(self, filterDialog):
        _translate = QtCore.QCoreApplication.translate
        filterDialog.setWindowTitle(_translate("filterDialog", "Filters"))
        self.label_4.setText(_translate("filterDialog", "Filters"))
        self.label.setText(_translate("filterDialog", "Kernel Size"))
        self.slider1_counter.setText(_translate("filterDialog", "0"))
        self.dx_checkBox.setText(_translate("filterDialog", "Dx"))
        self.dy_checkBox.setText(_translate("filterDialog", "Dy"))
        self.apply_button.setText(_translate("filterDialog", "Apply Filter"))
        self.tabs.setTabText(self.tabs.indexOf(self.tab), _translate("filterDialog", "Sobel Filter"))
        self.label_2.setText(_translate("filterDialog", "Kernel Size"))
        self.slider2_counter.setText(_translate("filterDialog", "0"))
        self.apply_button_2.setText(_translate("filterDialog", "Apply Filter"))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_2), _translate("filterDialog", "Laplace Filter"))

    def getComponents(self):
        return [self.tabs,
                self.slider1, self.slider2,
                self.slider1_counter, self.slider2_counter,
                self.dx_checkBox, self.dy_checkBox,
                self.apply_button, self.apply_button_2]