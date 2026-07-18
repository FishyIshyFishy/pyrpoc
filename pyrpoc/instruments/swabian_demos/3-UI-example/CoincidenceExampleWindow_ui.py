# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'CoincidenceExampleWindow.ui'
##
## Created by: Qt User Interface Compiler version 6.10.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QDoubleSpinBox, QGridLayout,
    QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSizePolicy, QSpacerItem, QSpinBox, QVBoxLayout,
    QWidget)

class Ui_CoincidenceExample(object):
    def setupUi(self, CoincidenceExample):
        if not CoincidenceExample.objectName():
            CoincidenceExample.setObjectName(u"CoincidenceExample")
        CoincidenceExample.resize(889, 753)
        self.centralwidget = QWidget(CoincidenceExample)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setEnabled(True)
        self.centralwidget.setLayoutDirection(Qt.LeftToRight)
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(20, 20, 20, 20)
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setHorizontalSpacing(15)
        self.label_13 = QLabel(self.centralwidget)
        self.label_13.setObjectName(u"label_13")

        self.gridLayout.addWidget(self.label_13, 0, 4, 1, 1)

        self.delayB = QSpinBox(self.centralwidget)
        self.delayB.setObjectName(u"delayB")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.delayB.sizePolicy().hasHeightForWidth())
        self.delayB.setSizePolicy(sizePolicy)
        self.delayB.setMinimum(-99999)
        self.delayB.setMaximum(99999)

        self.gridLayout.addWidget(self.delayB, 4, 2, 1, 1)

        self.channelA = QSpinBox(self.centralwidget)
        self.channelA.setObjectName(u"channelA")
        sizePolicy.setHeightForWidth(self.channelA.sizePolicy().hasHeightForWidth())
        self.channelA.setSizePolicy(sizePolicy)
        self.channelA.setMinimum(-99)
        self.channelA.setValue(1)

        self.gridLayout.addWidget(self.channelA, 2, 1, 1, 1)

        self.channelB = QSpinBox(self.centralwidget)
        self.channelB.setObjectName(u"channelB")
        sizePolicy.setHeightForWidth(self.channelB.sizePolicy().hasHeightForWidth())
        self.channelB.setSizePolicy(sizePolicy)
        self.channelB.setMinimum(-99)
        self.channelB.setValue(2)

        self.gridLayout.addWidget(self.channelB, 4, 1, 1, 1)

        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 0, 2, 1, 1)

        self.delayA = QSpinBox(self.centralwidget)
        self.delayA.setObjectName(u"delayA")
        sizePolicy.setHeightForWidth(self.delayA.sizePolicy().hasHeightForWidth())
        self.delayA.setSizePolicy(sizePolicy)
        self.delayA.setMinimum(-99999)
        self.delayA.setMaximum(99999)

        self.gridLayout.addWidget(self.delayA, 2, 2, 1, 1)

        self.label_11 = QLabel(self.centralwidget)
        self.label_11.setObjectName(u"label_11")

        self.gridLayout.addWidget(self.label_11, 4, 0, 1, 1)

        self.label_10 = QLabel(self.centralwidget)
        self.label_10.setObjectName(u"label_10")

        self.gridLayout.addWidget(self.label_10, 2, 0, 1, 1)

        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 1, 1, 1)

        self.label_5 = QLabel(self.centralwidget)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout.addWidget(self.label_5, 0, 6, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer, 2, 5, 1, 1)

        self.label_12 = QLabel(self.centralwidget)
        self.label_12.setObjectName(u"label_12")

        self.gridLayout.addWidget(self.label_12, 0, 3, 1, 1)

        self.triggerB = QDoubleSpinBox(self.centralwidget)
        self.triggerB.setObjectName(u"triggerB")
        self.triggerB.setMinimum(-2.500000000000000)
        self.triggerB.setMaximum(2.500000000000000)
        self.triggerB.setSingleStep(0.100000000000000)
        self.triggerB.setValue(0.500000000000000)

        self.gridLayout.addWidget(self.triggerB, 4, 3, 1, 1)

        self.testsignalA = QCheckBox(self.centralwidget)
        self.testsignalA.setObjectName(u"testsignalA")
        self.testsignalA.setLayoutDirection(Qt.LeftToRight)
        self.testsignalA.setAutoFillBackground(False)
        self.testsignalA.setStyleSheet(u"")
        self.testsignalA.setChecked(False)
        self.testsignalA.setAutoRepeat(False)

        self.gridLayout.addWidget(self.testsignalA, 2, 4, 1, 1, Qt.AlignHCenter)

        self.triggerA = QDoubleSpinBox(self.centralwidget)
        self.triggerA.setObjectName(u"triggerA")
        self.triggerA.setMinimum(-2.500000000000000)
        self.triggerA.setMaximum(2.500000000000000)
        self.triggerA.setSingleStep(0.100000000000000)
        self.triggerA.setValue(0.500000000000000)

        self.gridLayout.addWidget(self.triggerA, 2, 3, 1, 1)

        self.testsignalB = QCheckBox(self.centralwidget)
        self.testsignalB.setObjectName(u"testsignalB")

        self.gridLayout.addWidget(self.testsignalB, 4, 4, 1, 1, Qt.AlignHCenter)

        self.coincidenceWindow = QSpinBox(self.centralwidget)
        self.coincidenceWindow.setObjectName(u"coincidenceWindow")
        sizePolicy.setHeightForWidth(self.coincidenceWindow.sizePolicy().hasHeightForWidth())
        self.coincidenceWindow.setSizePolicy(sizePolicy)
        self.coincidenceWindow.setMinimum(1)
        self.coincidenceWindow.setMaximum(9999)
        self.coincidenceWindow.setValue(1000)

        self.gridLayout.addWidget(self.coincidenceWindow, 0, 7, 1, 1)

        self.label_6 = QLabel(self.centralwidget)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout.addWidget(self.label_6, 2, 6, 1, 1)

        self.correlationBinwidth = QSpinBox(self.centralwidget)
        self.correlationBinwidth.setObjectName(u"correlationBinwidth")
        sizePolicy.setHeightForWidth(self.correlationBinwidth.sizePolicy().hasHeightForWidth())
        self.correlationBinwidth.setSizePolicy(sizePolicy)
        self.correlationBinwidth.setMinimum(1)
        self.correlationBinwidth.setMaximum(9999)
        self.correlationBinwidth.setValue(40)

        self.gridLayout.addWidget(self.correlationBinwidth, 2, 7, 1, 1)

        self.label_7 = QLabel(self.centralwidget)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout.addWidget(self.label_7, 4, 6, 1, 1)

        self.correlationBins = QSpinBox(self.centralwidget)
        self.correlationBins.setObjectName(u"correlationBins")
        sizePolicy.setHeightForWidth(self.correlationBins.sizePolicy().hasHeightForWidth())
        self.correlationBins.setSizePolicy(sizePolicy)
        self.correlationBins.setMinimum(1)
        self.correlationBins.setMaximum(99999)
        self.correlationBins.setValue(1000)

        self.gridLayout.addWidget(self.correlationBins, 4, 7, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout)

        self.plotWidget = QWidget(self.centralwidget)
        self.plotWidget.setObjectName(u"plotWidget")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.plotWidget.sizePolicy().hasHeightForWidth())
        self.plotWidget.setSizePolicy(sizePolicy1)
        self.verticalLayout_2 = QVBoxLayout(self.plotWidget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, -1, 0, -1)

        self.verticalLayout.addWidget(self.plotWidget)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setSpacing(40)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.startButton = QPushButton(self.centralwidget)
        self.startButton.setObjectName(u"startButton")

        self.horizontalLayout_2.addWidget(self.startButton)

        self.stopButton = QPushButton(self.centralwidget)
        self.stopButton.setObjectName(u"stopButton")

        self.horizontalLayout_2.addWidget(self.stopButton)

        self.clearButton = QPushButton(self.centralwidget)
        self.clearButton.setObjectName(u"clearButton")

        self.horizontalLayout_2.addWidget(self.clearButton)

        self.saveButton = QPushButton(self.centralwidget)
        self.saveButton.setObjectName(u"saveButton")

        self.horizontalLayout_2.addWidget(self.saveButton)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        CoincidenceExample.setCentralWidget(self.centralwidget)
#if QT_CONFIG(shortcut)
        self.label_2.setBuddy(self.label_2)
#endif // QT_CONFIG(shortcut)
        QWidget.setTabOrder(self.channelA, self.delayA)
        QWidget.setTabOrder(self.delayA, self.channelB)
        QWidget.setTabOrder(self.channelB, self.delayB)
        QWidget.setTabOrder(self.delayB, self.startButton)
        QWidget.setTabOrder(self.startButton, self.stopButton)
        QWidget.setTabOrder(self.stopButton, self.clearButton)
        QWidget.setTabOrder(self.clearButton, self.saveButton)

        self.retranslateUi(CoincidenceExample)

        QMetaObject.connectSlotsByName(CoincidenceExample)
    # setupUi

    def retranslateUi(self, CoincidenceExample):
        CoincidenceExample.setWindowTitle(QCoreApplication.translate("CoincidenceExample", u"CoincidenceExample", None))
        self.label_13.setText(QCoreApplication.translate("CoincidenceExample", u"Test signal", None))
        self.delayB.setSuffix(QCoreApplication.translate("CoincidenceExample", u" ps", None))
        self.label_2.setText(QCoreApplication.translate("CoincidenceExample", u"Input delay", None))
        self.delayA.setSuffix(QCoreApplication.translate("CoincidenceExample", u" ps", None))
        self.label_11.setText(QCoreApplication.translate("CoincidenceExample", u"B:", None))
        self.label_10.setText(QCoreApplication.translate("CoincidenceExample", u"A:", None))
        self.label.setText(QCoreApplication.translate("CoincidenceExample", u"Input channel", None))
        self.label_5.setText(QCoreApplication.translate("CoincidenceExample", u"Coincidence window", None))
        self.label_12.setText(QCoreApplication.translate("CoincidenceExample", u"Trigger level", None))
        self.triggerB.setSuffix(QCoreApplication.translate("CoincidenceExample", u" V", None))
        self.testsignalA.setText("")
        self.triggerA.setSuffix(QCoreApplication.translate("CoincidenceExample", u" V", None))
        self.testsignalB.setText("")
        self.coincidenceWindow.setSuffix(QCoreApplication.translate("CoincidenceExample", u" ps", None))
        self.label_6.setText(QCoreApplication.translate("CoincidenceExample", u"Correlation bin width", None))
        self.correlationBinwidth.setSuffix(QCoreApplication.translate("CoincidenceExample", u" ps", None))
        self.label_7.setText(QCoreApplication.translate("CoincidenceExample", u"Correlation bins", None))
        self.startButton.setText(QCoreApplication.translate("CoincidenceExample", u"Start", None))
        self.stopButton.setText(QCoreApplication.translate("CoincidenceExample", u"Stop", None))
        self.clearButton.setText(QCoreApplication.translate("CoincidenceExample", u"Clear", None))
        self.saveButton.setText(QCoreApplication.translate("CoincidenceExample", u"Save", None))
    # retranslateUi

