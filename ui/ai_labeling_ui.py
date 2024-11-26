# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ai_labeling.ui'
##
## Created by: Qt User Interface Compiler version 6.7.0
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
from PySide6.QtWidgets import (QAbstractItemView, QAbstractSpinBox, QApplication, QHBoxLayout,
    QLabel, QLayout, QListWidget, QListWidgetItem,
    QPushButton, QSizePolicy, QSpacerItem, QSpinBox,
    QVBoxLayout, QWidget)
import resource_rc

class Ui_labeling_window(object):
    def setupUi(self, labeling_window):
        if not labeling_window.objectName():
            labeling_window.setObjectName(u"labeling_window")
        labeling_window.setWindowModality(Qt.WindowModality.ApplicationModal)
        labeling_window.resize(1280, 720)
        labeling_window.setMaximumSize(QSize(999999, 9999999))
        labeling_window.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        labeling_window.setWindowTitle(u"Labeling")
        labeling_window.setStyleSheet(u"background-color: rgb(3, 3, 13);")
        self.verticalLayout_6 = QVBoxLayout(labeling_window)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.top_logo_2 = QLabel(labeling_window)
        self.top_logo_2.setObjectName(u"top_logo_2")
        self.top_logo_2.setMinimumSize(QSize(202, 32))
        self.top_logo_2.setMaximumSize(QSize(202, 32))
        font = QFont()
        font.setFamilies([u"Sans"])
        self.top_logo_2.setFont(font)
        self.top_logo_2.setPixmap(QPixmap(u":/newPrefix/ui/logo.png"))
        self.top_logo_2.setScaledContents(True)

        self.horizontalLayout_6.addWidget(self.top_logo_2)

        self.YOLO_bnt = QPushButton(labeling_window)
        self.YOLO_bnt.setObjectName(u"YOLO_bnt")
        self.YOLO_bnt.setMinimumSize(QSize(61, 31))
        self.YOLO_bnt.setMaximumSize(QSize(61, 31))
        font1 = QFont()
        font1.setFamilies([u"NanumSquareRound"])
        font1.setPointSize(10)
        font1.setBold(False)
        self.YOLO_bnt.setFont(font1)
        self.YOLO_bnt.setCursor(QCursor(Qt.PointingHandCursor))
        self.YOLO_bnt.setStyleSheet(u"QPushButton {\n"
"    background-color: rgb(36, 39, 44);\n"
"    color: rgb(255, 255, 255);\n"
"    border-radius: 15px;\n"
"}\n"
"\n"
"QPushButton:checked {\n"
"    background-color: rgb(30, 195, 55);\n"
"    color: rgb(255, 255, 255);\n"
"    border-radius: 15px;\n"
"}")
        self.YOLO_bnt.setCheckable(False)

        self.horizontalLayout_6.addWidget(self.YOLO_bnt)

        self.SAM2_bnt = QPushButton(labeling_window)
        self.SAM2_bnt.setObjectName(u"SAM2_bnt")
        self.SAM2_bnt.setMinimumSize(QSize(61, 31))
        self.SAM2_bnt.setMaximumSize(QSize(61, 31))
        self.SAM2_bnt.setFont(font1)
        self.SAM2_bnt.setCursor(QCursor(Qt.PointingHandCursor))
        self.SAM2_bnt.setStyleSheet(u"QPushButton {\n"
"    background-color: rgb(36, 39, 44);\n"
"    color: rgb(255, 255, 255);\n"
"    border-radius: 15px;\n"
"}\n"
"\n"
"QPushButton:checked {\n"
"    background-color: rgb(30, 195, 55);\n"
"    color: rgb(255, 255, 255);\n"
"    border-radius: 15px;\n"
"}")
        self.SAM2_bnt.setCheckable(True)

        self.horizontalLayout_6.addWidget(self.SAM2_bnt)

        self.label_setting_bnt = QPushButton(labeling_window)
        self.label_setting_bnt.setObjectName(u"label_setting_bnt")
        self.label_setting_bnt.setMinimumSize(QSize(61, 31))
        self.label_setting_bnt.setMaximumSize(QSize(61, 31))
        self.label_setting_bnt.setFont(font1)
        self.label_setting_bnt.setCursor(QCursor(Qt.PointingHandCursor))
        self.label_setting_bnt.setStyleSheet(u"background-color: rgb(36, 39, 44);\n"
"color: rgb(255, 255, 255);\n"
"border-radius: 15px;\n"
"")

        self.horizontalLayout_6.addWidget(self.label_setting_bnt)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_6.addItem(self.horizontalSpacer_4)

        self.shutdown_bnt = QPushButton(labeling_window)
        self.shutdown_bnt.setObjectName(u"shutdown_bnt")
        self.shutdown_bnt.setMinimumSize(QSize(61, 31))
        self.shutdown_bnt.setMaximumSize(QSize(61, 31))
        self.shutdown_bnt.setFont(font1)
        self.shutdown_bnt.setCursor(QCursor(Qt.PointingHandCursor))
        self.shutdown_bnt.setStyleSheet(u"background-color: rgb(237, 51, 59);\n"
"color: rgb(255, 255, 255);\n"
"border-radius: 15px;\n"
"")

        self.horizontalLayout_6.addWidget(self.shutdown_bnt)


        self.horizontalLayout_9.addLayout(self.horizontalLayout_6)


        self.verticalLayout_4.addLayout(self.horizontalLayout_9)

        self.verticalSpacer_2 = QSpacerItem(20, 1, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self.verticalLayout_4.addItem(self.verticalSpacer_2)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_info_widget = QWidget(labeling_window)
        self.label_info_widget.setObjectName(u"label_info_widget")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_info_widget.sizePolicy().hasHeightForWidth())
        self.label_info_widget.setSizePolicy(sizePolicy)
        self.label_info_widget.setMinimumSize(QSize(251, 510))
        self.label_info_widget.setMaximumSize(QSize(251, 9999))
        self.verticalLayout_2 = QVBoxLayout(self.label_info_widget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.img_dir_folder_select_bnt = QPushButton(self.label_info_widget)
        self.img_dir_folder_select_bnt.setObjectName(u"img_dir_folder_select_bnt")
        self.img_dir_folder_select_bnt.setMinimumSize(QSize(80, 31))
        self.img_dir_folder_select_bnt.setMaximumSize(QSize(235, 30))
        font2 = QFont()
        font2.setFamilies([u"NanumSquareRound"])
        font2.setPointSize(9)
        font2.setBold(False)
        self.img_dir_folder_select_bnt.setFont(font2)
        self.img_dir_folder_select_bnt.setCursor(QCursor(Qt.PointingHandCursor))
        self.img_dir_folder_select_bnt.setStyleSheet(u"\n"
"background-color: rgb(36, 39, 44);\n"
"color: rgb(255, 255, 255);\n"
"border-radius: 15px;\n"
"\n"
"")

        self.verticalLayout_2.addWidget(self.img_dir_folder_select_bnt)

        self.label_dir_folder_select_bnt = QPushButton(self.label_info_widget)
        self.label_dir_folder_select_bnt.setObjectName(u"label_dir_folder_select_bnt")
        self.label_dir_folder_select_bnt.setMinimumSize(QSize(80, 31))
        self.label_dir_folder_select_bnt.setMaximumSize(QSize(233, 31))
        self.label_dir_folder_select_bnt.setFont(font2)
        self.label_dir_folder_select_bnt.setCursor(QCursor(Qt.PointingHandCursor))
        self.label_dir_folder_select_bnt.setStyleSheet(u"\n"
"background-color: rgb(36, 39, 44);\n"
"color: rgb(255, 255, 255);\n"
"border-radius: 15px;\n"
"\n"
"")

        self.verticalLayout_2.addWidget(self.label_dir_folder_select_bnt)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label_list = QListWidget(self.label_info_widget)
        self.label_list.setObjectName(u"label_list")
        self.label_list.setMaximumSize(QSize(233, 16777215))
        font3 = QFont()
        font3.setPointSize(10)
        self.label_list.setFont(font3)
        self.label_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.label_list.setStyleSheet(u"QTableWidget {\n"
"    background-color: rgb(13, 16, 23); /* \ud14c\uc774\ube14 \uc804\uccb4 \ubc30\uacbd\uc0c9 */\n"
"    color: rgb(255, 255, 255);\n"
"}\n"
"\n"
"QHeaderView::section {\n"
"    color: rgb(209, 209, 209); /* \ud5e4\ub354 \ud14d\uc2a4\ud2b8 \uc0c9\uc0c1 - \ud68c\uc0c9 */\n"
"background-color: rgb(7, 7, 16); /* \ud14c\uc774\ube14 \uc804\uccb4 \ubc30\uacbd\uc0c9 */\n"
"}\n"
"\n"
"QListWidget::item {\n"
"    color: rgb(255, 255, 255); /* \uae30\ubcf8 \uc0c1\ud0dc\uc5d0\uc11c\uc758 \ud14d\uc2a4\ud2b8 \uc0c9\uc0c1 - \ud770\uc0c9 */\n"
"\n"
"}\n"
"\n"
"QListWidget::item:selected {\n"
"    background-color: rgb(140, 167, 123); /* \uc120\ud0dd\ub41c \uc140\uc758 \ubc30\uacbd\uc0c9 */\n"
"    color: rgb(255, 255, 255);\n"
"\n"
"}\n"
"\n"
"QScrollBar:vertical {\n"
"    border: 1px solid #999999;\n"
"    background: #b3b3b3c6;\n"
"    width: 8px;\n"
"}\n"
"QScrollBar::handle:vertical {\n"
"background: #2f2f2f; \n"
"min-height: 10px;\n"
"width: 8px;\n"
"\n"
"}\n"
"")
        self.label_list.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.label_list.setDragEnabled(False)
        self.label_list.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)

        self.verticalLayout.addWidget(self.label_list)

        self.widget = QWidget(self.label_info_widget)
        self.widget.setObjectName(u"widget")
        self.widget.setStyleSheet(u"    QWidget {\n"
"background-color: rgb(36, 39, 44);\n"
"color: rgb(255, 255, 255);\n"
"border-radius: 10px;\n"
"\n"
"}")
        self.verticalLayout_8 = QVBoxLayout(self.widget)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_9 = QVBoxLayout()
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.object_conf_label = QLabel(self.widget)
        self.object_conf_label.setObjectName(u"object_conf_label")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.object_conf_label.sizePolicy().hasHeightForWidth())
        self.object_conf_label.setSizePolicy(sizePolicy1)
        self.object_conf_label.setMinimumSize(QSize(106, 31))
        self.object_conf_label.setMaximumSize(QSize(111, 16777215))
        font4 = QFont()
        font4.setFamilies([u"Sans Serif"])
        font4.setPointSize(10)
        font4.setBold(False)
        self.object_conf_label.setFont(font4)
        self.object_conf_label.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.object_conf_label.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_11.addWidget(self.object_conf_label)

        self.object_conf_value = QSpinBox(self.widget)
        self.object_conf_value.setObjectName(u"object_conf_value")
        sizePolicy1.setHeightForWidth(self.object_conf_value.sizePolicy().hasHeightForWidth())
        self.object_conf_value.setSizePolicy(sizePolicy1)
        self.object_conf_value.setMinimumSize(QSize(70, 31))
        self.object_conf_value.setMaximumSize(QSize(16777215, 31))
        self.object_conf_value.setFont(font3)
        self.object_conf_value.setStyleSheet(u"\n"
"    QSpinBox {\n"
"        subcontrol-origin: padding;\n"
"        subcontrol-position: top right;\n"
"        background: rgb(13, 16, 23);\n"
"        color: rgb(255, 255, 255);\n"
"		border-radius: 1px;\n"
"	\n"
"\n"
"\n"
"    }\n"
"\n"
"\n"
"\n"
"QSpinBox::up-button {\n"
"    subcontrol-origin: border;\n"
"    subcontrol-position: top right; /* \uc5c5 \ubc84\ud2bc \uc704\uce58 */\n"
"    width: 20px;\n"
"    height: 20px;\n"
"}\n"
"\n"
"\n"
"\n"
"QSpinBox::down-button {\n"
"    subcontrol-origin: border;\n"
"    subcontrol-position: bottom right; /* \uc5c5 \ubc84\ud2bc \uc704\uce58 */\n"
"    width: 20px;\n"
"    height: 20px;\n"
"}")
        self.object_conf_value.setMinimum(1)
        self.object_conf_value.setMaximum(100)
        self.object_conf_value.setValue(33)

        self.horizontalLayout_11.addWidget(self.object_conf_value)


        self.verticalLayout_9.addLayout(self.horizontalLayout_11)

        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.object_IOU_label = QLabel(self.widget)
        self.object_IOU_label.setObjectName(u"object_IOU_label")
        sizePolicy1.setHeightForWidth(self.object_IOU_label.sizePolicy().hasHeightForWidth())
        self.object_IOU_label.setSizePolicy(sizePolicy1)
        self.object_IOU_label.setMinimumSize(QSize(106, 31))
        self.object_IOU_label.setMaximumSize(QSize(111, 16777215))
        self.object_IOU_label.setFont(font4)
        self.object_IOU_label.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.object_IOU_label.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_12.addWidget(self.object_IOU_label)

        self.object_IOU_value = QSpinBox(self.widget)
        self.object_IOU_value.setObjectName(u"object_IOU_value")
        sizePolicy1.setHeightForWidth(self.object_IOU_value.sizePolicy().hasHeightForWidth())
        self.object_IOU_value.setSizePolicy(sizePolicy1)
        self.object_IOU_value.setMinimumSize(QSize(70, 31))
        self.object_IOU_value.setMaximumSize(QSize(16777215, 31))
        self.object_IOU_value.setFont(font3)
        self.object_IOU_value.setStyleSheet(u"\n"
"    QSpinBox {\n"
"        subcontrol-origin: padding;\n"
"        subcontrol-position: top right;\n"
"        background: rgb(13, 16, 23);\n"
"        color: rgb(255, 255, 255);\n"
"		border-radius: 1px;\n"
"	\n"
"\n"
"\n"
"    }\n"
"\n"
"\n"
"\n"
"QSpinBox::up-button {\n"
"    subcontrol-origin: border;\n"
"    subcontrol-position: top right; /* \uc5c5 \ubc84\ud2bc \uc704\uce58 */\n"
"\n"
"    width: 20px;\n"
"    height: 20px;\n"
"}\n"
"\n"
"\n"
"\n"
"QSpinBox::down-button {\n"
"    subcontrol-origin: border;\n"
"    subcontrol-position: bottom right; /* \uc5c5 \ubc84\ud2bc \uc704\uce58 */\n"
"    width: 20px;\n"
"    height: 20px;\n"
"}")
        self.object_IOU_value.setMinimum(1)
        self.object_IOU_value.setMaximum(100)
        self.object_IOU_value.setSingleStep(1)
        self.object_IOU_value.setStepType(QAbstractSpinBox.StepType.DefaultStepType)
        self.object_IOU_value.setValue(50)
        self.object_IOU_value.setDisplayIntegerBase(10)

        self.horizontalLayout_12.addWidget(self.object_IOU_value)


        self.verticalLayout_9.addLayout(self.horizontalLayout_12)


        self.verticalLayout_8.addLayout(self.verticalLayout_9)


        self.verticalLayout.addWidget(self.widget)


        self.verticalLayout_2.addLayout(self.verticalLayout)


        self.horizontalLayout_2.addWidget(self.label_info_widget)

        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.label_class_widget = QWidget(labeling_window)
        self.label_class_widget.setObjectName(u"label_class_widget")
        self.label_class_widget.setMinimumSize(QSize(0, 48))
        self.label_class_widget.setMaximumSize(QSize(9999, 48))
        self.verticalLayout_5 = QVBoxLayout(self.label_class_widget)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setSpacing(6)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.cls_0 = QPushButton(self.label_class_widget)
        self.cls_0.setObjectName(u"cls_0")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.cls_0.sizePolicy().hasHeightForWidth())
        self.cls_0.setSizePolicy(sizePolicy2)
        self.cls_0.setMinimumSize(QSize(55, 25))
        self.cls_0.setMaximumSize(QSize(55, 25))
        font5 = QFont()
        font5.setFamilies([u"Sans"])
        font5.setPointSize(10)
        self.cls_0.setFont(font5)
        self.cls_0.setCursor(QCursor(Qt.PointingHandCursor))
        self.cls_0.setStyleSheet(u"\n"
"QPushButton{\n"
"background-color: rgb(255, 56, 56);\n"
"color: rgb(255, 255, 255);\n"
"border-radius: 9px;\n"
"border: 1px solid rgba(255, 255, 255, 100);\n"
"}\n"
"\n"
"\n"
"QPushButton:checked {\n"
"                color: white;\n"
"				border-radius: 9px;\n"
"				border: 2px solid rgb(255, 255, 255);\n"
"\n"
"            }")
        self.cls_0.setCheckable(True)

        self.horizontalLayout.addWidget(self.cls_0)


        self.horizontalLayout_7.addLayout(self.horizontalLayout)

        self.horizontalSpacer_2 = QSpacerItem(562, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_7.addItem(self.horizontalSpacer_2)

        self.label_del_bnt = QPushButton(self.label_class_widget)
        self.label_del_bnt.setObjectName(u"label_del_bnt")
        sizePolicy2.setHeightForWidth(self.label_del_bnt.sizePolicy().hasHeightForWidth())
        self.label_del_bnt.setSizePolicy(sizePolicy2)
        self.label_del_bnt.setMinimumSize(QSize(102, 25))
        self.label_del_bnt.setMaximumSize(QSize(9999, 25))
        self.label_del_bnt.setFont(font5)
        self.label_del_bnt.setCursor(QCursor(Qt.PointingHandCursor))
        self.label_del_bnt.setStyleSheet(u"border-radius: 9px;\n"
"background-color: rgb(255, 49, 38);\n"
"color: rgb(255, 255, 255);")

        self.horizontalLayout_7.addWidget(self.label_del_bnt)


        self.horizontalLayout_8.addLayout(self.horizontalLayout_7)


        self.verticalLayout_5.addLayout(self.horizontalLayout_8)


        self.verticalLayout_3.addWidget(self.label_class_widget)

        self.label_image_viewer = QLabel(labeling_window)
        self.label_image_viewer.setObjectName(u"label_image_viewer")
        sizePolicy2.setHeightForWidth(self.label_image_viewer.sizePolicy().hasHeightForWidth())
        self.label_image_viewer.setSizePolicy(sizePolicy2)
        self.label_image_viewer.setMinimumSize(QSize(640, 460))
        self.label_image_viewer.setMaximumSize(QSize(9999, 9999))
        self.label_image_viewer.setFont(font)
        self.label_image_viewer.setStyleSheet(u"border: 1px solid rgb(119, 118, 123);\n"
"border-radius: 10px ;")
        self.label_image_viewer.setTextFormat(Qt.TextFormat.PlainText)
        self.label_image_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_3.addWidget(self.label_image_viewer)


        self.horizontalLayout_2.addLayout(self.verticalLayout_3)


        self.verticalLayout_4.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalSpacer = QSpacerItem(940, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_save_bnt = QPushButton(labeling_window)
        self.label_save_bnt.setObjectName(u"label_save_bnt")
        sizePolicy1.setHeightForWidth(self.label_save_bnt.sizePolicy().hasHeightForWidth())
        self.label_save_bnt.setSizePolicy(sizePolicy1)
        self.label_save_bnt.setMinimumSize(QSize(76, 39))
        self.label_save_bnt.setMaximumSize(QSize(76, 39))
        self.label_save_bnt.setFont(font5)
        self.label_save_bnt.setCursor(QCursor(Qt.PointingHandCursor))
        self.label_save_bnt.setStyleSheet(u"background-color: rgb(30, 195, 55);\n"
"color: rgb(255, 255, 255);\n"
"border-radius: 19px;\n"
"\n"
"")

        self.horizontalLayout_4.addWidget(self.label_save_bnt)

        self.label_data_del_bnt = QPushButton(labeling_window)
        self.label_data_del_bnt.setObjectName(u"label_data_del_bnt")
        sizePolicy1.setHeightForWidth(self.label_data_del_bnt.sizePolicy().hasHeightForWidth())
        self.label_data_del_bnt.setSizePolicy(sizePolicy1)
        self.label_data_del_bnt.setMinimumSize(QSize(76, 39))
        self.label_data_del_bnt.setMaximumSize(QSize(76, 39))
        self.label_data_del_bnt.setFont(font5)
        self.label_data_del_bnt.setCursor(QCursor(Qt.PointingHandCursor))
        self.label_data_del_bnt.setStyleSheet(u"background-color: rgb(255, 49, 38);\n"
"color: rgb(255, 255, 255);\n"
"border-radius: 19px;\n"
"")

        self.horizontalLayout_4.addWidget(self.label_data_del_bnt)


        self.horizontalLayout_5.addLayout(self.horizontalLayout_4)


        self.verticalLayout_4.addLayout(self.horizontalLayout_5)


        self.verticalLayout_6.addLayout(self.verticalLayout_4)


        self.retranslateUi(labeling_window)

        QMetaObject.connectSlotsByName(labeling_window)
    # setupUi

    def retranslateUi(self, labeling_window):
        self.top_logo_2.setText("")
        self.YOLO_bnt.setText(QCoreApplication.translate("labeling_window", u"YOLO", None))
        self.SAM2_bnt.setText(QCoreApplication.translate("labeling_window", u"SAM2", None))
        self.label_setting_bnt.setText(QCoreApplication.translate("labeling_window", u"\ub77c\ubca8 \uc124\uc815", None))
        self.shutdown_bnt.setText(QCoreApplication.translate("labeling_window", u"\ub2eb\uae30", None))
        self.img_dir_folder_select_bnt.setText(QCoreApplication.translate("labeling_window", u"\uc774\ubbf8\uc9c0 \ud3f4\ub354 \uc120\ud0dd", None))
        self.label_dir_folder_select_bnt.setText(QCoreApplication.translate("labeling_window", u"\ub77c\ubca8 \ud3f4\ub354 \uc120\ud0dd", None))
        self.object_conf_label.setText(QCoreApplication.translate("labeling_window", u"object conf", None))
        self.object_IOU_label.setText(QCoreApplication.translate("labeling_window", u"object IOU", None))
        self.cls_0.setText(QCoreApplication.translate("labeling_window", u"person", None))
        self.label_del_bnt.setText(QCoreApplication.translate("labeling_window", u"\ub77c\ubca8 \ubaa8\ub450 \uc0ad\uc81c", None))
        self.label_image_viewer.setText("")
        self.label_save_bnt.setText(QCoreApplication.translate("labeling_window", u"\uc800\uc7a5", None))
        self.label_data_del_bnt.setText(QCoreApplication.translate("labeling_window", u"\uc0ad\uc81c", None))
        pass
    # retranslateUi

