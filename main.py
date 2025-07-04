import sys
import os
from pathlib import Path
import cv2
import numpy as np
import shutil

ROOT = Path(__file__).resolve().parents[1]

from PySide6.QtWidgets import QDialog, QTableWidgetItem, QLabel, QApplication, QListWidgetItem, QPushButton, QFileDialog
from datetime import datetime
from PySide6.QtCore import QTimer, QDate, Qt, QSize, Signal, QPoint, QRect, QEvent, QDir
from PySide6.QtGui import QImage, QPainter, QPen, QColor, QPolygon, QBrush, QMouseEvent, QPixmap, QCursor, QFont
from PySide6.QtWidgets import QSizePolicy

from ui import Ui_labeling_window

import yaml
import torch
import gc
import traceback
from ultralytics import YOLO
from sam2.build_sam import build_sam2_video_predictor

from multiprocessing import Process
torch.multiprocessing.set_start_method('spawn', force=True)
COLOR = {
    0: (220, 20, 60),   # Crimson - person
    1: (60, 179, 113),  # Medium Sea Green - bicycle
    2: (70, 130, 180),  # Steel Blue - car
    3: (255, 140, 0),   # Dark Orange - motorcycle
    4: (147, 112, 219), # Medium Purple - bus
    5: (72, 209, 204),  # Medium Turquoise - truck
    6: (255, 20, 147)   # Deep Pink - fire
}
class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))    

class Labeling_Viewer(QLabel):
    clicked = Signal(QPoint)  # 사용자가 클릭한 위치를 전달하는 시그널

    def __init__(self, parent):
        super().__init__(parent)
        
        self.point_list = []
        self.non_active_point_list = []
        self.parent = parent
        self.selected_box = None
        self.dragging = False
        self.resizing = False
        self.drawing_new_box = False
        self.drag_start_pos = None
        self.resize_corner = None
        self.mouse_pos = None
        self.box_resize_mode = False
        self.labeling_flag = False

        self.frame = None

        self.cls_bnt_list = []
        self.label_list = []

        self.class_name_dict = {}

    def make_cls_bnt(self, bnt_num, cls, color):
        cls_bnt = QPushButton(self.parent.label_ui.label_class_widget)
        bnt_name = f"{self.class_name_dict[cls]}"
        cls_bnt.setObjectName(bnt_name)
        cls_bnt.setMinimumSize(QSize(55, 25))
        cls_bnt.setMaximumSize(QSize(83, 25))

        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(cls_bnt.sizePolicy().hasHeightForWidth())
        cls_bnt.setSizePolicy(sizePolicy2)

        font = QFont()
        font.setFamilies([u"Sans"])
        font.setPointSize(10)
        cls_bnt.setFont(font)
        cls_bnt.setCursor(QCursor(Qt.PointingHandCursor))
        cls_bnt.setStyleSheet(
                                "QPushButton{\n"
                                f"background-color: rgb({color[0]}, {color[1]}, {color[2]});\n"
                                "color: rgb(255, 255, 255);\n"
                                "border-radius: 9px;\n"
                                "border: 1px solid rgba(255, 255, 255, 100);\n"
                                "}\n"
                                                
                                "QPushButton:checked {\n"
                                "color: white;\n"
                                "border-radius: 9px;\n"
                                "border: 2px solid rgb(255, 255, 255);\n"
                                "}"
                                )
        cls_bnt.setText(str(self.class_name_dict[cls]))
        cls_bnt.setCheckable(True)

        # 버튼이 체크될 때, 나머지 버튼의 체크를 해제하는 메서드 연결
        cls_bnt.toggled.connect(lambda checked, btn=cls_bnt: self.handle_button_toggled(checked, btn))

        self.parent.label_ui.horizontalLayout_7.insertWidget(bnt_num, cls_bnt)

        self.cls_bnt_list.append(cls_bnt)

    # 새로운 메서드 추가
    def handle_button_toggled(self, checked, button):
        if checked:
            # 현재 체크된 버튼을 제외한 나머지 버튼들은 체크 해제
            for btn in self.cls_bnt_list:
                if btn != button:
                    btn.setChecked(False)

    def display_image(self):
        if self.frame is not None:
            img = cv2.resize(self.frame, dsize=(self.width(), self.height()))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            height, width, channel = img.shape
            bytes_per_line = 3 * width
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.setPixmap(pixmap)
            self.update()

    def display_label(self, label_list):
        # for bnt in self.cls_bnt_list:
        #     try:
        #         bnt.deleteLater()
                
        #     except:
        #         pass
        
        # self.cls_bnt_list = []
        self.label_list = []
        for cls, xc, yc, w, h, color in label_list:
            self.label_list.append([int(cls), float(xc), float(yc), float(w), float(h), color])
            # self.cls_bnt_list.append(self.make_cls_bnt(len(self.cls_bnt_list), cls, color))
            
        self.update()


    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        for label in self.label_list:
            cls, xc, yc, w, h, color = label
            x1 = int((xc - w / 2) * self.width())
            y1 = int((yc - h / 2) * self.height())
            x2 = int((xc + w / 2) * self.width())
            y2 = int((yc + h / 2) * self.height())
            rect = QRect(x1, y1, x2 - x1, y2 - y1)

            pen = QPen(QColor(color[0], color[1], color[2]), 3)
            painter.setPen(pen)
            brush = QBrush(QColor(color[0], color[1], color[2], 100))  # 100 is the alpha value for transparency
            painter.setBrush(brush)

            painter.drawRect(rect)
            
            # Draw corners as circles
            corner_radius = 3
            painter.setBrush(QBrush(QColor(color[0], color[1], color[2])))
            corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
            for corner in corners:
                painter.drawEllipse(QPoint(corner[0], corner[1]), corner_radius, corner_radius)

        if self.drawing_new_box and self.drag_start_pos and self.mouse_pos:
            pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            brush = QBrush(QColor(0, 0, 0, 0))
            painter.setBrush(brush)

            painter.drawRect(QRect(self.drag_start_pos, self.mouse_pos))

        if self.mouse_pos:
            pen = QPen(QColor(81, 174, 50), 2)
            painter.setPen(pen)
            painter.drawLine(0, self.mouse_pos.y(), self.width(), self.mouse_pos.y())
            painter.drawLine(self.mouse_pos.x(), 0, self.mouse_pos.x(), self.height())

            corner = self.get_corner(self.mouse_pos)
            if corner:
                if corner == 'top_left':
                    self.setCursor(QCursor(Qt.SizeFDiagCursor))  # ↖↘ 방향
                elif corner == 'top_right':
                    self.setCursor(QCursor(Qt.SizeBDiagCursor))  # ↗↙ 방향
                elif corner == 'bottom_left':
                    self.setCursor(QCursor(Qt.SizeBDiagCursor))  # ↗↙ 방향
                elif corner == 'bottom_right':
                    self.setCursor(QCursor(Qt.SizeFDiagCursor))  # ↖↘ 방향
            else:
                self.setCursor(QCursor(Qt.ArrowCursor))  # 기본 포인터

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.labeling_flag:
            self.drag_start_pos = event.position().toPoint()
            self.drawing_new_box = True
            for index, label in enumerate(self.label_list):
                cls, xc, yc, w, h, color = label
                x1 = int((xc - w / 2) * self.width()) - 3
                y1 = int((yc - h / 2) * self.height()) - 3
                x2 = int((xc + w / 2) * self.width()) + 3
                y2 = int((yc + h / 2) * self.height()) + 3
                rect = QRect(x1, y1, x2 - x1, y2 - y1)

                if rect.contains(event.position().toPoint()):
                    self.drawing_new_box = False
                    self.selected_box = index
                    corner = self.get_resize_corner(event.position().toPoint())

                    if corner:
                        self.resizing = True
                        self.resize_corner = corner
                    else:
                        self.dragging = True
                    break

        if event.button() == Qt.RightButton:
            check_index = []
            check_distance = []

            for index, label in enumerate(self.label_list):
                cls, xc, yc, w, h, color = label
                x1 = int((xc - w / 2) * self.width())
                y1 = int((yc - h / 2) * self.height())
                x2 = int((xc + w / 2) * self.width())
                y2 = int((yc + h / 2) * self.height())
                rect = QRect(x1, y1, x2 - x1, y2 - y1)

                if rect.contains(event.position().toPoint()):
                    check_index.append(index)
                    check_distance.append(abs(event.position().toPoint().x() - xc*self.width()) + abs(event.position().toPoint().y() - yc*self.height()))

            if check_index:
                del_index = check_index[np.argmin(np.array(check_distance))]
                del self.label_list[del_index]

                # self.cls_bnt_list[del_index].deleteLater()
                # del self.cls_bnt_list[del_index]
                self.update()

    def mouseMoveEvent(self, event):
        self.mouse_pos = event.position().toPoint()
        if self.drawing_new_box:
            self.update()
        elif self.dragging and self.selected_box is not None:
            dx = event.position().toPoint().x() - self.drag_start_pos.x()
            dy = event.position().toPoint().y() - self.drag_start_pos.y()
            cls, xc, yc, w, h, color = self.label_list[self.selected_box]

            new_xc = (int(xc * self.width()) + dx) / self.width()
            new_yc = (int(yc * self.height()) + dy) / self.height()

            # xc = (int(xc * self.width()) + dx) / self.width()
            # yc = (int(yc * self.height()) + dy) / self.height()

            if (new_xc - w/2) <= 0 or (new_xc + w/2) >= 1 : xc = xc
            else: xc = new_xc
            if (new_yc - h/2) <= 0 or (new_yc + h/2) >= 1 : yc = yc
            else: yc = new_yc

            self.label_list[self.selected_box] = [cls, xc, yc, w, h, color]
            self.drag_start_pos = event.position().toPoint()
            self.update()
        elif self.resizing and self.selected_box is not None:
            self.resize_box(event.position().toPoint())
            self.update()
        else:
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing_new_box:
            self.create_new_box(event.position().toPoint())
        self.dragging = False
        self.resizing = False
        self.resize_corner = None
        self.drawing_new_box = False

    def enterEvent(self, event):
        self.setMouseTracking(True)
        self.mouse_pos = None
        self.update()

    def leaveEvent(self, event):
        self.setMouseTracking(False)
        self.mouse_pos = None
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.display_image()

    def get_resize_corner(self, pos):
        """Determine which corner is being dragged."""
        if self.selected_box is None or self.selected_box >= len(self.label_list):
            return None
        cls, xc, yc, w, h, color = self.label_list[self.selected_box]
        x1 = int((xc - w / 2) * self.width())
        y1 = int((yc - h / 2) * self.height())
        x2 = int((xc + w / 2) * self.width())
        y2 = int((yc + h / 2) * self.height())
        corners = {
            'top_left': QPoint(x1, y1),
            'top_right': QPoint(x2, y1),
            'bottom_left': QPoint(x1, y2),
            'bottom_right': QPoint(x2, y2)
        }
        for corner, point in corners.items():
            if (point - pos).manhattanLength() < 4:  # Tolerance for detecting corner clicks
                return corner
        return False

    def get_corner(self, pos):
        for cls, xc, yc, w, h, color in self.label_list:
            x1 = int((xc - w / 2) * self.width())
            y1 = int((yc - h / 2) * self.height())
            x2 = int((xc + w / 2) * self.width())
            y2 = int((yc + h / 2) * self.height())

            corners = {
                'top_left': QPoint(x1, y1),
                'top_right': QPoint(x2, y1),
                'bottom_left': QPoint(x1, y2),
                'bottom_right': QPoint(x2, y2)
            }

            # 탐지 범위를 확대하여 마우스 위치를 비교
            for corner, point in corners.items():
                if (point - pos).manhattanLength() < 4:  # 허용 범위: 4픽셀
                    return corner
                
        return None

    def resize_box(self, pos):
        cls, xc, yc, w, h, color = self.label_list[self.selected_box]
        dx = pos.x() - self.drag_start_pos.x()
        dy = pos.y() - self.drag_start_pos.y()

        dx /= self.width()
        dy /= self.height()

        x1 = (xc - w / 2)
        y1 = (yc - h / 2)
        x2 = (xc + w / 2)
        y2 = (yc + h / 2)

        # Resize based on the corner being dragged
        if self.resize_corner == 'top_left':
            x1 += dx
            y1 += dy
        elif self.resize_corner == 'top_right':
            x2 += dx
            y1 += dy
        elif self.resize_corner == 'bottom_left':
            x1 += dx
            y2 += dy
        elif self.resize_corner == 'bottom_right':
            x2 += dx
            y2 += dy

        new_xc = (x2 + x1) / 2
        new_yc = (y2 + y1) / 2
        new_w = (x2 - x1) 
        new_h = (y2 - y1) 

        # 경계 조건 체크 수정: new_w와 new_h 사용
        if (new_xc - new_w / 2) < 0 or (new_xc + new_w / 2) > 1:
            new_xc = xc
            new_w = w

        if (new_yc - new_h / 2) < 0 or (new_yc + new_h / 2) > 1:
            new_yc = yc
            new_h = h

        # Update the bounding box size and position
        self.label_list[self.selected_box][1] = new_xc
        self.label_list[self.selected_box][2] = new_yc
        self.label_list[self.selected_box][3] = new_w
        self.label_list[self.selected_box][4] = new_h

        self.drag_start_pos = pos

    def create_new_box(self, end_pos):
        start_x = self.drag_start_pos.x()
        start_y = self.drag_start_pos.y()
        end_x = end_pos.x()
        end_y = end_pos.y()

        new_xc = (start_x + end_x) / 2 / self.width()
        new_yc = (start_y + end_y) / 2 / self.height()
        new_w = abs(end_x - start_x) / self.width()
        new_h = abs(end_y - start_y) / self.height()

        # 체크된 버튼의 인덱스를 찾습니다.
        cls_num = None
        for index, button in enumerate(self.cls_bnt_list):
            if button.isChecked():
                cls_num = index
                break

        # 만약 체크된 버튼이 없다면, 경고 메시지를 출력하고 종료
        if cls_num is None:
            print("아무 버튼도 체크되지 않았습니다. 클래스 번호를 지정하려면 하나의 버튼을 체크하세요.")
            return

        # 새 박스가 너무 작지 않은 경우에만 추가
        if new_w * new_h * self.width() * self.height() > 100:
            # color = self.parent.color(cls_num)
            color = self.parent.color[cls_num]


            self.label_list.append([cls_num, new_xc, new_yc, new_w, new_h, color])

            # 필요한 경우 버튼을 추가하는 로직이 있는지 확인
            # self.cls_bnt_list.append(self.make_cls_bnt(len(self.cls_bnt_list), 0, color))

            self.update()

class LabelingDialog(QDialog):
    def __init__(self):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} : open labeling window")

        super().__init__()
        self.label_ui = Ui_labeling_window()
        self.label_ui.setupUi(self)
        self.setFocusPolicy(Qt.StrongFocus)


        self.label_ui.cls_0.hide()

        # self.color = Colors()
        self.color = COLOR


        self.event_data_exist = False
        self.cnt = 0
        self.img_buffer = []
        self.label_buffer = []
        self.label_ui.label_list.clear()
        self.setFocusPolicy(Qt.StrongFocus)

        # 카메라 페이지 영상 뷰어
        self.label_ui.label_image_viewer.hide()
        self.label_ui.label_image_viewer = Labeling_Viewer(self)
        self.label_ui.label_image_viewer.setObjectName(u"camera_page_viewer")
        self.label_ui.label_image_viewer.setMinimumSize(QSize(343, 581))
        self.label_ui.label_image_viewer.setStyleSheet(u"border: 1px solid rgb(119, 118, 123);\n"
                                                         "background-color: rgba(255, 255, 255, 0);")
        self.label_ui.label_image_viewer.setScaledContents(False)
        self.label_ui.verticalLayout_3.addWidget(self.label_ui.label_image_viewer)

        self.label_ui.label_setting_bnt.clicked.connect(self.open_label_info)
        self.open_label_info(yaml_file_path = os.path.join("./", "cfg", "ms-ai-v1.3.yaml"))

        self.label_ui.label_del_bnt.clicked.connect(self.del_all_label)
        self.label_ui.img_dir_folder_select_bnt.clicked.connect(self.open_img_directory)
        self.label_ui.label_dir_folder_select_bnt.clicked.connect(self.open_label_directory)
        self.label_ui.SAM2_bnt.clicked.connect(self.run_SAM2)
        self.label_ui.YOLO_bnt.clicked.connect(self.run_YOLO)
        self.label_ui.label_data_del_bnt.clicked.connect(self.delete_label_data)


        self.label_ui.label_list.itemDoubleClicked.connect(self.load_data)

        self.label_ui.shutdown_bnt.clicked.connect(self.close_window)

    def delete_label_data(self):
        # 현재 선택된 항목 확인
        selected_item = self.label_ui.label_list.currentItem()
        
        if not selected_item:
            print("삭제할 이미지를 선택해주세요.")
            return
        
        # 선택된 이미지 파일명
        filename = selected_item.text()
        
        # 이미지 파일 경로
        img_file_path = os.path.join(self.img_directory, filename)
        
        # 라벨 파일 경로
        label_file_path = os.path.join(self.label_directory, filename[:-4] + ".txt")
        
        try:
            # 이미지 파일 삭제
            if os.path.exists(img_file_path):
                os.remove(img_file_path)
                print(f"이미지 파일 삭제됨: {filename}")
            
            # 라벨 파일 삭제
            if os.path.exists(label_file_path):
                os.remove(label_file_path)
                print(f"라벨 파일 삭제됨: {filename[:-4]}.txt")
            
            # 목록에서 항목 제거
            current_row = self.label_ui.label_list.currentRow()
            self.label_ui.label_list.takeItem(current_row)
            
            # 현재 뷰어 초기화
            self.label_ui.label_image_viewer.frame = None
            self.label_ui.label_image_viewer.display_image()
            self.label_ui.label_image_viewer.display_label([])
            
            print(f"'{filename}' 파일이 성공적으로 삭제되었습니다.")
            
        except Exception as e:
            print(f"파일 삭제 중 오류 발생: {e}")

    def run_YOLO(self):
        cls_num_select = None

        for index, button in enumerate(self.label_ui.label_image_viewer.cls_bnt_list):
                if button.isChecked():
                    cls_num_select = index
                    break

        if cls_num_select is None:
            print("아무 버튼도 체크되지 않았습니다. 클래스 번호를 지정하려면 하나의 버튼을 체크하세요.")

        # model_name = f"ms-ai_24-09-30-M.pt"
        
        # model_name = f"yolo11x.pt"
        model_name = "ms-ai_24-12-31-M.pt"
        # model_name = "test_swim_1.pt"


        weight_path = os.path.join(os.getcwd(), "..","weights", "yolo", model_name)
        yolo_model = YOLO(weight_path) 

        selected_indexes = [index.row() for index in self.label_ui.label_list.selectedIndexes()]

        item = self.label_ui.label_list.item(selected_indexes[0])
        img_name = item.text()
        img = cv2.imread(os.path.join(self.img_directory, img_name))


        pred = yolo_model(img, 
                            imgsz = 640, 
                            conf = self.label_ui.object_conf_value.value() / 100, 
                            iou = 0.5,
                            verbose=False,
                            classes = [cls_num_select]
                            
                            )

        boxes = pred[0].boxes.data.cpu().numpy().astype(float)

        label = []
        label_ori = self.label_ui.label_image_viewer.label_list

        updated_labels = []

        for i, boxes in enumerate(boxes):
                if len(boxes) != 0:
                    x1, y1, x2, y2 = boxes[0:4].astype('int') # float64 to int
                    # conf = data[4]
                    cls = boxes[-1].astype('int')

                    new_xc = (x1 + x2) / 2 / img.shape[1]
                    new_yc = (y1 + y2) / 2 / img.shape[0]
                    new_w = abs(x2 - x1) / img.shape[1]
                    new_h = abs(y2 - y1) / img.shape[0]
                    # ind = tracks[i, 7].astype('int') # float64 to int

                    # label.append([cls, new_xc, new_yc, new_w, new_h, self.color(cls)])
                    label.append([cls, new_xc, new_yc, new_w, new_h, self.color[cls]])


        for new_box in label:
            new_cls_num, new_x, new_y, new_w, new_h, color = new_box
            
            merged = False

            for ori_box in label_ori:
                ori_cls_num, ori_x, ori_y, ori_w, ori_h, color = ori_box

                if ori_cls_num == new_cls_num:
                    iou = get_iou_2([ori_x, ori_y, ori_w, ori_h], [new_x, new_y, new_w, new_h])

                    if iou >= self.label_ui.object_IOU_value.value() / 100:
                        # ori_box와 new_box의 좌표를 [x1, y1, x2, y2] 형식으로 변환
                        ori_x1 = ori_x - ori_w / 2
                        ori_y1 = ori_y - ori_h / 2
                        ori_x2 = ori_x + ori_w / 2
                        ori_y2 = ori_y + ori_h / 2

                        new_x1 = new_x - new_w / 2
                        new_y1 = new_y - new_h / 2
                        new_x2 = new_x + new_w / 2
                        new_y2 = new_y + new_h / 2

                        # 병합된 박스의 좌표 계산
                        merged_x1 = min(ori_x1, new_x1)
                        merged_y1 = min(ori_y1, new_y1)
                        merged_x2 = max(ori_x2, new_x2)
                        merged_y2 = max(ori_y2, new_y2)

                        # 병합된 좌표를 중심 기반 형식으로 변환하여 저장
                        merged_x_center = (merged_x1 + merged_x2) / 2
                        merged_y_center = (merged_y1 + merged_y2) / 2
                        merged_width = merged_x2 - merged_x1
                        merged_height = merged_y2 - merged_y1

                        # 병합된 좌표를 업데이트 리스트에 추가
                        # updated_labels.append([new_cls_num, merged_x_center, merged_y_center, merged_width, merged_height, self.color(new_cls_num)])
                        updated_labels.append([new_cls_num, merged_x_center, merged_y_center, merged_width, merged_height, self.color[new_cls_num]])

                            
                        # 기존 박스 제거
                        label_ori.remove(ori_box)
                        merged = True
                        break

            if not merged:
                # updated_labels.append([new_cls_num, new_x, new_y, new_w, new_h, self.color(new_cls_num)])
                updated_labels.append([new_cls_num, new_x, new_y, new_w, new_h, self.color[new_cls_num]])


        # 새로운 박스와 겹치지 않은 기존 박스를 updated_labels에 추가
        for remaining_ori_box in label_ori:
            updated_labels.append(remaining_ori_box)

        self.label_ui.label_image_viewer.display_label(updated_labels)
        print("done")

        del pred, yolo_model

        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()


    def run_SAM2(self):
        cls_num_select = None
        predictor = None

        for index, button in enumerate(self.label_ui.label_image_viewer.cls_bnt_list):
            if button.isChecked():
                cls_num_select = index
                break

        if cls_num_select is None:
            print("아무 버튼도 체크되지 않았습니다. 클래스 번호를 지정하려면 하나의 버튼을 체크하세요.")
            if self.label_ui.SAM2_bnt.isChecked():
                self.label_ui.SAM2_bnt.setChecked(False)
            return
        temp_path = os.path.join(self.img_directory, "temp")
        
        selected_indexes = [index.row() for index in self.label_ui.label_list.selectedIndexes()]

        with torch.no_grad():
            os.makedirs(temp_path, exist_ok=True)

            img_file_name_list = []
            for index in selected_indexes:
                item = self.label_ui.label_list.item(index)
                img_name = item.text()
                img_file_name_list.append(img_name)

                copy_img_name = f"{index:04d}_" + img_name.split("_")[-1]
                # copy_img_file_name_list.append(copy_img_name)

                source_path = os.path.join(self.img_directory, img_name)  # 원본 파일 경로
                destination_path = os.path.join(temp_path, copy_img_name)  # 복사할 파일의 대상 경로
                
                # 파일 복사
                shutil.copy(source_path, destination_path)
            batch_multiprocess_sam(img_file_name_list, self.img_directory, temp_path, self.label_directory, cls_num_select, self.label_ui.object_IOU_value.value())

        if self.label_ui.SAM2_bnt.isChecked():
            self.label_ui.SAM2_bnt.setChecked(False)
        print("Done")


    def close_window(self):
        self.close()

    def open_label_info(self, yaml_file_path = None):
        if yaml_file_path:
            file_path = yaml_file_path
        else:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(
                None,  # 부모 위젯은 None
                "Select YAML File",  # 창 제목
                "",  # 초기 디렉토리
                "YAML Files (*.yaml *.yml);;All Files (*)"  # 필터
            )

        # 파일이 선택되었는지 확인하고 경로 저장
        if file_path:
            self.yaml_file_path = file_path
            print(f"선택된 YAML 파일 경로: {self.yaml_file_path}")

            # YAML 파일 읽기
            with open(self.yaml_file_path, 'r') as file:
                data = yaml.safe_load(file)
            
            # names 항목이 있는지 확인
            if 'names' in data:
                # names 요소를 딕셔너리로 가져와 self.label_name에 저장
                self.label_name = {int(key): value for key, value in data['names'].items()}
                print(self.label_name)
                self.label_ui.label_image_viewer.class_name_dict = self.label_name
            else:
                print("YAML 파일에 'names' 항목이 존재하지 않습니다.")

        for num, cls_name in self.label_name.items():
            # self.label_ui.label_image_viewer.make_cls_bnt(num, num, self.color(num))
            self.label_ui.label_image_viewer.make_cls_bnt(num, num, self.color[num])



    def load_data(self):
        # print("데이터 불러오는 중")
        selected_item = self.label_ui.label_list.currentItem()

        if selected_item:  # 선택된 항목이 있는지 확인
            data_list = []

            # 선택된 항목의 텍스트 (파일명) 가져오기
            filename = selected_item.text()

            # 이미지 파일 경로 생성
            img = cv2.imread(os.path.join(self.img_directory, filename))

            
            if os.path.exists(os.path.join(self.label_directory, filename[:-4] + ".txt")):
                # 텍스트 파일 열기
                with open(os.path.join(self.label_directory, filename[:-4] + ".txt"), 'r') as file:
                    lines = file.readlines()

                    for line in lines:
                        # 공백으로 데이터를 분리
                        items = line.strip().split()
                        
                        # 문자열을 float 또는 int 타입으로 변환
                        cls = int(items[0])
                        ncx = float(items[1])
                        ncy = float(items[2])
                        nw = float(items[3])
                        nh = float(items[4])
                        
                        # 리스트에 데이터를 추가 (튜플 형태로 저장)
                        # data_list.append((cls, ncx, ncy, nw, nh, self.color(cls)))
                        data_list.append((cls, ncx, ncy, nw, nh, self.color[cls]))


            if img is not None:
                # 이미지를 성공적으로 읽었다면 원하는 처리를 추가
                self.label_ui.label_image_viewer.frame = img
                self.label_ui.label_image_viewer.display_image()
                self.label_ui.label_image_viewer.labeling_flag = True

            # if data_list:
            self.label_ui.label_image_viewer.display_label(data_list)

    def open_label_directory(self, init_path = None):
        if init_path:
            self.label_directory = init_path
        else:
            self.label_directory = QFileDialog.getExistingDirectory(None, "Select Directory")
        # self.label_directory = os.path.join("./dataset/coco/train/labels_new")

        if self.label_directory:
            # 선택한 디렉토리의 파일 목록 가져오기
            dir = QDir(self.label_directory)
            self.label_file_name_list = dir.entryList(QDir.Files)  # 파일만 목록에 추가

    def open_img_directory(self):
        self.img_directory = QFileDialog.getExistingDirectory(None, "Select Directory")
        # self.img_directory = os.path.join("./dataset/coco/train/images")


        if self.img_directory:
            self.label_ui.label_list.clear()

            # 선택한 디렉토리의 파일 목록 가져오기
            dir = QDir(self.img_directory)
            files = dir.entryList(QDir.Files)  # 파일만 목록에 추가
            # 이미지 파일 확장자 리스트
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            
            # 이미지 파일만 필터링
            files = [f for f in files if os.path.splitext(f.lower())[1] in image_extensions]
            
            # 숫자 기준으로 정렬 (1,2,3,10,20 순서로)
            files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x)

            # 파일 목록을 QListWidget에 추가
            for file in files:
                item = QListWidgetItem(file)
                # Center-align the text using a stylesheet
                item.setTextAlignment(Qt.AlignCenter)
                self.label_ui.label_list.addItem(item)

            self.open_label_directory(os.path.join(self.img_directory, "..", "labels"))

    def eventFilter(self, source, event):
        if event.type() == QEvent.KeyPress:
            self.keyPressEvent(event)
            return True
        return super(LabelingDialog, self).eventFilter(source, event)

    def del_all_label(self):
        self.label_ui.label_image_viewer.label_list = []
        self.label_ui.label_image_viewer.display_image()

    def keyPressEvent(self, event):
        # 다음 인덱스로 이동

        if event.key() == Qt.Key_D:
            current_index = self.label_ui.label_list.currentRow()
            next_index = current_index + 1
            if next_index >= self.label_ui.label_list.count():
                next_index = 0  # 리스트의 끝에 도달하면 처음으로 돌아갑니다.

            # 리스트에서 다음 항목 선택
            self.label_ui.label_list.setCurrentRow(next_index)

            # 선택된 이미지와 관련된 레이블 데이터를 보여줌
            self.load_data()

        # 이전 이미지로 이동 (Key_A)
        elif event.key() == Qt.Key_A:
            current_index = self.label_ui.label_list.currentRow()
            prev_index = current_index - 1
            if prev_index < 0:
                prev_index = self.label_ui.label_list.count() - 1  # 리스트의 처음에 도달하면 마지막으로 돌아갑니다.

            # 리스트에서 이전 항목 선택
            self.label_ui.label_list.setCurrentRow(prev_index)

            # 선택된 이미지와 관련된 레이블 데이터를 보여줌
            self.load_data()


        elif event.key() == Qt.Key_W:
            self.label_ui.label_image_viewer.box_resize_mode = True

        elif event.key() == Qt.Key_F:
            self.delete_label_data()
            # self.label_ui.label_image_viewer.display_label([])
            pass            

        elif event.key() == Qt.Key_E:
            self.del_all_label()
            # self.label_ui.label_image_viewer.display_label([])
            pass            


        # elif (event.key() == Qt.Key_S and event.modifiers() == Qt.ControlModifier) or event.key() == Qt.Key_Space:
        elif (event.key() == Qt.Key_S and event.modifiers() == Qt.ControlModifier) or event.key() == Qt.Key_S:

            self.save_label_buffer()
            current_index = self.label_ui.label_list.currentRow()
            next_index = current_index + 1
            if next_index >= self.label_ui.label_list.count():
                next_index = 0  # 리스트의 끝에 도달하면 처음으로 돌아갑니다.

            # 리스트에서 다음 항목 선택
            self.label_ui.label_list.setCurrentRow(next_index)

            # 선택된 이미지와 관련된 레이블 데이터를 보여줌
            self.load_data()


        elif event.key() == Qt.Key_Q:
            self.run_YOLO()
            pass

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_W:
            self.label_ui.label_image_viewer.box_resize_mode = False

    def close_window(self):
        self.close()

    def del_label_data(self):
        camera_name = self.label_ui.camera_name_box.currentText()
        date = self.label_ui.event_date_box.currentText()
        selected_indexes = self.label_ui.label_list.selectedIndexes()

        event_name_list = []

    def save_label_buffer(self):
        selected_item = self.label_ui.label_list.currentItem()
        if selected_item: 
            filename = selected_item.text()
            save_file_path = os.path.join(self.label_directory, filename[:-4] + ".txt")
            with open(save_file_path, 'w') as file:
                line = ""
                for cls, ncx, ncy, nw, nh, color in self.label_ui.label_image_viewer.label_list:
                    # 레이블 정보를 텍스트 파일로 작성합니다.
                    line += f"{cls} {ncx:.3f} {ncy:.3f} {nw:.3f} {nh:.3f}\n"
                file.write(line)

            print(f"레이블 정보가 '{save_file_path}'에 저장되었습니다.")

def get_bboxes_from_binary_img(mask):
    mask_uint8 = np.squeeze(mask).astype(np.uint8)  # 데이터 타입 변환
    # cv2.imshow("test",mask_uint8*255)
    # cv2.waitKey(0)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return [cv2.boundingRect(contour) for contour in contours]


def get_iou(box1, box2):
    """
    두 박스 간의 IOU를 계산합니다.
    box1, box2: [x1, y1, x2, y2] 형태
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 교집합 영역 계산
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # 합집합 영역 계산
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def get_iou_2(box1, box2):
    # box format: [ncx, ncy, nw, nh]
    # Convert [ncx, ncy, nw, nh] to [x1, y1, x2, y2]
    
    # Box1 변환
    box1_x1 = box1[0] - box1[2] / 2  # x1 = ncx - nw / 2
    box1_y1 = box1[1] - box1[3] / 2  # y1 = ncy - nh / 2
    box1_x2 = box1[0] + box1[2] / 2  # x2 = ncx + nw / 2
    box1_y2 = box1[1] + box1[3] / 2  # y2 = ncy + nh / 2

    # Box2 변환
    box2_x1 = box2[0] - box2[2] / 2  # x1 = ncx - nw / 2
    box2_y1 = box2[1] - box2[3] / 2  # y1 = ncy - nh / 2
    box2_x2 = box2[0] + box2[2] / 2  # x2 = ncx + nw / 2
    box2_y2 = box2[1] + box2[3] / 2  # y2 = ncy + nh / 2

    # Intersection 계산
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Area 계산
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union = box1_area + box2_area - intersection

    return intersection / union if union != 0 else 0

def batch_multiprocess_sam(img_file_name_list, img_directory, temp_path, label_directory, cls_num_select, object_IOU_value):
    try:
        # 이미지 로드 및 리스트에 저장
        img = cv2.imread(os.path.join(img_directory, img_file_name_list[0]))

        torch_autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        torch_autocast_ctx.__enter__()
        sam2_checkpoint = os.path.join(os.getcwd(), "..", "weights", "segment_anything_2", "sam2.1_hiera_base_plus.pt")
        # model_cfg = "./sam2.1_hiera_l.yaml"
        # model_cfg = os.path.join(os.getcwd(), "weights", "segment_anything_2","sam2.1_hiera_l.yaml")
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        inference_state = predictor.init_state(video_path=os.path.join(temp_path),
                                                offload_video_to_cpu = True,
                                            offload_state_to_cpu = True)


        predictor.reset_state(inference_state)

        frist_label_list = []

        if os.path.exists(os.path.join(label_directory, img_file_name_list[0][:-4] + ".txt")):
            # 텍스트 파일 열기
            with open(os.path.join(label_directory, img_file_name_list[0][:-4] + ".txt"), 'r') as file:
                lines = file.readlines()

                for line in lines:
                    # 공백으로 데이터를 분리
                    items = line.strip().split()
                    
                    # 문자열을 float 또는 int 타입으로 변환
                    cls = int(items[0])
                    ncx = float(items[1])
                    ncy = float(items[2])
                    nw = float(items[3])
                    nh = float(items[4])
                    
                    # 리스트에 데이터를 추가 (튜플 형태로 저장)
                    frist_label_list.append((cls, ncx, ncy, nw, nh))

        for i , (cls, ncx, ncy, nw, nh) in enumerate(frist_label_list):
            if cls_num_select == cls:
                x1 = int((ncx - nw / 2) * img.shape[1])
                y1 = int((ncy - nh / 2) * img.shape[0])
                x2 = int((ncx + nw / 2) * img.shape[1])
                y2 = int((ncy + nh / 2) * img.shape[0])

                ann_frame_idx = 0  # the frame index we interact with
                ann_obj_id = int(i)  # give a unique id to each object we interact with (it can be any integers)
                box = np.array([int(x1), int(y1), int(x2), int(y2)])

                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                                                        inference_state=inference_state,
                                                        frame_idx=ann_frame_idx,
                                                        obj_id=ann_obj_id,
                                                        box=box,
                                                        )

        video_segments = {} 
        label_dict = {}

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.).cpu().numpy()
                                                for i, out_obj_id in enumerate(out_obj_ids)
                                                }
                
        for out_frame_idx in video_segments:
            # item_text = self.label_ui.label_list.item(out_frame_idx).text()
            item_text = img_file_name_list[out_frame_idx]
            
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                bboxes = get_bboxes_from_binary_img(out_mask)

                if item_text not in label_dict:
                    label_dict[item_text] = []

                # if len(bboxes) > 0:
                #     for x, y, w, h in bboxes:
                #         if w * h >= 250:  # 너무 작은 bbox는 제거
                #             ncx = float((x + w / 2) / img.shape[1])
                #             ncy = float((y + h / 2) / img.shape[0])
                #             nw = w / img.shape[1]
                #             nh = h / img.shape[0]

                #             label_dict[item_text].append([cls_num_select, ncx, ncy, nw, nh])

                if len(bboxes) > 0:
                    for box in bboxes:
                        x, y, w, h = box
                        if w * h >= 250:  # 너무 작은 bbox는 제거

                            min_x = min(box[0] for box in bboxes)
                            min_y = min(box[1] for box in bboxes)
                            max_x = max(box[0] + box[2] for box in bboxes)  # x + width
                            max_y = max(box[1] + box[3] for box in bboxes)  # y + height

                            ncx = float(((min_x + max_x) / 2) / img.shape[1])
                            ncy = float(((min_y + max_y) / 2) / img.shape[0])
                            nw = (max_x - min_x) / img.shape[1]
                            nh = (max_y - min_y) / img.shape[0]


                            label_dict[item_text].append([cls_num_select, ncx, ncy, nw, nh])
        
        for image_name, bboxes in label_dict.items():
            label_file_path = os.path.join(label_directory, f"{image_name[:-4]}.txt")
            labels_ori = []
            updated_labels = []

            # 2. Check if label file exists, and read existing label data if present
            if os.path.exists(label_file_path):
                with open(label_file_path, 'r') as file:
                    for line in file:
                        parts = line.strip().split()
                        cls_num_ori = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # Convert YOLO format to [x1, y1, x2, y2]
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2
                    
                        if cls_num_ori == cls_num_select:
                            labels_ori.append([cls_num_ori, x1, y1, x2, y2])

                        else: updated_labels.append([cls_num_ori, x1, y1, x2, y2])
                            

            # 3. 새로운 박스를 기존 박스와 비교하여 IOU 이상일 경우 병합
            for new_box in bboxes:
                new_cls_num, new_x, new_y, new_w, new_h = new_box
                new_x1 = new_x - new_w / 2
                new_y1 = new_y - new_h / 2
                new_x2 = new_x + new_w / 2
                new_y2 = new_y + new_h / 2

                merged = False

                for ori_box in labels_ori:
                    ori_cls_num, ori_x1, ori_y1, ori_x2, ori_y2 = ori_box
                    if new_cls_num == ori_cls_num:
                        iou = get_iou([ori_x1, ori_y1, ori_x2, ori_y2], [new_x1, new_y1, new_x2, new_y2])
                        print([ori_x1, ori_y1, ori_x2, ori_y2], [new_x1, new_y1, new_x2, new_y2], iou)

                        if iou >= object_IOU_value / 100:
                            # 병합된 박스의 좌표 계산
                            merged_x1 = min(ori_x1, new_x1)
                            merged_y1 = min(ori_y1, new_y1)
                            merged_x2 = max(ori_x2, new_x2)
                            merged_y2 = max(ori_y2, new_y2)
                            updated_labels.append([new_cls_num, merged_x1, merged_y1, merged_x2, merged_y2])
                            
                            # 기존 박스 제거
                            labels_ori.remove(ori_box)
                            merged = True
                            break

                if not merged:
                    updated_labels.append([new_cls_num, new_x1, new_y1, new_x2, new_y2])

            # 새로운 박스와 겹치지 않은 기존 박스를 updated_labels에 추가
            for remaining_ori_box in labels_ori:
                updated_labels.append(remaining_ori_box)

            #updated_labels 모든 박스에 대해서 IOU 계산후 겹친 박스 제거
            iou_threshold = 0.9
            final_labels = []
            used = [False] * len(updated_labels)

            for i in range(len(updated_labels)):
                if used[i]:
                    continue
                cls_i, x1_i, y1_i, x2_i, y2_i = updated_labels[i]
                merged = False
                for j in range(i + 1, len(updated_labels)):
                    if used[j]:
                        continue
                    cls_j, x1_j, y1_j, x2_j, y2_j = updated_labels[j]
                    if cls_i != cls_j:
                        continue
                    iou = get_iou([x1_i, y1_i, x2_i, y2_i], [x1_j, y1_j, x2_j, y2_j])
                    if iou >= iou_threshold:
                        # 병합
                        x1_new = min(x1_i, x1_j)
                        y1_new = min(y1_i, y1_j)
                        x2_new = max(x2_i, x2_j)
                        y2_new = max(y2_i, y2_j)
                        final_labels.append([cls_i, x1_new, y1_new, x2_new, y2_new])
                        used[j] = True
                        merged = True
                        break
                if not merged:
                    final_labels.append([cls_i, x1_i, y1_i, x2_i, y2_i])
                used[i] = True

            print(label_file_path)

            # 최종 박스만 저장
            with open(label_file_path, 'w') as file:
                for label in final_labels:
                    cls_num, x1, y1, x2, y2 = label
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    file.write(f"{cls_num} {x_center:.3} {y_center:.3} {width:.3} {height:.3}\n")

            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)
                
        predictor.reset_state(inference_state)

        del predictor
        del inference_state
        # 기타 대용량 객체 삭제
        del video_segments, label_dict
        # autocast context 닫기
        torch_autocast_ctx.__exit__(None, None, None)
        # gc 및 cuda 메모리 반환
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()

    except Exception as e:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tb = traceback.format_exc()
        print(f"Error occurred at {current_time}: {e}\n{tb}", file=sys.stderr)

        del predictor
        del inference_state
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()


    print("Done")

if __name__ == "__main__":
    app = QApplication(sys.argv)  # Create the QApplication instance
    labeling_window = LabelingDialog()
    labeling_window.show()
    sys.exit(app.exec())  # Start the event loop