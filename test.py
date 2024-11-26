# import sys
# from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QListWidget, QPushButton, QFileDialog
# from PySide6.QtCore import QDir

# class FileListApp(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("File List Viewer")
        
#         # 레이아웃 설정
#         layout = QVBoxLayout()

#         # 파일 목록을 표시할 QListWidget
#         self.list_widget = QListWidget()
#         layout.addWidget(self.list_widget)

#         # 디렉토리를 선택할 버튼
#         self.open_button = QPushButton("Select Directory")
#         self.open_button.clicked.connect(self.open_directory)
#         layout.addWidget(self.open_button)

#         self.setLayout(layout)

#     def open_directory(self):
#         # 디렉토리 선택 대화상자 열기
#         directory = QFileDialog.getExistingDirectory(self, "Select Directory")
#         if directory:
#             self.list_files(directory)

#     def list_files(self, directory):
#         # 파일 목록을 초기화
#         self.list_widget.clear()

#         # 선택한 디렉토리의 파일 목록 가져오기
#         dir = QDir(directory)
#         files = dir.entryList(QDir.Files)  # 파일만 목록에 추가

#         # 파일 목록을 QListWidget에 추가
#         for file in files:
#             self.list_widget.addItem(file)


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = FileListApp()
#     window.resize(400, 300)
#     window.show()
#     sys.exit(app.exec())


# import os

# def modify_cls_in_folder(input_folder, output_folder):
#     # 출력 폴더가 없으면 생성
#     os.makedirs(output_folder, exist_ok=True)
    
#     # 입력 폴더에서 모든 텍스트 파일 순차적으로 읽기
#     for filename in os.listdir(input_folder):
#         if filename.endswith(".txt"):
#             file_path = os.path.join(input_folder, filename)
#             output_file_path = os.path.join(output_folder, filename)
            
#             with open(file_path, 'r') as file:
#                 lines = file.readlines()
            
#             # 각 줄을 읽고 cls가 1인 경우 9로 수정
#             modified_lines = []
#             for line in lines:
#                 parts = line.strip().split()
#                 cls = int(parts[0])  # cls 값을 가져옴
#                 if cls in [0, 1, 2, 3, 4, 5, 9]:
#                     if cls == 9:
#                         parts[0] = '6'  # cls가 1이면 9로 변경

#                 modified_lines.append(" ".join(parts))  # 수정된 줄을 리스트에 추가
            
#             # 수정된 내용을 새로운 파일에 저장
#             with open(output_file_path, 'w') as file:
#                 file.write("\n".join(modified_lines))

#             print(f"Modified file saved to: {output_file_path}")

# # 사용 예시
# input_folder = "./dataset/test/train/labels"       # 텍스트 파일들이 있는 폴더 경로
# output_folder = "./dataset/test/train/labels_new"     # 수정된 파일을 저장할 폴더 경로
# modify_cls_in_folder(input_folder, output_folder)


# import os

# def delete_every_third_file(folder_path):
#     # 폴더에서 파일 목록을 가져옴
#     files = sorted(os.listdir(folder_path))  # 정렬된 파일 목록

#     # 파일을 3개마다 하나씩 제거
#     for i, file_name in enumerate(files):
#         if (i + 1) % 3 == 0:  # 3번째마다
#             file_path = os.path.join(folder_path, file_name)
#             os.remove(file_path)  # 파일 제거
#             print(f"Deleted file: {file_path}")

# # 사용 예시
# folder_path = "./dataset/명지_산학_3/images"  # 폴더 경로
# delete_every_third_file(folder_path)

# # 사용 예시
# folder_path = "./dataset/명지_산학_3/labels"  # 폴더 경로
# delete_every_third_file(folder_path)

# import sys
# from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QListWidget, QPushButton, QFileDialog
# from PySide6.QtCore import QDir

# class FileListApp(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("File List Viewer")
        
#         # 레이아웃 설정
#         layout = QVBoxLayout()

#         # 파일 목록을 표시할 QListWidget
#         self.list_widget = QListWidget()
#         layout.addWidget(self.list_widget)

#         # 디렉토리를 선택할 버튼
#         self.open_button = QPushButton("Select Directory")
#         self.open_button.clicked.connect(self.open_directory)
#         layout.addWidget(self.open_button)

#         self.setLayout(layout)

#     def open_directory(self):
#         # 디렉토리 선택 대화상자 열기
#         directory = QFileDialog.getExistingDirectory(self, "Select Directory")
#         if directory:
#             self.list_files(directory)

#     def list_files(self, directory):
#         # 파일 목록을 초기화
#         self.list_widget.clear()

#         # 선택한 디렉토리의 파일 목록 가져오기
#         dir = QDir(directory)
#         files = dir.entryList(QDir.Files)  # 파일만 목록에 추가

#         # 파일 목록을 QListWidget에 추가
#         for file in files:
#             self.list_widget.addItem(file)


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = FileListApp()
#     window.resize(400, 300)
#     window.show()
#     sys.exit(app.exec())


# import os

# def modify_cls_in_folder(input_folder, output_folder):
#     # 출력 폴더가 없으면 생성
#     os.makedirs(output_folder, exist_ok=True)
    
#     # 입력 폴더에서 모든 텍스트 파일 순차적으로 읽기
#     for filename in os.listdir(input_folder):
#         if filename.endswith(".txt"):
#             file_path = os.path.join(input_folder, filename)
#             output_file_path = os.path.join(output_folder, filename)
            
#             with open(file_path, 'r') as file:
#                 lines = file.readlines()
            
#             # 각 줄을 읽고 cls가 1인 경우 9로 수정
#             modified_lines = []
#             for line in lines:
#                 parts = line.strip().split()
#                 cls = int(parts[0])  # cls 값을 가져옴
#                 if cls in [0, 1, 2, 3, 4, 5, 9]:
#                     if cls == 9:
#                         parts[0] = '6'  # cls가 1이면 9로 변경

#                 modified_lines.append(" ".join(parts))  # 수정된 줄을 리스트에 추가
            
#             # 수정된 내용을 새로운 파일에 저장
#             with open(output_file_path, 'w') as file:
#                 file.write("\n".join(modified_lines))

#             print(f"Modified file saved to: {output_file_path}")

# # 사용 예시
# input_folder = "./dataset/test/train/labels"       # 텍스트 파일들이 있는 폴더 경로
# output_folder = "./dataset/test/train/labels_new"     # 수정된 파일을 저장할 폴더 경로
# modify_cls_in_folder(input_folder, output_folder)


# import os

# def delete_every_third_file(folder_path):
#     # 폴더에서 파일 목록을 가져옴
#     files = sorted(os.listdir(folder_path))  # 정렬된 파일 목록

#     # 파일을 3개마다 하나씩 제거
#     for i, file_name in enumerate(files):
#         if (i + 1) % 3 == 0:  # 3번째마다
#             file_path = os.path.join(folder_path, file_name)
#             os.remove(file_path)  # 파일 제거
#             print(f"Deleted file: {file_path}")

# # 사용 예시
# folder_path = "./dataset/명지_산학_3/images"  # 폴더 경로
# delete_every_third_file(folder_path)

# # 사용 예시
# folder_path = "./dataset/명지_산학_3/labels"  # 폴더 경로
# delete_every_third_file(folder_path)

import threading
import os
import cv2
import sys
import numpy as np
import time
from ultralytics import YOLO
import torch

class VideoStreamBuffer:
    def __init__(self, rtsp_url):
        self.capture = cv2.VideoCapture(rtsp_url)
        self.buffer_frame = None
        self.lock = threading.Lock()
        self.stopped = False

        # Start the frame update thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.ret = False

    def update(self):
        while not self.stopped:
            if self.capture.isOpened():
                self.ret, frame = self.capture.read()
                if self.ret:
                    with self.lock:
                        self.buffer_frame = frame

    def read(self):
        with self.lock:
            frame = self.buffer_frame
        return self.ret, frame

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.capture.release()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(os.getcwd(), "weights", "yolo", "ms-ai_v1.3_24-11-19-M.pt")
    rtsp_url = 'rtsp://admin:admin13579@117.17.159.195:554/stream1'
    video_stream = cv2.VideoCapture(rtsp_url)
    model = YOLO(model_path, task="detect").to(device=device)  # load a pretrained model (recommended for training)\
    print("load model")
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = video_stream.read()
        
        if ret and frame is not None:
            img = cv2.resize(frame,(640,480))
            # dets = model.predict(source=frame, imgsz = 640, conf = 0.50, iou = 0.5, classes = [0], half = True, verbose=False)
            dets = model(source=img, imgsz = 640, conf = 0.50, iou = 0.5, classes = [0,1,2,3,4,5],half = False, verbose=False)
            torch.cuda.synchronize()
            # 검출된 바운딩 박스 정보 가져오기
            for det in dets[0].boxes:  # dets[0]에 각 프레임의 검출 결과가 포함됨
                # 바운딩 박스 좌표 및 클래스 정보 추출
                x1, y1, x2, y2 = map(int, det.xyxy[0])  # 좌표를 정수로 변환
                confidence = det.conf[0]  # 신뢰도
                class_id = int(det.cls[0])  # 클래스 ID

                # Bounding Box 및 라벨 그리기
                label = f"{class_id}: {confidence:.2f}"  # 클래스 ID 및 신뢰도 표시
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # BBox 그리기
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 프레임을 화면에 출력
            cv2.imshow("test", img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    # 자원 해제
    video_stream.release()
    cv2.destroyAllWindows()


# from ultralytics import YOLO
# from pathlib import Path
# from datetime import datetime
# import os

# def remove_npy() -> None:
#     import glob
#     npy_files = glob.glob(os.path.join(os.getcwd(), "dataset", '**', '*.npy'), recursive=True)
#     cache_files = glob.glob(os.path.join(os.getcwd(), "dataset", '**', '*.cache'), recursive=True)

#     for file_path in npy_files:
#         try:
#             os.remove(file_path)
#             # print(f"파일 '{file_path}'이(가) 삭제되었습니다.")
#         except Exception as e:
#             print(f"파일 '{file_path}'을(를) 삭제하는 중 오류가 발생했습니다: {e}")

#     for file_path in cache_files:
#         try:
#             os.remove(file_path)
#             # print(f"파일 '{file_path}'이(가) 삭제되었습니다.")
#         except Exception as e:
#             print(f"파일 '{file_path}'을(를) 삭제하는 중 오류가 발생했습니다: {e}")


# train_txt = ""
# val_txt = ""
# dataset_path = os.path.join(os.getcwd(), "dataset")
# folder_list = os.listdir(dataset_path)

# train_folder_list = ["명지_산학_1", "명지_산학_2", "명지_산학_3", "명지_입회시험", "coco", "fire", "KISA_국내", "KISA_해외", "미르스타디움"]

# for folder_name in folder_list:


#     if folder_name not in ["train.txt", "val.txt"] and folder_name in train_folder_list:
#         train_img_path = os.path.join(dataset_path, folder_name, "train", "images")
#         val_img_path = os.path.join(dataset_path, folder_name, "val", "images")

#         train_img_list = os.listdir(train_img_path)

#         try:
#             val_img_list = os.listdir(val_img_path)
#         except:
#             val_img_list = []

#         if train_img_list:
#             train_img_list = sorted(train_img_list)
#             for img_name in train_img_list:
#                 save_path = os.path.join("./", folder_name, "train", "images")
#                 img_path = os.path.join(save_path, img_name)
#                 train_txt += f"{img_path}\n"

#         if val_img_list:
#             val_img_list = sorted(val_img_list)
#             for img_name in val_img_list:
#                 save_path = os.path.join("./", folder_name, "val", "images")
#                 img_path = os.path.join(save_path, img_name)
#                 val_txt += f"{img_path}\n"

# if len(train_txt) > 0:
#     with open("./dataset/train.txt", "w") as f:
#         f.write(train_txt)

# if len(val_txt) > 0:
#     with open("./dataset/val.txt", "w") as f:
#         f.write(val_txt)
