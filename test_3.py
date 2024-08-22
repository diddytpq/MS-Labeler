import boxmot
from ultralytics import YOLO
from ultralytics.trackers.bot_sort import BOTSORT



import cv2
import numpy as np
from pathlib import Path
import sys
import os
import random


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

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# model = YOLO('./weights/yolo/ms-ai2405-finetune_M.pt') 
# model = YOLO('./weights/yolo/ms-ai2401.pt') 
# model = YOLO('./weights/yolo/yolov8m.pt') 

model = YOLO('./weights/yolo/ms-ai_24-08-12-M.pt') 



camera_path = os.path.join(os.getcwd(), "videos", "test_video2")
camera_list = os.listdir(camera_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

for camera_name in camera_list:
    video_path = os.path.join(os.getcwd(), "videos", "test_video2", camera_name)
    video_list = os.listdir(video_path)

    for video_name in video_list:
        vid = cv2.VideoCapture(os.path.join(video_path, video_name))
        # vid = cv2.VideoCapture(os.path.join(os.getcwd(), "videos", "test_video", "미르스타디움_6.mp4"))


        frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_file = f"./results/{camera_name}_{video_name}.mp4"
        # output_file = f"./results/test.mp4"


        out = cv2.VideoWriter(output_file, fourcc, 30.0, (frame_width, frame_height))

        while True:
            ret, im = vid.read()
            if ret == False : 
                # exit()
                break
            dets = model.predict(source=im, imgsz = (640, 640), conf = 0.3, iou = 0.5, classes = [0], half = False, verbose=False, agnostic_nms = True)
            result_img = dets[0].plot(line_width = 2, font_size = 2)
            cv2.imshow("frame", result_img)
            out.write(result_img)

            if cv2.waitKey(1) & 0xFF == 27:
                break

cv2.destroyAllWindows()
vid.release()