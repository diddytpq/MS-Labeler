import cv2
cv2.namedWindow("main", cv2.WINDOW_AUTOSIZE) 

import numpy as np
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] # current directory

# from yolo_tracking.boxmot.tracker_zoo import create_tracker
# from yolo_tracking.boxmot.utils import ROOT, WEIGHTS
from ultralytics import YOLO

import time
import datetime
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

def save_image_text(img, save_path, img_name,frame_num, text): # 시험 모듈 (지연이랑, 박스 카운트)
    img_path = save_path + img_name.format(frame_num)

# model = YOLO('../weight/yolo/ms-ai_24-12-31-M.pt', task="detect")  # load a pretrained model (recommended for training)\
model = YOLO('../weights/yolo/test_swim_1.pt', task="detect")  # load a pretrained model (recommended for training)\

source = "../../videos/수영장_실험_영상/test2.mp4"

vid = cv2.VideoCapture(source)

total_frame = vid.get(cv2.CAP_PROP_FRAME_COUNT)
# vid.set(cv2.CAP_PROP_POS_FRAMES, total_frame / 3)

color = (0, 0, 255)  # BGR
thickness = 2
fontscale = 1
frame_num = 0
save_txt = ""
colors = Colors()

output_file = f'output_test.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# 비디오 프레임의 너비와 높이 가져오기
frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 비디오 작성자 객체 생성 및 출력 파일 설정
out = cv2.VideoWriter(output_file, fourcc, 60.0, (frame_width, frame_height))
ret, im = vid.read()

while ret:
    t0 = time.time()

    frame_num += 1
    ret, im = vid.read()
    # im = cv2.imread("./1717477280.png")
    
    # if video_streamer.frame_available():
    #     im = video_streamer.get_frame()
    # else:
    #     im = np.zeros((720, 1280, 3), np.uint8)
    #     print(f"video is empty")

    # im_test = [im,im,im,im]
    dets = model.predict(source=im, classes = [0], imgsz = 640, conf = 0.15, iou = 0.5, half = True, verbose=False)
    
    # dets = model.predict(source=im, imgsz = 640, conf = 0.30, iou = 0.5, half = True, verbose=False)
    # dets = model.predict(source=im, imgsz = 1280, conf = 0.30, iou = 0.5, half = True, verbose=False)

    boxes = dets[0].boxes.data.cpu().numpy().astype(float)

    for (x1, y1, x2, y2, conf, cls) in boxes:
        xyxy = np.array([x1, y1, x2, y2],dtype="int") # float64 to int
        conf = conf
        cls = cls.astype('int')

        label = f'{model.names[cls]}'

        bbox_color = colors(int(cls))

        plot_one_box(xyxy, im, label=label, color=bbox_color, line_thickness=2)

    # tracks = tracker.update(boxes, im) # --> (x, y, x, y, id, conf, cls, ind)
    # if tracks.shape[0] != 0:
    #     xyxys = tracks[:, 0:4].astype('int') # float64 to int
    #     ids = tracks[:, 4].astype('int') # float64 to int
    #     confs = tracks[:, 5]
    #     clss = tracks[:, 6].astype('int') # float64 to int
    #     inds = tracks[:, 7].astype('int') # float64 to int

    # # print bboxes with their associated id, cls and conf
        # for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
        #     label = None if False else f"{id} {dets[0].names[cls]} {conf:.2f}"
        #     plot_one_box(xyxy, im, label=label, color=colors(cls), line_thickness=3) # 박스 그리기
    # result_img = dets[0].plot()
    # result_img = im

    # cv2.imshow("frame", cv2.resize(result_img, (0, 0), fx = 0.5, fy = 0.5))
    cv2.imshow("main", im)
    out.write(im)

    # if len(boxes):
    #     now = datetime.datetime.fromtimestamp(time.time()/1000.0)
    #     for i in range(len(boxes)):
    #         save_txt += f"{frame_num}, person, {np.round(boxes[i][-2],2)}, {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
    #     cv2.imwrite(f"./images/{frame_num}.jpg", result_img)
        
    #     print(save_txt)

    print(f"FPS: {1/(time.time() - t0)}")

    # break on pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# with open(f"./images/{frame_num}.txt", "w") as f:
#     f.write(save_txt)

cv2.destroyAllWindows()
vid.release()
out.release()