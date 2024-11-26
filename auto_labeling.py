from ultralytics import YOLO
import os
from tqdm import tqdm
import cv2
import numpy as np

# model = YOLO("./weights/yolo/yolo11x.pt")
model = YOLO("./weights/yolo/ms-ai_v1.3_24-11-19-M.pt")


dataset_path = os.path.join("./", "dataset", "명지_산학_3")
output_dir = os.path.join(dataset_path, "labels_new")

# 결과를 저장할 디렉토리가 없으면 생성
os.makedirs(output_dir, exist_ok=True)

coco_class_num_dict = {0 : 0, #person
                  1 : 1, #bicycle
                  2 : 2, #car
                  3 : 3, #motorcycle
                  5 : 4, #bus
                  7 : 5, #truck
                  14 : 6, #bire
                  15 : 7, #cat
                  16 : 8  #dog
}



img_name_list = os.listdir(os.path.join(dataset_path, "images"))

for img_name in tqdm(img_name_list):
    img = cv2.imread(os.path.join(dataset_path, "images", img_name))
    img_height, img_width = img.shape[:2]


    # pred = model(img, classes=[0,1,2,3,5,6,14,15,16])
    # pred = model(img, classes=[0], verbose = False)
    # pred = model(img, classes=[1,2,3,5,7,14,15,16], verbose = False)
    pred = model(img, classes=[1,2,3,4,5], verbose = False)

    


    # 결과 파일 저장 경로 설정
    txt_output_path = os.path.join(output_dir, img_name.replace(".jpg", ".txt"))


    label_path = os.path.join(dataset_path, "labels", img_name.replace(".jpg", ".txt"))

    # 기존 라벨 파일 내용 읽기
    existing_labels = []
    if os.path.exists(label_path):
        with open(label_path, "r") as lf:
            existing_labels = lf.readlines()  # 기존 라벨 데이터 저장

    # 예측 결과를 텍스트 파일로 저장
    with open(txt_output_path, "w") as f:
        # 기존 라벨 내용 먼저 쓰기
        for line in existing_labels:
                try:
                    # if int(line[0]) not in [2, 5, 1, 3, 6, 7, 8]: 
                        # f.write(line.strip() + "\n")
                    if int(line[0]) in [0]: 
                        f.write(line.strip() + "\n")

                except:
                    pass
            # f.write(line.strip() + "\n")
        
        for result in pred:
            boxes = result.boxes  # 예측된 바운딩 박스 정보
            
            for box in boxes:
                cls = int(box.cls)  # 클래스 인덱스
                conf = box.conf      # 신뢰도
                x, y, w, h = box.xywh.cpu().numpy()[0]  # 바운딩 박스 좌표 (x, y, w, h)

                # 텍스트 파일에 저장 (YOLO 형식: class x_center y_center width height)
                ncx = np.round(x / img_width, 3)     # 정규화된 중심 x 좌표
                ncy = np.round(y / img_height, 3)    # 정규화된 중심 y 좌표
                nw = np.round(w / img_width, 3)      # 정규화된 너비
                nh = np.round(h / img_height,3 )     # 정규화된 높이

                # 텍스트 파일에 저장 (COCO 형식: class ncx ncy nw nh)
                # f.write(f"{class_num_dict[cls]} {ncx} {ncy} {nw} {nh}\n")
                f.write(f"{cls} {ncx} {ncy} {nw} {nh}\n")


os.system("chmod 777 -R ./")