
from tqdm import tqdm
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import os

def remove_npy() -> None:
    import glob
    npy_files = glob.glob(os.path.join(os.getcwd(), "dataset", '**', '*.npy'), recursive=True)
    cache_files = glob.glob(os.path.join(os.getcwd(), "dataset", '**', '*.cache'), recursive=True)

    for file_path in npy_files:
        try:
            os.remove(file_path)
            # print(f"파일 '{file_path}'이(가) 삭제되었습니다.")
        except Exception as e:
            print(f"파일 '{file_path}'을(를) 삭제하는 중 오류가 발생했습니다: {e}")

    for file_path in cache_files:
        try:
            os.remove(file_path)
            # print(f"파일 '{file_path}'이(가) 삭제되었습니다.")
        except Exception as e:
            print(f"파일 '{file_path}'을(를) 삭제하는 중 오류가 발생했습니다: {e}")


train_txt = ""
val_txt = ""
dataset_path = os.path.join(os.getcwd(), "dataset")
folder_list = os.listdir(dataset_path)

# train_folder_list = ["명지_산학_1", "명지_산학_2", "명지_산학_3", "명지_입회시험", "coco", "fire", "KISA_국내", "KISA_해외", "미르스타디움", "수영장_해외_1"]
train_folder_list = ["coco","산단_주차장"]

if os.path.exists("./dataset/train.txt"):
    try:
        os.remove("./dataset/train.txt")
    except Exception as e:
        print(f"train.txt 파일을 삭제하는 중 오류가 발생했습니다: {e}")

if os.path.exists("./dataset/val.txt"):
    try:
        os.remove("./dataset/val.txt")
    except Exception as e:
        print(f"val.txt 파일을 삭제하는 중 오류가 발생했습니다: {e}")

for folder_name in folder_list:
    if folder_name in train_folder_list:
        train_img_path = os.path.join(dataset_path, folder_name, "train", "images")
        val_img_path = os.path.join(dataset_path, folder_name, "val", "images")

        train_img_list = os.listdir(train_img_path)

        try:
            val_img_list = os.listdir(val_img_path)
        except:
            val_img_list = []

        if train_img_list:
            train_img_list = sorted(train_img_list)
            for img_name in tqdm(train_img_list, desc="train"):
                save_path = os.path.join("./", folder_name, "train", "images")
                img_path = os.path.join(save_path, img_name)
                train_txt += f"{img_path}\n"

        if val_img_list:
            val_img_list = sorted(val_img_list)
            for img_name in tqdm(val_img_list, desc="val"):
                save_path = os.path.join("./", folder_name, "val", "images")
                img_path = os.path.join(save_path, img_name)
                val_txt += f"{img_path}\n"

if len(train_txt) > 0:
    with open("./dataset/train.txt", "w") as f:
        f.write(train_txt)

if len(val_txt) > 0:
    with open("./dataset/val.txt", "w") as f:
        f.write(val_txt)

# yolo_weight_path = "../weights/yolo/ms-ai_24-12-31-M.pt"
# yolo_weight_path = "../weights/yolo/yolov8m.pt"
# yolo_weight_path = "./train/yolov8m_convert/weights/best.pt"
yolo_weight_path = "./train/산단주차장/weights/last.pt"


model = YOLO(yolo_weight_path)
results = model.train(data=f'{os.getcwd()}/cfg/ms-ai-v1.3.yaml',
                        project = "./train",
                        name = f"산단주차장",
                        exist_ok = True,
                        epochs = 50,
                        imgsz = 640,
                        batch = 32,
                        device = '0',
                        save_period = 5,
                        freeze = 10,
                        plots = True,
                        workers = 4,
                        cache = "disk",
                        resume=True
                        # cache = False
                        )     

remove_npy()