
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime

    #train data move to main dataset

    # current_time = datetime.now().strftime("%Y%m%d")
# yolo_weight_path = "./train/weights/last/weights/last.pt"
yolo_weight_path = "./weights/yolo/ms-ai2405-finetune_M.pt"

model = YOLO(yolo_weight_path)
results = model.train(data='./cfg/main.yaml',
                        project = "./train/weights",
                        name = f"last",
                        exist_ok = True,
                        epochs = 10,
                        imgsz = 640,
                        batch = 0.6 ,
                        device = '0',
                        save_period = -1,
                        freeze = 10,
                        plots = True,
                        resume=False
                        )