
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime

# yolo_weight_path = "./weights/yolo/ms-ai2405-finetune_M.pt"

yolo_weight_path = "./train/weights/last/weights/last.pt"

model = YOLO(yolo_weight_path)
results = model.train(data='./cfg/main.yaml',
                        project = "./train",
                        name = f"last",
                        exist_ok = True,
                        epochs = 10,
                        imgsz = 640,
                        batch = 0.6 ,
                        device = '0',
                        save_period = -1,
                        freeze = 10,
                        plots = True,
                        resume=True
                        )