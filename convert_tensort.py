import numpy as np
from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] # current directory
from ultralytics import YOLO

# 모델 로드

for i in range(1,5):
    model_name = f"ms-ai2405-finetune_M{i}"
    model_path = os.path.join(".", "train", "weights", "last", "weights", f"{model_name}.pt")

    model = YOLO(model_path)
    # model = YOLO("./weight/yolo/pt/ms-ai2401-finetune_M.pt")

    # PyTorch to TensorRT
    # model.export(format='engine', device=0, half=True, batch = 1)
    model.export(format='engine', device=0, half=True, int8 = True, batch = i)

# model_name = "ms-ai2405-finetune_M2"
# model = YOLO(f"./weight/yolo/pt/{model_name}.pt")

# model.export(format='engine', device=0, half=True, batch = 2)


# model_name = "ms-ai2405-finetune_M3"
# model = YOLO(f"./weight/yolo/pt/{model_name}.pt")
# model.export(format='engine', device=0, half=True, batch = 3)


# model_name = "ms-ai2405-finetune_M4"
# model = YOLO(f"./weight/yolo/pt/{model_name}.pt")
# model.export(format='engine', device=0, half=True, batch = 4)

