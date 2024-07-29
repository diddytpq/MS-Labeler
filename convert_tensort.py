import numpy as np
from pathlib import Path
import sys
import os

from ultralytics import YOLO

# 모델 로드

weight_path = os.path.join(os.getcwd(), "train", "last", "weights")
weight_list = os.listdir(weight_path)

cfg_path = os.path.join(os.getcwd(), "cfg", "main.yaml")

if "last.pt" in weight_list:
    for i in range(1,5):
        model_name = f"ms-ai2407-finetune_M{i}"

        cmd = f"cp {weight_path}/last.pt {weight_path}/{model_name}.pt"
        os.system(cmd)

        model_path = os.path.join(os.getcwd(), "train", "last", "weights", f"{model_name}.pt")

        model = YOLO(model_path)
        # model = YOLO("./weight/yolo/pt/ms-ai2401-finetune_M.pt")

        # PyTorch to TensorRT
        # model.export(format='engine', device=0, half=True, batch = 1)
        model.export(format='engine', device=0, int8 = True, batch = i, data = cfg_path)

        cmd = f"rm -rf {weight_path}/{model_name}.cache"
        os.system(cmd)
        cmd = f"rm -rf {weight_path}/{model_name}.onnx"
        os.system(cmd)
        cmd = f"rm -rf {weight_path}/{model_name}.pt"
        os.system(cmd)


os.system("chmod 777 -R ./")