from ultralytics import YOLO
import os
# 1. YOLO 모델 불러오기
# model_path = "./weights/yolo/yolov8m.pt"  # YOLOv8 모델 가중치 파일 (필요에 따라 yolov8s.pt, yolov8m.pt 등을 선택)
model_path = "./weights/yolo/ms-ai_v1.3_24-11-19-M.pt"  # YOLOv8 모델 가중치 파일 (필요에 따라 yolov8s.pt, yolov8m.pt 등을 선택)


model = YOLO(model_path)


# 3. COCO 데이터셋에 대해 평가 수행
results = model.val(
    data=f'{os.getcwd()}/cfg/ms-ai-v1.3.yaml',
    conf=0.001,  # 낮은 confidence로 모든 예측을 평가
    iou=0.50,    # IoU 임계값
    task="val",  # 평가 모드
    imgsz=640,   # 입력 이미지 크기
    device=0     # GPU 사용 (0), CPU 사용은 "cpu"
)


# 4. 전체 mAP 출력
print("Overall Evaluation Results:")
print(f"mAP@50: {results.box.maps[0]:.4f}")       # mAP at IoU=0.5
print(f"mAP@50-95: {results.box.maps.mean():.4f}")  # mAP across IoU=0.5 to 0.95
