from ultralytics import YOLO
import os

val_txt = ""
dataset_path = os.path.join(os.getcwd(), "dataset")
folder_list = os.listdir(dataset_path)

train_folder_list = ["산단_주차장"]

if os.path.exists("./dataset/val_test.txt"):
    try:
        os.remove("./dataset/val_test.txt")
    except Exception as e:
        print(f"val.txt 파일을 삭제하는 중 오류가 발생했습니다: {e}")

for folder_name in folder_list:
    if folder_name in train_folder_list:
        val_img_path = os.path.join(dataset_path, folder_name, "val", "images")

        try:
            val_img_list = os.listdir(val_img_path)
        except:
            val_img_list = []

        if val_img_list:
            val_img_list = sorted(val_img_list)
            for img_name in val_img_list:
                save_path = os.path.join("./", folder_name, "val", "images")
                img_path = os.path.join(save_path, img_name)
                val_txt += f"{img_path}\n"


if len(val_txt) > 0:
    with open("./dataset/val_test.txt", "w") as f:
        f.write(val_txt)

# 1. YOLO 모델 불러오기
# model_path = "../weights/yolo/yolov8m_standard.pt"  # YOLOv8 모델 가중치 파일 (필요에 따라 yolov8s.pt, yolov8m.pt 등을 선택)
# model_path = "../weights/yolo/ms-ai_24-12-31-M.pt"  # YOLOv8 모델 가중치 파일 (필요에 따라 yolov8s.pt, yolov8m.pt 등을 선택)
model_path = "./train/산단주차장/weights/best.pt"  # YOLOv8 모델 가중치 파일 (필요에 따라 yolov8s.pt, yolov8m.pt 등을 선택)

model = YOLO(model_path)


# 3. COCO 데이터셋에 대해 평가 수행
results = model.val(
    data=f'{os.getcwd()}/cfg/ms-ai-v1.3_val.yaml',
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
