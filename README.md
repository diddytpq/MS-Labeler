# VLM 기반 학습데이터 생성 프로젝트

## **📋 프로젝트 개요**

| 항목 | 내용 |
| --- | --- |
| **프로젝트명** | MS-Labeler |
| **개발 기간** | 2024년 ~ 현재 |
| **프로젝트 유형** | AI 기반 자동 라벨링 시스템 |
| **목적** | VLM(Vision Language Model)과 다중 AI 모델을 활용한 객체 검출 학습 데이터 자동 생성 |

---

## **🎯 프로젝트 목표**

객체 검출(Object Detection) 모델 학습을 위한 대규모 라벨링 데이터 생성의 **시간과 비용을 대폭 절감**하기 위해, 다양한 AI 모델을 파이프라인으로 연결하여 **반자동/자동 라벨링 시스템**을 구축

---

## **🔧 기술 스택**

**AI/ML 프레임워크**

| 기술 | 용도 |
| --- | --- |
| **PyTorch** | 딥러닝 프레임워크 |
| **Ultralytics YOLO** | 객체 검출 (YOLOv8) |
| **Transformers (HuggingFace)** | VLM/LLM 모델 로딩 |
| **vLLM** | 고속 LLM 추론 엔진 |
| **SAM2 (Segment Anything 2)** | 비디오 객체 세그멘테이션 및 추적 |

**Vision-Language Models**

| 모델 | 역할 |
| --- | --- |
| **Qwen2.5-VL / Qwen3-VL** | 검출 결과 검증 (Yes/No 분류) |
| **Moondream2** | Zero-shot 객체 검출 |

**GUI & 유틸리티**

| 기술 | 용도 |
| --- | --- |
| **PySide6 (Qt)** | GUI 라벨링 도구 |
| **OpenCV** | 이미지/비디오 처리 |
| **NumPy** | 수치 연산 |

---

## **🏗️ 시스템 아키텍처**

```python
┌─────────────────────────────────────────────────────────────────────────┐
│                        MS-Labeler Pipeline                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐ │
│  │  Input   │───▶│  Detection   │───▶│ Verification│───▶│  Tracking  │ │
│  │ (Video/  │    │  (YOLO +     │    │   (VLM)     │    │   (SAM2)   │ │
│  │  Image)  │    │  Moondream)  │    │             │    │            │ │
│  └──────────┘    └──────────────┘    └─────────────┘    └────────────┘ │
│                         │                   │                  │        │
│                         ▼                   ▼                  ▼        │
│                  ┌─────────────────────────────────────────────────┐   │
│                  │              NMS & Label Merge                   │   │
│                  └─────────────────────────────────────────────────┘   │
│                                        │                                │
│                                        ▼                                │
│                              ┌─────────────────┐                       │
│                              │  Output Labels  │                       │
│                              │  (YOLO Format)  │                       │
│                              └─────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## **📂 프로젝트 구조**

```python
MS-Labeler/
├── 📄 main.py                    # GUI 라벨링 도구 (PySide6)
├── 📄 auto_label_vlm_video.py    # 비디오 자동 라벨링 파이프라인
├── 📄 auto_label_vlm_img.py      # 이미지 자동 라벨링 파이프라인
├── 📄 auto_label_yolo.py         # YOLO 단독 라벨링
├── 📄 train.py                   # YOLO 모델 학습
├── 📄 train_detr.py              # DETR 모델 학습
├── 📄 edit_label.py              # 라벨 편집 유틸리티
├── 📄 make_detr_dataset_formot.py # DETR용 COCO 포맷 변환
│
├── 📁 ui/                        # GUI 관련 파일
│   ├── ai_labeling_ui.py
│   └── ai_labeling.ui
│
├── 📁 cfg/                       # 설정 파일
│   ├── ms-ai-v1.3.yaml
│   ├── detr.yaml
│   └── eval.yaml
│
├── 📁 eval/                      # 모델 평가 도구
│   ├── evaluate_models.py
│   ├── evaluate_models_optimized.py
│   └── evaluation_results/
│
├── 📁 dataset/                   # 데이터셋
├── 📁 train/                     # 학습 결과
├── 📁 runs/                      # 실행 로그
└── 📁 output/                    # 출력 결과
```

---

## **⚙️ 핵심 기능**

**1. 다단계 자동 라벨링 파이프라인**

```python
# 파이프라인 흐름 (auto_label_vlm_video.py)
1. YOLO 검출 → 초기 바운딩 박스 생성
2. Moondream2 Zero-shot → 미검출 객체 보완
3. NMS (Non-Maximum Suppression) → 중복 제거
4. VLM 검증 (Qwen) → False Positive 제거
5. SAM2 추적 → 시퀀스 전체로 라벨 전파
6. 최종 라벨 저장
```

**2. 지원 클래스**

```python
NAMES = {
    0: "person",      # 사람
    1: "bicycle",     # 자전거
    2: "car",         # 자동차
    3: "motorcycle",  # 오토바이
    4: "bus",         # 버스
    5: "truck",       # 트럭
    6: "fire"         # 화재
}
```

**3. GUI 라벨링 도구 (main.py)**

| 기능 | 설명 |
| --- | --- |
| **수동 라벨링** | 마우스 드래그로 바운딩 박스 생성/수정/삭제 |
| **YOLO 검출** | 선택한 클래스에 대해 YOLO 자동 검출 |
| **Zero-shot 검출** | Moondream2를 이용한 텍스트 기반 검출 |
| **SAM2 추적** | 다중 이미지에 대한 객체 추적 및 라벨 전파 |
| **키보드 단축키** | A/D(이전/다음), S(저장), 1-7(클래스 선택), Q(YOLO), W(Zero-shot) |

**4. VLM 기반 검증 시스템**

```python
# Qwen VLM을 이용한 검출 결과 검증
def run_LLM(img_buffer, label):
    # 각 검출 박스에 대해 VLM에 질의
    # "You determine {class_name} in this image."
    # → "yes" 응답시만 라벨 유지
```

**5. SAM2 비디오 객체 추적**

```python
# 양방향 전파로 시퀀스 전체 라벨링
def run_sam(img_buffer, label, img_path):
    # 정방향 전파
    for frame in predictor.propagate_in_video(inference_state):
        # 마스크 → 바운딩 박스 변환
    
    # 역방향 전파
    for frame in predictor.propagate_in_video(inference_state, reverse=True):
        # 마스크 → 바운딩 박스 변환
```

---

## **📊 평가 시스템**

eval/ 디렉토리에서 다양한 평가 메트릭 제공:

- **mAP (mean Average Precision)**
- **클래스별 AP**
- **Precision / Recall**
- **F1 Score**

---

## **🔄 데이터 포맷 지원**

| 포맷 | 용도 |
| --- | --- |
| **YOLO TXT** | 기본 라벨 포맷 (cls, x_center, y_center, w, h) |
| **COCO JSON** | DETR 학습용 변환 지원 |

```python
# YOLO → COCO 변환 (make_detr_dataset_formot.py)
def convert_yolo_to_coco_combined(input_dirs, yaml_path, output_path):
    # 다중 데이터셋 폴더를 하나의 COCO JSON으로 통합
```

---

## **🖥️ 실행 환경**

### **하드웨어 요구사항**

- **GPU**: NVIDIA GPU (VRAM 16GB+ 권장)
- **RAM**: 32GB+ 권장

**소프트웨어 요구사항**

```python
Python 3.10+
PyTorch 2.0+
CUDA 11.8+
```

**주요 의존성**

```python
ultralytics
transformers
pyside6
opencv-python
numpy
vllm
sam2
qwen-vl-utils
```

---

## **📈 성과 및 기대효과**

| 항목 | 효과 |
| --- | --- |
| **라벨링 시간 단축** | 수동 대비 70-80% 시간 절감 |
| **품질 향상** | VLM 검증으로 False Positive 감소 |
| **일관성** | SAM2 추적으로 시퀀스 라벨 일관성 확보 |
| **확장성** | 다양한 검출 클래스로 쉽게 확장 가 |

---
