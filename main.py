import os
from pathlib import Path
import cv2
import torch
from utils import check_LLM, get_yolo_label, get_zero_shot_label, save_final_dataset, plot_one_box, get_img_buffer, nms_test, create_dataset_list, train_model

import numpy as np
from PIL import Image
from tqdm import tqdm

from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoModelForZeroShotObjectDetection, logging

logging.set_verbosity_error()
logging.disable_progress_bar()

from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

import torchvision.transforms as T

TEST = True

HOME = Path.home()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
camera_list_path = os.path.join(os.getcwd(), "videos", "117.17.159.143")
camera_name_list = os.listdir(camera_list_path)

yolo_weight_path = os.path.join(os.getcwd(), "weights", "yolo", "ms-ai2407-finetune_M.pt")

for camera_name in tqdm(camera_name_list, desc="Processing Cameras") :
    date_list_path = os.path.join(camera_list_path, camera_name)
    date_list = os.listdir(date_list_path)

    for date in date_list:
        video_list_path = os.path.join(camera_list_path, camera_name, date, "videos")

        video_name_list = os.listdir(video_list_path)
        with torch.no_grad():
            for video_name in tqdm(video_name_list, desc=f"Processing Videos for {camera_name} on {date}", leave=True):
                # video_name = "09.52.44_침입.mp4"

                yolo_model = YOLO(yolo_weight_path)  # load a pretrained model (recommended for training)\

                processor = AutoProcessor.from_pretrained(os.path.join(os.getcwd(),"weights/grounding-dino-base"))
                zero_shot_ob_model = AutoModelForZeroShotObjectDetection.from_pretrained(os.path.join(os.getcwd(),"weights/grounding-dino-base")).to(device)
                # video_name = "people-walking.mp4"
                img_save_dir = os.path.join(os.getcwd(), "dataset", camera_name, "train", "images")
                label_save_dir = os.path.join(os.getcwd(), "dataset", camera_name, "train", "labels")

                if TEST:
                    img_save_path = os.getcwd() + f"/results/{camera_name}/{date}/{video_name}/"
                    os.makedirs(img_save_path, exist_ok=True)

                img_buffer = get_img_buffer(video_path = os.path.join(video_list_path, video_name))
                # img_buffer = [cv2.imread("./a.png"), cv2.imread("./b.png"),cv2.imread("./c.png"),cv2.imread("./d.png"),cv2.imread("./e.png"),cv2.imread("./f.png"),cv2.imread("./g.png")]
                # img_buffer = [cv2.imread("./Screenshot from 2024-07-10 16-09-51.png")]

                yolo_label_data = get_yolo_label(model = yolo_model, buffer = img_buffer)
                zeroshot_label_data = get_zero_shot_label(model = zero_shot_ob_model, buffer = img_buffer, processor = processor, device = device)


                non_llm_input_bboxes = []
                llm_input_bboxes = []

                for i in range(len(yolo_label_data)):
                    all_boxes = yolo_label_data[i] + zeroshot_label_data[i]

                    # nms_boxes = nms(all_boxes, iou_threshold=0.9)
                    nms_boxes, non_nms_boxes = nms_test(yolo_label_data[i], zeroshot_label_data[i], iou_threshold=0.75)

                    
                    nms_boxes_original_format = [[0, box[0], box[1], box[2] - box[0], box[3] - box[1], box[4]] for box in nms_boxes]
                    non_nms_boxes_original_format = [[0, box[0], box[1], box[2] - box[0], box[3] - box[1], box[4]] for box in non_nms_boxes]

                    non_llm_input_bboxes.append(nms_boxes_original_format)
                    llm_input_bboxes.append(non_nms_boxes_original_format)

                del yolo_model
                del zero_shot_ob_model
                del processor

                llm_model = AutoModel.from_pretrained(os.getcwd() + "/weights/InternVL2-4B",
                                                        torch_dtype=torch.bfloat16,
                                                        low_cpu_mem_usage=True,
                                                        trust_remote_code=True).eval().cuda()

                tokenizer = AutoTokenizer.from_pretrained(os.getcwd() + "/weights/InternVL2-4B", trust_remote_code=True)

                llm_model.eval()

                LLM_label = check_LLM(model = llm_model,
                                            tokenizer = tokenizer,
                                            img_buffer = img_buffer,
                                            label = llm_input_bboxes,
                                            verbose = False
                                            )
                final_bboxes_list = []

                for i in range(len(img_buffer)):
                    final_bboxes_list.append(non_llm_input_bboxes[i] + LLM_label[i])

                del llm_model
                del tokenizer
                
                if TEST:
                    for i, img in enumerate(img_buffer):
                        img_yolo = img.copy()
                        img_zero = img.copy()
                        img_nms = img.copy()
                        img_final = img.copy()

                        if len(yolo_label_data):
                            for cls, x1, y1, x2, y2, score in yolo_label_data[i]:
                                xyxy = [x1, y1, x2, y2]
                                plot_one_box(xyxy, img_yolo, label=None, color=(255,0,0), line_thickness=2) # 박스 그리기

                        if len(zeroshot_label_data):
                            for cls, x1, y1, x2, y2, score in zeroshot_label_data[i]:
                                xyxy = [int(x1), int(y1), int(x2), int(y2)]
                                plot_one_box(xyxy, img_zero, label=None, color=(181,186,126), line_thickness=2) # 박스 그리기

                        if len(non_llm_input_bboxes):
                            for cls, x1, y1, x2, y2, score in non_llm_input_bboxes[i]:
                                xyxy = [x1, y1, x2, y2]
                                plot_one_box(xyxy, img_nms, label=None, color=(0,0,255), line_thickness=2) # 박스 그리기

                        if len(final_bboxes_list):
                            for cls, x1, y1, x2, y2, score in final_bboxes_list[i]:
                                xyxy = [int(x1), int(y1), int(x2), int(y2)]
                                plot_one_box(xyxy, img_final, label=None, color=(0,255,0), line_thickness=2) # 박스 그리기

                        concat_img_1 = np.vstack((img_yolo, img_zero))
                        concat_img_2 = np.vstack((img_nms, img_final))

                        concat_img_final = np.hstack((concat_img_1, concat_img_2))

                        output_path = os.path.join(img_save_path, f"{i}.png")

                        cv2.imwrite(output_path, concat_img_final)

                        # cv2.imshow("frame", img)
                        # cv2.imshow("img_new", img_new)
                        # while True:
                        #     cv2.imshow("frame", img)
                        #     cv2.imshow("img_new", img_new)
                        #     key = cv2.waitKey(0)

                        #     if key == 27 : break

                
                save_final_dataset(video_name = video_name,
                                   date = date,
                                   img_buffer = img_buffer, 
                                   label = final_bboxes_list,
                                   img_save_dir = img_save_dir, 
                                   label_save_dir = label_save_dir,
                                   )
                
dataset_path = os.path.join(os.getcwd(), "dataset")
create_dataset_list(dataset_path)

train_model(yolo_weight_path = yolo_weight_path)

cv2.destroyAllWindows()
cmd = "chmod 777 -R ./"
os.system(cmd)