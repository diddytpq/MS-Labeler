import os
from pathlib import Path
import cv2
import torch
from utils import make_square_bbox, get_yolo_label, get_zero_shot_label, nms, plot_one_box, get_img_buffer, nms_test, create_dataset_list, save_final_dataset, train_model, merge_overlapping_boxes

import numpy as np

from PIL import Image

from tqdm import tqdm
import time

from ultralytics import YOLO
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForZeroShotObjectDetection

from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T

def check_LLM(model, processor, img_buffer, label, verbose = False):
    final_label = []
    conversation = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                            },
                            {"type": "text", "text": "this is person? answer yes or no."},
                        ],
                    }
                ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    for i in tqdm(range(len(img_buffer)), desc="Processing Image", leave=False):
        img = img_buffer[i]
        bboxes = label[i]
        new_label = []

        for cls, x1, y1, x2, y2, score in bboxes:
            t1 = time.time()
            # extend_x1, extend_y1, extend_x2, extend_y2, score, cls = make_square_bbox([cls, x1, y1, x2, y2, score], img, extend_ratio=10)
            # cropped_img = img[int(extend_y1) : int(extend_y2), int(extend_x1) : int(extend_x2)]
            # pil_img_1 = Image.fromarray(cropped_img).convert('RGB')


            new_x1, new_y1, new_x2, new_y2, score, cls = make_square_bbox([cls, x1, y1, x2, y2, score], img, extend_ratio = 1.2)
            cropped_img = img[int(new_y1) : int(new_y2), int(new_x1) : int(new_x2)]
            pil_img_2 = Image.fromarray(cropped_img).convert('RGB')

            inputs = processor(text=[text_prompt], images=[pil_img_2], padding=True, return_tensors="pt").to("cuda")
            
            output_ids = model.generate(**inputs, max_new_tokens=10)
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            answer = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            print(answer)

            if "yes" in answer:
                new_label.append([cls, x1, y1, x2, y2, score])
                print("ADD person bbox")


            if verbose:
                test_img = img.copy()
                # cv2.imshow(f"cropped_img_extend", cropped_img_extend)
                # cv2.rectangle(test_img, (int(extend_x1), int(extend_y1)), (int(extend_x2), int(extend_y2)), (0,0,255), thickness=2, lineType=cv2.LINE_AA)
                cv2.rectangle(test_img, (int(new_x1), int(new_y1)), (int(new_x2), int(new_y2)), (255,0,0), thickness=2, lineType=cv2.LINE_AA)
                cv2.rectangle(test_img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), thickness=2, lineType=cv2.LINE_AA)
                
                cv2.imshow(f"example", test_img)

                print("-----------------------")
                print(time.time() - t1)
                # print("question 1 : ",res_main)
                print("question : ",answer)

                # cv2.waitKey(0)
                cv2.waitKey(1)

                # cv2.destroyAllWindows()

        # final_label.append(new_label)
        label[i] = new_label
    
    return label


TEST = True

HOME = Path.home()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

video_list_path = os.path.join(os.getcwd(), "videos")
video_name_list = os.listdir(video_list_path)
# yolo_weight_path = os.path.join(os.getcwd(), "weights", "yolo", "ms-ai2407-finetune_M.pt")
# yolo_weight_path = os.path.join(os.getcwd(), "train", "weights", "last", "weights", "last.pt")
yolo_weight_path = os.path.join(os.getcwd(), "weights", "yolo", "ms-ai2405-finetune_M.pt")


with torch.no_grad():
    for video_name in video_name_list:
        camera_name = "test"
        video_name = "09.08.09_침입.avi"

        yolo_model = YOLO(yolo_weight_path)  # load a pretrained model (recommended for training)\

        # zero_shot_ob_model = AutoModelForCausalLM.from_pretrained("./weights/Florence_2_large", trust_remote_code=True).to(device)
        # processor = AutoProcessor.from_pretrained("./weights/Florence_2_large", trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(os.path.join(os.getcwd(),"weights/grounding-dino-base"))
        zero_shot_ob_model = AutoModelForZeroShotObjectDetection.from_pretrained(os.path.join(os.getcwd(),"weights/grounding-dino-base")).to(device)
        # video_name = "people-walking.mp4"
        img_save_dir = os.path.join(os.getcwd(), "dataset", camera_name, "train", "images")
        label_save_dir = os.path.join(os.getcwd(), "dataset", camera_name, "train", "labels")


        if TEST:
            img_save_path = f"./results/{video_name}/"
            os.makedirs(img_save_path, exist_ok=True)

        img_buffer = get_img_buffer(video_path = os.path.join(video_list_path, camera_name, video_name))
        # img_buffer = [cv2.imread("./a.png"), cv2.imread("./b.png"),cv2.imread("./c.png"),cv2.imread("./d.png"),cv2.imread("./e.png"),cv2.imread("./f.png"),cv2.imread("./g.png")]
        # img_buffer = [cv2.imread("./Screenshot from 2024-07-10 16-09-51.png")]

        yolo_label_data = get_yolo_label(model = yolo_model, buffer = img_buffer)
        zeroshot_label_data = get_zero_shot_label(model = zero_shot_ob_model, buffer = img_buffer, processor = processor, device = device)

        non_llm_input_bboxes = []
        llm_input_bboxes = []

        for i in range(len(yolo_label_data)):
            all_boxes = yolo_label_data[i] + zeroshot_label_data[i]

            # nms_boxes = nms(all_boxes, iou_threshold=0.9)
            # nms_boxes, non_nms_boxes = nms_test(yolo_label_data[i], zeroshot_label_data[i], iou_threshold=0.6)
            nms_boxes_original_format, non_nms_boxes_original_format = nms_test(yolo_label_data[i], zeroshot_label_data[i], iou_threshold=0.6)

            
            # nms_boxes_original_format = [[0, box[0], box[1], box[2] - box[0], box[3] - box[1], box[4]] for box in nms_boxes]
            # non_nms_boxes_original_format = [[0, box[0], box[1], box[2] - box[0], box[3] - box[1], box[4]] for box in non_nms_boxes]

            non_llm_input_bboxes.append(nms_boxes_original_format)
            llm_input_bboxes.append(non_nms_boxes_original_format)

        del yolo_model
        del zero_shot_ob_model
        del processor

        llm_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "./weights/Qwen2-VL-2B", torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("./weights/Qwen2-VL-2B")

        llm_model.eval()

        LLM_label = check_LLM(model = llm_model,
                                    processor = processor,
                                    img_buffer = img_buffer,
                                    label = llm_input_bboxes,
                                    verbose = True
                                    )
        bboxes_list = []

        for i in range(len(img_buffer)):
            bboxes_list.append(non_llm_input_bboxes[i] + LLM_label[i])


        final_bboxes_list = bboxes_list
        # final_bboxes_list = []
        # for bbox_list in bboxes_list:
        #     final_bboxes_list.append(merge_overlapping_boxes(bbox_list, iou_threshold = 0.5)) 

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

            # print(final_bboxes_list)
            

        break

# dataset_path = os.path.join("./", "dataset")

# create_dataset_list(dataset_path)

# yolo_weight_path = './weights/yolo/ms-ai2401-finetune.pt'
# train_model(yolo_weight_path = yolo_weight_path)


cv2.destroyAllWindows()
cmd = "chmod 777 -R ./"
os.system(cmd)