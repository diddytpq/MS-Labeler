import os
from pathlib import Path
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
import matplotlib.pyplot as plt
import shutil
from utils import (check_LLM, 
                   get_yolo_label, 
                   get_zero_shot_label, 
                   build_transform, 
                   plot_one_box, 
                   nms_test, 
                   make_square_bbox, 
                   dynamic_preprocess, 
                   merge_overlapping_boxes
)
from transformers import(AutoProcessor, 
                         AutoModelForCausalLM, 
                         AutoTokenizer, 
                         AutoModel, 
                         AutoModelForZeroShotObjectDetection, 
                         logging
) 

logging.set_verbosity_error()
logging.disable_progress_bar()

TEST = True

HOME = Path.home()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

video_list_path = os.path.join(os.getcwd(), "videos")
video_name_list = os.listdir(video_list_path)
# yolo_weight_path = os.path.join(os.getcwd(), "weights", "yolo", "ms-ai_24-07-30-M.pt")
yolo_weight_path = os.path.join(os.getcwd(), "weights", "yolo", "yolov8m.pt")

# yolo_weight_path = os.path.join(os.getcwd(), "weights", "yolo", "ms-ai2401-finetune.pt")

# yolo_weight_path = os.path.join(os.getcwd(), "train", "weights", "last", "weights", "last.pt")

def get_bboxes(mask):
    mask = np.squeeze(mask)
    mask_uint8 = mask.astype(np.uint8)  # 데이터 타입 변환
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(contour) for contour in contours]


def get_label_data(label_path, half_index, img_size = (640, 480)):
    label_file_name = sorted(os.listdir(label_path))
    print(f"{half_index}.txt")
    if os.path.exists(os.path.join(label_path, f"{half_index}.txt")):
        label_list = []
        with open(os.path.join(label_path, f"{half_index}.txt"), 'r') as file:
            for line in file:
                line = line.strip()
                cls, xc, yc, w, h = line.split(" ")
                color = (tuple(np.random.randint(0, 255, size=3).tolist()))
                label_list.append([int(cls), float(xc) * 640, float(yc) * 480, float(w) * 640, float(h) * 480, color])

    return label_list

def overlay_mask_on_image(image, mask, color):
    overlay = image.copy()
    alpha = color[3]  # 투명도
    for i in range(3):
        overlay[:, :, i] = np.where(mask, image[:, :, i] * (1 - alpha) + color[i] * 255 * alpha, image[:, :, i])
    return overlay

def SAM_label(img_buffer, img_path, label):
    # from lib.segment_anything_2.sam2.build_sam import build_sam2_video_predictor
    # from lib.segment_anything_2 import sam2
    from sam2.build_sam import build_sam2_video_predictor

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    # sam2_checkpoint = os.path.join(os.getcwd(), "weights", "segment_anything_2", "sam2_hiera_large.pt")
    sam2_checkpoint = os.path.join(os.getcwd(), "weights", "segment_anything_2", "sam2_hiera_large.pt")

    # model_cfg = os.path.join(os.getcwd(), "weights", "segment_anything_2", "sam2_hiera_l.yaml")
    model_cfg = "./sam2_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    inference_state = predictor.init_state(video_path=img_path)
    
    label_dict = {}
    final_label = {}
    for frame_num, bbox_list in label.items():
        if len(bbox_list):
            for cls, x1, y1, x2, y2, score in bbox_list:
                predictor.reset_state(inference_state)

                ori = cv2.rectangle(img_buffer[frame_num].copy(), (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)


                ann_frame_idx = frame_num  # the frame index we interact with
                ann_obj_id = int(frame_num)  # give a unique id to each object we interact with (it can be any integers)

                points = np.array([[(x2 + x1)/2, (y2 + y1)/2]], dtype=np.float32)
                labels = np.array([1], np.int32)
                _, out_obj_ids, out_mask_logits = predictor.add_new_points(inference_state=inference_state,
                                                                            frame_idx=ann_frame_idx,
                                                                            obj_id=ann_obj_id,
                                                                            points=points,
                                                                            labels=labels,
                                                                            )

                video_segments = {} 

                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                    video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.5).cpu().numpy()
                                                    for i, out_obj_id in enumerate(out_obj_ids)
                                                    }
                    
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):

                    video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.5).cpu().numpy()
                                                    for i, out_obj_id in enumerate(out_obj_ids)
                                                    }
                

                
                for out_frame_idx in video_segments.keys():
                    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                        # new_label = []
                        binary_image = out_mask

                        h, w = binary_image.shape[-2:]

                        bboxes = get_bboxes(binary_image)
                        for bbox in bboxes:
                            x, y, w, h = bbox
                            # new_label.append([0, x, y, x+w, y+h, 1])

                            if w * h < 100: continue

                            if out_frame_idx in label_dict.keys():
                                label_dict[out_frame_idx].append([0, x, y, x+w, y+h, 1])

                            else:
                                label_dict[out_frame_idx] = [[0, x, y, x+w, y+h, 1]]

                            # # 이진 이미지를 색상 이미지로 변환
                            # color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)  
                            # color_image = img_buffer[out_frame_idx]
                            # cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                            # # for i in range(3):
                            # #     color_image[:, :, i] = binary_image * int(color[i] * 255)

                            # # OpenCV를 사용하여 이미지 표시
                            # cv2.imshow("ori", ori)
                            # cv2.imshow("Overlayed Image", color_image)
                            
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()

    for frame_num, box_list in label_dict.items():
        final_label[frame_num] = merge_overlapping_boxes(box_list, iou_threshold = 0.3)

    return final_label

def get_img_buffer(video_path):
    cap = cv2.VideoCapture(video_path)

    frame_num = 0

    img_buffer = {}

    while True:
        ret, img = cap.read()
        
        if ret == False:
            break
        
        frame_num += 1

        img_buffer[len(img_buffer)] = img

    return img_buffer

def save_img(video_name, img_buffer, img_save_dir):
    if os.path.exists(img_save_dir):
        shutil.rmtree(img_save_dir)
    os.makedirs(img_save_dir, exist_ok=True)

    video_name = video_name[:-4]

    for frame_num, img in img_buffer.items():
        cv2.imwrite(f"{img_save_dir}/{frame_num}.jpg", img)

def get_yolo_label(model, buffer):
    total_label = {}
    for frame_num, img in buffer.items():
        label = []

        if frame_num % 3 == 0:
            heigth, width = img.shape[0], img.shape[1]

            pred = model(img, 
                        imgsz = 640, 
                        conf = 0.22, 
                        iou = 0.5, 
                        verbose=False)

            boxes = pred[0].boxes.data.cpu().numpy().astype(float)

            for i in range(len(boxes)):
                data = boxes[i]
                
                if len(data) != 0:
                    x1, y1, x2, y2 = data[0:4].astype('int') # float64 to int
                    # conf = data[4]
                    cls = data[-1].astype('int')
                    # ind = tracks[i, 7].astype('int') # float64 to int

                    label.append([cls, x1, y1, x2, y2, data[4]])

        # total_label.append(label)
        total_label[frame_num] = label

    return total_label

def get_zero_shot_label(processor, model, device, buffer):
    total_label = {}


    for frame_num, img in buffer.items():
        label_list = []

        if frame_num % 3 == 0:
            text = "person. car. dog. cat. tree."

            pil_img = Image.fromarray(img.astype('uint8'), 'RGB')

            inputs = processor(images=pil_img, text=text, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)

                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=0.1,
                    # text_threshold=0.55,
                    text_threshold=0.60,
                    target_sizes=[pil_img.size[::-1]]
                )

            for i, boxes in enumerate(results[0]["boxes"].tolist()):
                if results[0]["labels"][i] == "person":
                    label_list.append([0] + boxes + [results[0]["scores"][i]])

            label_list = merge_overlapping_boxes(label_list, iou_threshold = 0.5)

        total_label[frame_num] = label_list

    return total_label

def check_LLM(model, tokenizer, img_buffer, label, verbose = False):
    transform = build_transform(input_size=448)

    generation_config_1 = dict(num_beams=1,
                            max_new_tokens=16,
                            do_sample=False,
                            )
    
    generation_config_2 = dict(num_beams=1,
                            max_new_tokens=1,
                            do_sample=False,
                            )
    for frame_num, img in img_buffer.items():
        bboxes = label[frame_num]
        new_label = []

        for cls, x1, y1, x2, y2, score in bboxes:
            res_2 = None

            extend_x1, extend_y1, extend_x2, extend_y2, score, cls = make_square_bbox([cls, x1, y1, x2, y2, score], img, extend_ratio=10)
            cropped_img_extend = img[int(extend_y1) : int(extend_y2), int(extend_x1) : int(extend_x2)]
            pil_img = Image.fromarray(cropped_img_extend).convert('RGB')
            images = dynamic_preprocess(pil_img, image_size=448, use_thumbnail=True, max_num=6)
            pixel_values_1 = [transform(image) for image in images]
            pixel_values_1 = torch.stack(pixel_values_1).to(torch.bfloat16).cuda()
            question = '<image>\Please describe the image.'
            _, history = model.chat(tokenizer, pixel_values_1, question, generation_config_1, history=None, return_history=True)

            new_x1, new_y1, new_x2, new_y2, score, cls = make_square_bbox([cls, x1, y1, x2, y2, score], img, extend_ratio = 1.2)
            cropped_img = img[int(new_y1) : int(new_y2), int(new_x1) : int(new_x2)]

            pil_img = Image.fromarray(cropped_img).convert('RGB')
            images = dynamic_preprocess(pil_img, image_size=448, use_thumbnail=True, max_num=6)
            pixel_values_1 = [transform(image) for image in images]
            pixel_values_1 = torch.stack(pixel_values_1).to(torch.bfloat16).cuda()

            question = "<image>\nDo you see person in this image?"

            res = model.chat(tokenizer, pixel_values_1, question, generation_config_2, history=history, return_history=False).lower()

            # if "yes" in res_1 and "no" not in res_1.split(" "):
            if "yes" in res:
                new_label.append([cls, x1, y1, x2, y2, score])
                print("ADD person bbox")

            if verbose:
                test_img = img.copy()
                # cv2.imshow(f"cropped_img_extend", cropped_img_extend)
                cv2.rectangle(test_img, (int(extend_x1), int(extend_y1)), (int(extend_x2), int(extend_y2)), (0,0,255), thickness=2, lineType=cv2.LINE_AA)
                cv2.rectangle(test_img, (int(new_x1), int(new_y1)), (int(new_x2), int(new_y2)), (255,0,0), thickness=2, lineType=cv2.LINE_AA)
                cv2.rectangle(test_img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), thickness=2, lineType=cv2.LINE_AA)
                
                cv2.imshow(f"example", test_img)

                print("-----------------------")
                print("question 2 : ",res)
                if res_2 is not None:
                    print("question 3 : ",res_2)

                cv2.waitKey(1)
                # cv2.destroyAllWindows()

        label[frame_num] = new_label
        del pixel_values_1

    return label

def save_final_dataset(video_name, date, img_buffer, label_buffer, img_save_dir, label_save_dir):
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)

    video_name = video_name[:-4]

    for frame_num, img in img_buffer.items():
        if frame_num % 5 == 0 :
            label_txt = ""

            # cv2.imwrite(f"{img_save_dir}/{date}_{video_name}_{str(0) * (4 - len(str(i)))}{i}.jpg", img_buffer[i])
            cv2.imwrite(f"{img_save_dir}/{frame_num}.jpg", img)

            width, heigth = img.shape[1], img.shape[0]

            if frame_num in label_buffer.keys():
                for cls, x1, y1, x2, y2, score in label_buffer[frame_num]:
                    x1 = float(x1/width)
                    x2 = float(x2/width)
                    y1 = float(y1/heigth)
                    y2 = float(y2/heigth)

                    w = np.round(x2 - x1, 3)
                    h = np.round(y2 - y1, 3)

                    ncx = np.round(x1 + w / 2,3)
                    ncy = np.round(y1 + h / 2,3)

                    label_txt += f"{cls} {ncx} {ncy} {w} {h}\n"

                if len(label_txt) > 0:
                    # label_name = f"{label_save_dir}/{date}_{video_name}_{str(0) * (4 - len(str(i)))}{i}.txt"
                    label_name = f"{label_save_dir}/{frame_num}.txt"

                    with open(label_name, "w") as f:
                        f.write(label_txt)
        

def save_result_img(img_save_path, img_buffer, yolo_label_data, zeroshot_label_data, non_llm_input_bboxes, bboxes_list, sam_label):
    for frame_num, img in img_buffer.items():
            img_yolo, img_zero, img_nms, img_final, sam_final = [img.copy() for _ in range(5)]

            if len(yolo_label_data[frame_num]):
                for cls, x1, y1, x2, y2, score in yolo_label_data[frame_num]:
                    xyxy = [x1, y1, x2, y2]
                    plot_one_box(xyxy, img_yolo, label=None, color=(255,0,0), line_thickness=2) # 박스 그리기

            if len(zeroshot_label_data[frame_num]):
                for cls, x1, y1, x2, y2, score in zeroshot_label_data[frame_num]:
                    xyxy = [int(x1), int(y1), int(x2), int(y2)]
                    plot_one_box(xyxy, img_zero, label=None, color=(181,186,126), line_thickness=2) # 박스 그리기

            if len(non_llm_input_bboxes[frame_num]):
                for cls, x1, y1, x2, y2, score in non_llm_input_bboxes[frame_num]:
                    xyxy = [x1, y1, x2, y2]
                    plot_one_box(xyxy, img_nms, label=None, color=(0,0,255), line_thickness=2) # 박스 그리기

            if len(bboxes_list[frame_num]):
                for cls, x1, y1, x2, y2, score in bboxes_list[frame_num]:
                    xyxy = [int(x1), int(y1), int(x2), int(y2)]
                    plot_one_box(xyxy, img_final, label=None, color=(0,255,0), line_thickness=2) # 박스 그리기

            if frame_num in sam_label.keys():
                for cls, x1, y1, x2, y2, score in sam_label[frame_num]:
                    xyxy = [int(x1), int(y1), int(x2), int(y2)]
                    plot_one_box(xyxy, sam_final, label=None, color=(192,154,25), line_thickness=2) # 박스 그리기

            concat_img_1 = np.vstack((img_yolo, img_zero))
            # concat_img_2 = np.vstack((img_nms, img_final))
            concat_img_2 = np.vstack((img_final, sam_final))

            concat_img_final = np.hstack((concat_img_1, concat_img_2))
            # concat_img_final = np.vstack((img_final, sam_final))
            output_path = os.path.join(img_save_path, f"{frame_num}.png")
            cv2.imwrite(output_path, concat_img_final)

with torch.no_grad():
    camera_name = "test_video2"
    video_name = "08.48.12_배회1111.avi"
    # video_name = "미르스타디움_6.mp4"

    date = "test"

    yolo_model = YOLO(yolo_weight_path)  # load a pretrained model (recommended for training)\

    processor = AutoProcessor.from_pretrained(os.path.join(os.getcwd(),"weights/grounding-dino-base"))
    zero_shot_ob_model = AutoModelForZeroShotObjectDetection.from_pretrained(os.path.join(os.getcwd(),"weights/grounding-dino-base")).to(device)
    # video_name = "people-walking.mp4"
    img_save_dir = os.path.join(os.getcwd(), "dataset", camera_name, "train", "images")
    temp_save_dir = os.path.join(os.getcwd(), "dataset", "temp", "images")
    label_save_dir = os.path.join(os.getcwd(), "dataset", camera_name, "train", "labels")

    if TEST:
        img_save_path = f"./results/{video_name}/"
        os.makedirs(img_save_path, exist_ok=True)

    img_buffer = get_img_buffer(video_path = os.path.join(video_list_path, camera_name, video_name))
    yolo_label_data = get_yolo_label(model = yolo_model, 
                                     buffer = img_buffer)
    

    save_img(video_name = video_name, 
             img_buffer = img_buffer, 
             img_save_dir = temp_save_dir)

    sam_label = SAM_label(img_buffer = img_buffer,
                        img_path = temp_save_dir,
                        label = yolo_label_data,
                        )
    
    for frame_num, img in img_buffer.items():
        sam_final = img.copy()

        if frame_num in sam_label.keys():
            for cls, x1, y1, x2, y2, score in sam_label[frame_num]:
                xyxy = [int(x1), int(y1), int(x2), int(y2)]
                plot_one_box(xyxy, sam_final, label=None, color=(192,154,25), line_thickness=2) # 박스 그리기

    output_path = os.path.join(img_save_path, f"{frame_num}.png")
    cv2.imwrite(output_path, sam_final)

    # save_final_dataset(video_name = video_name,
    #                     date = "test",
    #                     img_buffer = img_buffer, 
    #                     label_buffer = sam_label,
    #                     img_save_dir = img_save_dir, 
    #                     label_save_dir = label_save_dir,
    #                     )
    
cv2.destroyAllWindows()
cmd = "chmod 777 -R ./"
os.system(cmd)