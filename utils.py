import os
from pathlib import Path
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
import time

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))  

def convert_box(box):
    """Convert box from (class, x, y, w, h, score) to (x1, y1, x2, y2, score)."""
    _, x, y, w, h, score = box
    return [x, y, x + w, y + h, score]

# def iou(box1, box2):
#     """Calculate Intersection over Union (IoU) of two boxes."""
#     x1, y1, x2, y2 = box1[:4]
#     xx1, yy1, xx2, yy2 = box2[:4]
    
#     inter_x1 = max(x1, xx1)
#     inter_y1 = max(y1, yy1)
#     inter_x2 = min(x2, xx2)
#     inter_y2 = min(y2, yy2)
    
#     inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
#     box1_area = (x2 - x1) * (y2 - y1)
#     box2_area = (xx2 - xx1) * (yy2 - yy1)
    
#     iou = inter_area / float(box1_area + box2_area - inter_area)
#     return iou

def nms(boxes, iou_threshold=0.5):
    """Perform Non-Maximum Suppression (NMS) on the boxes."""
    # Convert all boxes to (x1, y1, x2, y2, score)
    boxes = [convert_box(box) for box in boxes]
    
    # Sort boxes by score in descending order
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    
    selected_boxes = []
    
    while boxes:
        # Select the box with the highest score
        current_box = boxes.pop(0)
        selected_boxes.append(current_box)
        
        boxes = [box for box in boxes if iou(current_box, box) < iou_threshold]
    
    return selected_boxes


def iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two boxes."""
    x1, y1, x2, y2 = box1[:4]
    xx1, yy1, xx2, yy2 = box2[:4]
    
    inter_x1 = max(x1, xx1)
    inter_y1 = max(y1, yy1)
    inter_x2 = min(x2, xx2)
    inter_y2 = min(y2, yy2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (xx2 - xx1) * (yy2 - yy1)
    
    return inter_area / float(box1_area + box2_area - inter_area)

def merge_boxes(box1, box2):
    """Merge two boxes by averaging their coordinates and taking the maximum score."""
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    # x2 = max(box1[2], box2[2])
    # y2 = max(box1[3], box2[3])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    score = max(box1[4], box2[4])
    return [x1, y1, x2, y2, score]

def nms_test(bbox_list_1, bbox_list_2, iou_threshold=0.8):
    merge_list = []
    unmerge_list = []

    bbox_list_1 = [convert_box(box) for box in bbox_list_1]
    bbox_list_2 = [convert_box(box) for box in bbox_list_2]


    
    # Create a copy to keep track of which boxes have been merged
    merged_1 = [False] * len(bbox_list_1)
    merged_2 = [False] * len(bbox_list_2)
    
    for i, box1 in enumerate(bbox_list_1):
        for j, box2 in enumerate(bbox_list_2):
            if iou(box1, box2) >= iou_threshold:
                merged_box = merge_boxes(box1, box2)
                merge_list.append(merged_box)
                merged_1[i] = True
                merged_2[j] = True
    
    for i, box in enumerate(bbox_list_1):
        if not merged_1[i]:
            unmerge_list.append(box)
    
    for j, box in enumerate(bbox_list_2):
        if not merged_2[j]:
            unmerge_list.append(box)
    
    return merge_list, unmerge_list


def plot_one_box(x, img, color=None, label=None, line_thickness=3, fill_color = False):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    
    if fill_color:
        alpha = 0.3
        overlay = img.copy()
        cv2.rectangle(overlay, c1, c2, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def get_img_buffer(video_path):
    cap = cv2.VideoCapture(video_path)

    ret, img = cap.read()
    frame_num = 1

    img_buffer_ori = []

    while ret:
        ret, img = cap.read()
        frame_num += 1

        if ret and frame_num % 30 == 0:
            img_buffer_ori.append(img)

    return img_buffer_ori


def save_final_dataset(video_name, date, img_buffer, label, img_save_dir, label_save_dir):
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)

    video_name = video_name[:-4]

    for i in range(len(img_buffer)):
        label_txt = ""

        # cv2.imwrite(f"{img_save_dir}/{date}_{video_name}_{str(0) * (4 - len(str(i)))}{i}.jpg", img_buffer[i])
        cv2.imwrite(f"{img_save_dir}/{i}.jpg", img_buffer[i])

        width, heigth = img_buffer[i].shape[1], img_buffer[i].shape[0]

        for cls, x1, y1, x2, y2, score in label[i]:
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
            label_name = f"{label_save_dir}/{i}.txt"

            with open(label_name, "w") as f:
                f.write(label_txt)
        

def get_yolo_label(model, buffer):
    total_label = []
    for i, img in enumerate(buffer):
        label = []

        # if i % 3 == 0 :
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

        total_label.append(label)

    return total_label

def convert_to_od_format(data):  
    """  
    Converts a dictionary with 'bboxes' and 'bboxes_labels' into a dictionary with separate 'bboxes' and 'labels' keys.  
  
    Parameters:  
    - data: The input dictionary with 'bboxes', 'bboxes_labels', 'polygons', and 'polygons_labels' keys.  
  
    Returns:  
    - A dictionary with 'bboxes' and 'labels' keys formatted for object detection results.  
    """  
    # Extract bounding boxes and labels  
    bboxes = data.get('bboxes', [])  
    labels = data.get('bboxes_labels', [])  
      
    # Construct the output format  
    od_results = {  
        'bboxes': bboxes,  
        'labels': labels  
    }  
      
    return od_results  

def get_Florence_label(model, processor, buffer, device):
    task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
    text_input = "person"
    prompt = task_prompt + text_input

    total_label = []

    for img in buffer:
        label_list = []
        pil_img = Image.fromarray(img.astype('uint8'), 'RGB')

        inputs = processor(text=prompt, images=pil_img, return_tensors="pt").to(device)

        generated_ids = model.generate(input_ids=inputs["input_ids"].to(device),
                                        pixel_values=inputs["pixel_values"].to(device),
                                        max_new_tokens=1024,
                                        early_stopping=False,
                                        do_sample=False,
                                        num_beams=3,
                                        )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(generated_text, 
                                                        task=task_prompt, 
                                                        image_size=(pil_img.width, pil_img.height)
                                                        )
        
        # bbox_results  = convert_to_od_format(parsed_answer)

        results = parsed_answer["<CAPTION_TO_PHRASE_GROUNDING>"]

        for i in range(len(results["bboxes"])):

            bbox = [int(results["bboxes"][i][0]), int(results["bboxes"][i][1]), int(results["bboxes"][i][2]), int(results["bboxes"][i][3])]
            label = results["labels"][i]

            if label == "person":
                label_list.append([0] + bbox + [1])


        total_label.append(label_list)


    return total_label

def make_square_bbox(bbox, img, extend_ratio = 1.5):
    label, x1, y1, x2, y2, conf = bbox
    img_height, img_width = img.shape[:2]

    # Calculate width, height, and maximum side length
    width = x2 - x1
    height = y2 - y1

    if (img_width < width * extend_ratio) or (img_height < height * extend_ratio):
        extend_ratio = 3

    max_side = max(width, height) * extend_ratio  # Increase by 1.3 times
    
    # Calculate the center of the bounding box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Calculate new coordinates
    new_x1 = center_x - max_side / 2
    new_y1 = center_y - max_side / 2
    new_x2 = center_x + max_side / 2
    new_y2 = center_y + max_side / 2
    
    # Ensure the new coordinates are within image boundaries
    new_x1 = max(new_x1, 0)
    new_y1 = max(new_y1, 0)
    new_x2 = min(new_x2, img_width)
    new_y2 = min(new_y2, img_height)
    
    return [new_x1, new_y1, new_x2, new_y2, conf, label]

def build_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def check_LLM(model, tokenizer, img_buffer, label, verbose = False):
    final_label = []
    transform = build_transform(input_size=448)

    generation_config_1 = dict(num_beams=1,
                            max_new_tokens=16,
                            do_sample=False,
                            )
    
    generation_config_2 = dict(num_beams=1,
                            max_new_tokens=1,
                            do_sample=False,
                            )
    
    for i in tqdm(range(len(img_buffer)), desc="Processing Image", leave=False):
        img = img_buffer[i]
        bboxes = label[i]
        new_label = []

        # pil_img = Image.fromarray(img).convert('RGB')
        # images = dynamic_preprocess(pil_img, image_size=448, use_thumbnail=True, max_num=6)
        # pixel_values_2 = [transform(image) for image in images]
        # pixel_values_2 = torch.stack(pixel_values_2).to(torch.bfloat16).cuda()

        # # pixel_values = torch.cat((pixel_values_1, pixel_values_2), dim=0)
        # # pixel_values = pixel_values_2
        # question = '<image>\nPlease describe the image in detail.'
        # # question = '<image>\Please describe the image.'

        # res_main, history = model.chat(tokenizer, pixel_values_2, question, generation_config_1, history=None, return_history=True)

        for cls, x1, y1, x2, y2, score in bboxes:
            res_2 = None

            t1 = time.time()
            extend_x1, extend_y1, extend_x2, extend_y2, score, cls = make_square_bbox([cls, x1, y1, x2, y2, score], img, extend_ratio=10)
            cropped_img_extend = img[int(extend_y1) : int(extend_y2), int(extend_x1) : int(extend_x2)]
            pil_img = Image.fromarray(cropped_img_extend).convert('RGB')
            images = dynamic_preprocess(pil_img, image_size=448, use_thumbnail=True, max_num=6)
            pixel_values_2 = [transform(image) for image in images]
            pixel_values_2 = torch.stack(pixel_values_2).to(torch.bfloat16).cuda()

            # pixel_values = torch.cat((pixel_values_1, pixel_values_2), dim=0)
            # pixel_values = pixel_values_2
            # question = '<image>\nPlease describe the image in detail.'
            question = '<image>\Please describe the image.'

            res_main, history = model.chat(tokenizer, pixel_values_2, question, generation_config_1, history=None, return_history=True)


            new_x1, new_y1, new_x2, new_y2, score, cls = make_square_bbox([cls, x1, y1, x2, y2, score], img, extend_ratio = 1.2)
            cropped_img = img[int(new_y1) : int(new_y2), int(new_x1) : int(new_x2)]

            # cropped_img = img[int(y1) : int(y2), int(x1) : int(x2)]

            pil_img = Image.fromarray(cropped_img).convert('RGB')
            images = dynamic_preprocess(pil_img, image_size=448, use_thumbnail=True, max_num=6)
            pixel_values_1 = [transform(image) for image in images]
            pixel_values_1 = torch.stack(pixel_values_1).to(torch.bfloat16).cuda()

            # question = "<image>\nDo you see entire person in this images?"
            # question = "<image>\nDo you see the whole person in the center of this image?"
            # question = "<image>\nDo you see the whole person in the this image?"
            # question = "<image>\nDo you see the person in the this image?"
            question = "<image>\nDo you see person in this image?"


            res = model.chat(tokenizer, pixel_values_1, question, generation_config_2, history=history, return_history=False)
            # res, history = model.chat(tokenizer, pixel_values_1, question, generation_config_2, history=history, return_history=True)

            # res = model.chat(tokenizer, pixel_values_1, question, generation_config_2, return_history=False)

            # question = 'Image-1: <image>\nImage-2: <image>\nReferring to the first image, answer yes or no whether the second image is a person or not.'

            # pixel_values = torch.cat((pixel_values_2, pixel_values_1), dim=0)
            # num_patches_list = [pixel_values_2.size(0), pixel_values_1.size(0)]
            # res, history = model.chat(tokenizer, pixel_values, question, generation_config_2,
            #                                 num_patches_list=num_patches_list,
            #                                 history=None, return_history=True)

            del pixel_values_1, pixel_values_2
            # del pixel_values_1

            res_1 = res.lower()

            # if "yes" in res_1 and "no" not in res_1.split(" "):
            if "yes" in res_1:

                new_label.append([cls, x1, y1, x2, y2, score])
                print("ADD person bbox")

            # else:
            #     cropped_img = img[int(y1) : int(y2), int(x1) : int(x2)]
            #     # cropped_img = img[int(new_y1) : int(new_y2), int(new_x1) : int(new_x2)]

            #     pil_img = Image.fromarray(cropped_img.astype('uint8'), 'RGB')
            #     images = dynamic_preprocess(pil_img, image_size=448, use_thumbnail=True, max_num=6)
            #     pixel_values_3 = [transform(image) for image in images]
            #     pixel_values_3 = torch.stack(pixel_values_3).to(torch.bfloat16).cuda()

            #     # pixel_values_3 = torch.cat((pixel_values_3, pixel_values_2), dim=0)
            #     # question = "<image>\nDo you detect person in this image center? ignore people visible in the background."
                
            #     # question = "<image>\nThis is an image of an assumed person. When you look at it, do you think it's really a person?"
            #     # question = "<image>\nDo you see human in this image?"
            #     question = "<image>\nDo you see person in this image?"
            #     # question = "<image>\nIs there a person in the image?"

            #     res = model.chat(tokenizer, pixel_values_3, question, generation_config_2, history=history, return_history=False)
            #     # res = model.chat(tokenizer, pixel_values_3, question, generation_config_2, return_history=False)

            #     res_2 = res.lower()
            #     del pixel_values_3

            #     # if "yes" in res_2 and "no" not in res_2.split(" "):
            #     if "yes" in res_2:

            #         new_label.append([cls, x1, y1, x2, y2, score])
            #         print("ADD person bbox")

            if verbose:

                test_img = img.copy()
                # cv2.imshow(f"cropped_img_extend", cropped_img_extend)
                cv2.rectangle(test_img, (int(extend_x1), int(extend_y1)), (int(extend_x2), int(extend_y2)), (0,0,255), thickness=2, lineType=cv2.LINE_AA)
                cv2.rectangle(test_img, (int(new_x1), int(new_y1)), (int(new_x2), int(new_y2)), (255,0,0), thickness=2, lineType=cv2.LINE_AA)
                cv2.rectangle(test_img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), thickness=2, lineType=cv2.LINE_AA)
                
                cv2.imshow(f"example", test_img)

                print("-----------------------")
                print(time.time() - t1)
                # print("question 1 : ",res_main)
                print("question 2 : ",res_1)
                if res_2 is not None:
                    print("question 3 : ",res_2)

                # cv2.waitKey(0)
                cv2.waitKey(1)

                # cv2.destroyAllWindows()

        # final_label.append(new_label)
        label[i] = new_label
    

    return label

def get_zero_shot_label(processor, model, device, buffer):
    total_label = []

    for i, img in enumerate(buffer):
        label_list = []

        # text = "person. car. dog. cat. bus. truck. pet."
        text = "person. car. dog. cat. tree."
        # text = "human. car. dog. cat. tree."

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

        # print(results)
        

        for i, boxes in enumerate(results[0]["boxes"].tolist()):
            if results[0]["labels"][i] == "person":
                label_list.append([0] + boxes + [results[0]["scores"][i]])

        label_list = merge_overlapping_boxes(label_list, iou_threshold = 0.5)

        total_label.append(label_list)

    return total_label



def train_model(yolo_weight_path):
    from ultralytics import YOLO
    from pathlib import Path
    from datetime import datetime

    #train data move to main dataset

    # current_time = datetime.now().strftime("%Y%m%d")

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
                            )     

    del model

def create_dataset_list(dataset_path):
    # train_txt = "train.txt"
    # val_txt = "val.txt"
    train_txt = ""
    val_txt = ""

    folder_list = os.listdir(dataset_path)

    for folder_name in folder_list:
        if folder_name not in ["train.txt", "val.txt"]:
            train_img_path = os.path.join(dataset_path, folder_name, "train", "images")
            val_img_path = os.path.join(dataset_path, folder_name, "val", "images")

            train_img_list = os.listdir(train_img_path)

            try:
                val_img_list = os.listdir(val_img_path)
            except:
                val_img_list = []

            if train_img_list:
                train_img_list = sorted(train_img_list)
                for img_name in train_img_list:
                    save_path = os.path.join("./", folder_name, "train", "images")
                    img_path = os.path.join(save_path, img_name)
                    train_txt += f"{img_path}\n"

            if val_img_list:
                val_img_list = sorted(val_img_list)
                for img_name in val_img_list:
                    save_path = os.path.join("./", folder_name, "val", "images")
                    img_path = os.path.join(save_path, img_name)
                    val_txt += f"{img_path}\n"

    if len(train_txt) > 0:
        with open("./dataset/train.txt", "w") as f:
            f.write(train_txt)

    if len(val_txt) > 0:
        with open("./dataset/val.txt", "w") as f:
            f.write(val_txt)
    


def iou_2(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1[1:5]
    x2_min, y2_min, x2_max, y2_max = box2[1:5]

    inter_x_min = max(x1_min, x2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_min = max(y1_min, y2_min)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area
    else:
        return 0.0

def merge_boxes_2(box1, box2):
    x1, y1, w1, h1 = box1[1:5]
    x2, y2, w2, h2 = box2[1:5]

    x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
    x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2

    new_x_min = min(x1_min, x2_min)
    new_y_min = min(y1_min, y2_min)
    new_x_max = max(x1_max, x2_max)
    new_y_max = max(y1_max, y2_max)
    new_conf = max(box1[5], box2[5])

    # return [box1[0], new_x_min, new_y_min, new_x_max, new_y_max, new_conf]
    return [box1[0], new_x_min, new_y_min, new_x_max - new_x_min, new_y_max - new_y_min, new_conf]

def merge_overlapping_boxes(boxes, iou_threshold):
    if len(boxes) == 0:
        return []
    
    merged_boxes = []
    while boxes:
        box = boxes.pop(0)
        to_merge = [box]
        for other_box in boxes[:]:
            if iou_2(box, other_box) >= iou_threshold:
                to_merge.append(other_box)
                boxes.remove(other_box)

        # Merge all to_merge boxes
        while len(to_merge) > 1:
            box1 = to_merge.pop(0)
            box2 = to_merge.pop(0)
            merged_box = merge_boxes_2(box1, box2)
            to_merge.append(merged_box)

        merged_boxes.append(to_merge[0])

    return merged_boxes