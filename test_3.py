import os
from pathlib import Path
import cv2
import torch
from utils import make_square_bbox, get_yolo_label, get_zero_shot_label, nms, plot_one_box

from PIL import Image

from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoModelForZeroShotObjectDetection


from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

import torchvision.transforms as T

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

def check_LLM_test_1(model, tokenizer, img_buffer, label):
    final_label = []
    generation_config = dict(num_beams=1,
                            max_new_tokens=1024,
                            do_sample=False,
                            )
    
    transform = build_transform(input_size=448)


    for i, img in enumerate(img_buffer):
        
        bboxes = label[i]

        new_label = []

        for cls, x1, y1, x2, y2, score in bboxes:
            new_x1, new_y1, new_x2, new_y2, score, cls = make_square_bbox([cls, x1, y1, x2, y2, score], img)
            cropped_img = img[int(new_y1) : int(new_y2), int(new_x1) : int(new_x2)]
            # pil_img = Image.fromarray(cropped_img.astype('uint8'), 'RGB')
            pil_img = Image.fromarray(cropped_img).convert('RGB')

            images = dynamic_preprocess(pil_img, image_size=448, use_thumbnail=True, max_num=6)
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()


            # question = 'What is in the image?'
            question = "<image>\nDo you detect entire person in this image?"

            res, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True).lower()

            print(res)

            if "yes" in res and "no" not in res.split(" "):
                new_label.append([cls, x1, y1, x2, y2, score])
                print("ADD person bbox")

            else:
                # cropped_img = img[int(y1) : int(y2), int(x1) : int(x2)]
                cropped_img = img[int(new_y1) : int(new_y2), int(new_x1) : int(new_x2)]

                pil_img = Image.fromarray(cropped_img.astype('uint8'), 'RGB')
                images = dynamic_preprocess(pil_img, image_size=448, use_thumbnail=True, max_num=6)
                pixel_values = [transform(image) for image in images]
                pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()


                question = "<image>\nDo you detect person in this image?"

                res, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True).lower()

                print(res)
                if "yes" in res and "no" not in res.split(" "):
                    new_label.append([cls, x1, y1, x2, y2, score])
                    print("ADD person bbox")


            # cv2.imshow(f"{i}, {x1}", cropped_img)

            # key = cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # if key == ord("q"):
            #     final_label.append(new_label)
            #     return final_label 

        final_label.append(new_label)
    

    return final_label

HOME = Path.home()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

video_list_path = os.path.join(HOME, "workspace", "MS-Auto-Label-Tool", "videos")
video_name_list = os.listdir(video_list_path)
with torch.no_grad():
    for video_name in video_name_list:
        video_name = "camera_7_edit.mp4"
        yolo_model = YOLO('./weights/yolo/ms-ai2401-finetune.pt')  # load a pretrained model (recommended for training)\

        # zero_shot_ob_model = AutoModelForCausalLM.from_pretrained("./weights/Florence_2_large", trust_remote_code=True).to(device)
        # processor = AutoProcessor.from_pretrained("./weights/Florence_2_large", trust_remote_code=True)
        processor = AutoProcessor.from_pretrained("./weights/grounding-dino-base")
        zero_shot_ob_model = AutoModelForZeroShotObjectDetection.from_pretrained("./weights/grounding-dino-base").to(device)
        # video_name = "people-walking.mp4"
        img_save_dir = os.path.join(HOME, "workspace", "MS-Auto-Label-Tool", "dataset", video_name, "images")

        # img_buffer = get_img_buffer(video_path = os.path.join(video_list_path, video_name))
        img_buffer = [cv2.imread("./a.png"), cv2.imread("./b.png"),cv2.imread("./c.png"),cv2.imread("./d.png"),cv2.imread("./e.png"),cv2.imread("./f.png"),cv2.imread("./g.png")]


        yolo_label_data = get_yolo_label(model = yolo_model, buffer = img_buffer)
        zeroshot_label_data = get_zero_shot_label(model = zero_shot_ob_model, buffer = img_buffer, processor = processor, device = device)


        total_new_label_list = []

        for i in range(len(yolo_label_data)):
            all_boxes = yolo_label_data[i] + zeroshot_label_data[i]

            nms_boxes = nms(all_boxes, iou_threshold=0.9)
            nms_boxes_original_format = [[0, box[0], box[1], box[2] - box[0], box[3] - box[1], box[4]] for box in nms_boxes]

            total_new_label_list.append(nms_boxes_original_format)

        del yolo_model
        del zero_shot_ob_model
        del processor

        llm_model = AutoModel.from_pretrained("./weights/InternVL2-4B",
                                                torch_dtype=torch.bfloat16,
                                                low_cpu_mem_usage=True,
                                                trust_remote_code=True).eval().cuda()

        tokenizer = AutoTokenizer.from_pretrained("./weights/InternVL2-4B", trust_remote_code=True)

        llm_model.eval()

        LLM_label = check_LLM_test_1(model = llm_model,
                                    tokenizer = tokenizer,
                                    img_buffer = img_buffer,
                                    label = total_new_label_list)

        del llm_model
        del tokenizer


        for i, img in enumerate(img_buffer):
            img_new = img.copy()
            # if len(yolo_label_data):
            #     for cls, x1, y1, x2, y2, score in yolo_label_data[i]:
            #         xyxy = [x1, y1, x2, y2]
            #         plot_one_box(xyxy, img, label=None, color=(255,0,0), line_thickness=2) # 박스 그리기

            # if len(florence_label_data):
            #     for cls, x1, y1, x2, y2, score in florence_label_data[i]:
            #         xyxy = [int(x1), int(y1), int(x2), int(y2)]
            #         plot_one_box(xyxy, img_new, label=None, color=(0,0,255), line_thickness=2) # 박스 그리기

            if len(total_new_label_list):
                for cls, x1, y1, x2, y2, score in total_new_label_list[i]:
                    xyxy = [x1, y1, x2, y2]
                    plot_one_box(xyxy, img, label=None, color=(255,0,0), line_thickness=2) # 박스 그리기

            if len(LLM_label):
                for cls, x1, y1, x2, y2, score in LLM_label[i]:
                    xyxy = [int(x1), int(y1), int(x2), int(y2)]
                    plot_one_box(xyxy, img_new, label=None, color=(0,255,0), line_thickness=2) # 박스 그리기

        
            cv2.imshow("frame", img)
            cv2.imshow("img_new", img_new)

            key = cv2.waitKey(0)

        
        # save_img_buffer(img_buffer = img_buffer, save_dir = img_save_dir, video_name = video_name)
        # save_label(img_buffer = img_buffer, save_dir = img_save_dir, video_name = video_name)
        
        break


cmd = "chmod 777 -R ./"
os.system(cmd)