import os
from pathlib import Path
import cv2
import torch
from utils import make_square_bbox, get_yolo_label, get_zero_shot_label, nms, plot_one_box, get_img_buffer, nms_test

import numpy as np

from PIL import Image

from tqdm import tqdm

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

def check_LLM_test_1(model, tokenizer, img_buffer, label, verbose = False):
    final_label = []
    transform = build_transform(input_size=448)

    generation_config_1 = dict(num_beams=1,
                            max_new_tokens=512,
                            do_sample=False,
                            )
    
    generation_config_2 = dict(num_beams=1,
                            max_new_tokens=512,
                            do_sample=False,
                            )
    
    for i, img in enumerate(img_buffer):
        
        bboxes = label[i]

        new_label = []

        for cls, x1, y1, x2, y2, score in bboxes:

            new_x1, new_y1, new_x2, new_y2, score, cls = make_square_bbox([cls, x1, y1, x2, y2, score], img, extend_ratio=10)
            cropped_img_extend = img[int(new_y1) : int(new_y2), int(new_x1) : int(new_x2)]
            pil_img = Image.fromarray(cropped_img_extend).convert('RGB')
            # pil_img = Image.fromarray(img).convert('RGB')

            images = dynamic_preprocess(pil_img, image_size=448, use_thumbnail=True, max_num=6)
            pixel_values_1 = [transform(image) for image in images]
            pixel_values_1 = torch.stack(pixel_values_1).to(torch.bfloat16).cuda()

            # pixel_values = torch.cat((pixel_values_1, pixel_values_2), dim=0)
            # pixel_values = pixel_values_2
            question = '<image>\nPlease describe the image in detail?'

            res, history = model.chat(tokenizer, pixel_values_1, question, generation_config_1, history=None, return_history=True)

            print("-----------------------")
            print("question 1 : ",res)

            # new_x1, new_y1, new_x2, new_y2, score, cls = make_square_bbox([cls, x1, y1, x2, y2, score], img, extend_ratio = 1)
            # cropped_img = img[int(new_y1) : int(new_y2), int(new_x1) : int(new_x2)]
            # pil_img = Image.fromarray(cropped_img.astype('uint8'), 'RGB')
            # pil_img = Image.fromarray(cropped_img).convert('RGB')
            # images = dynamic_preprocess(pil_img, image_size=448, use_thumbnail=True, max_num=6)
            # pixel_values_2 = [transform(image) for image in images]
            # pixel_values_2 = torch.stack(pixel_values_2).to(torch.bfloat16).cuda()

            # pixel_values = torch.cat((pixel_values_1, pixel_values_2), dim=0)

            img_2 = img.copy()
            
            cv2.rectangle(img_2, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), thickness=2, lineType=cv2.LINE_AA)
            cropped_img = img_2[int(new_y1) : int(new_y2), int(new_x1) : int(new_x2)]

            pil_img = Image.fromarray(cropped_img).convert('RGB')
            images = dynamic_preprocess(pil_img, image_size=448, use_thumbnail=True, max_num=6)
            pixel_values_2 = [transform(image) for image in images]
            pixel_values_2 = torch.stack(pixel_values_2).to(torch.bfloat16).cuda()


            cls = "[person, dog, car, cat, tree, etc]"
            # question = "<image>\nDo you see entire person in this images?"
            # question = f"<image>\n Considering the region{x1, y1, x2, y2} of the image , would you classify it as a {cls} category without any doubt? Respond with only ‘yes’ or ‘no’."
            # question = f"<image>\n Would you categorize the this image into the {cls} category without any doubt?" 
            question = f"<Image>\n Categorize the objects in this image that correspond to the red bounding box in the category without question?"

            # question = "<image>\nShort answer in list [person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush, tree]. Name 5 objects that are currently visible in this image."

            # question = "<image>\nThis is an image of an assumed person. When you look at it, do you think it's really a entire person?"

            res = model.chat(tokenizer, pixel_values_2, question, generation_config_2, history=history, return_history=False)

            del pixel_values_1, pixel_values_2
            res = res.lower()
            print("question 2 : ",res)

            # if "yes" in res and "no" not in res.split(" "):
            #     new_label.append([cls, x1, y1, x2, y2, score])
            #     print("ADD person bbox")
            if "person" in res or "people" in res:
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
            #     # question = "<image>\nDo you see person in this image?"
            #     question = "<image>\nDo you see person in the center of this image?"


            #     res = model.chat(tokenizer, pixel_values_3, question, generation_config_2, history=history, return_history=False)

            #     res = res.lower()
            #     print(res)

            #     del pixel_values_3

            #     if "yes" in res and "no" not in res.split(" "):
            #         new_label.append([cls, x1, y1, x2, y2, score])
            #         print("ADD person bbox")

            if verbose:
                cv2.imshow(f"{i}, {x1}", cropped_img)
                cv2.imshow(f"{i+1}, {x1}", cropped_img_extend)

                # key = cv2.waitKey(0)
                # cv2.destroyAllWindows()

        final_label.append(new_label)
    

    return final_label

TEST = True

HOME = Path.home()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

video_list_path = os.path.join(HOME, "workspace", "MS-Auto-Label-Tool", "videos")
video_name_list = os.listdir(video_list_path)
with torch.no_grad():
    for video_name in video_name_list:
        video_name = "camera_7_edit.mp4"
        # video_name = "test"

        yolo_model = YOLO('./weights/yolo/ms-ai2401-finetune.pt')  # load a pretrained model (recommended for training)\

        # zero_shot_ob_model = AutoModelForCausalLM.from_pretrained("./weights/Florence_2_large", trust_remote_code=True).to(device)
        # processor = AutoProcessor.from_pretrained("./weights/Florence_2_large", trust_remote_code=True)
        processor = AutoProcessor.from_pretrained("./weights/grounding-dino-base")
        zero_shot_ob_model = AutoModelForZeroShotObjectDetection.from_pretrained("./weights/grounding-dino-base").to(device)
        # video_name = "people-walking.mp4"
        img_save_dir = os.path.join(HOME, "workspace", "MS-Auto-Label-Tool", "dataset", video_name, "images")


        if TEST:
            img_save_path = f"./results/{video_name}/"
            os.makedirs(img_save_path, exist_ok=True)

        img_buffer = get_img_buffer(video_path = os.path.join(video_list_path, video_name))
        # img_buffer = [cv2.imread("./images/a.png"), cv2.imread("./images/b.png"),cv2.imread("./images/c.png"),cv2.imread("./images/d.png"),cv2.imread("./images/e.png"),cv2.imread("./images/f.png"),cv2.imread("./images/g.png")]
        # img_buffer = [cv2.imread("./Screenshot from 2024-07-10 16-09-51.png")]

        yolo_label_data = get_yolo_label(model = yolo_model, buffer = img_buffer)
        zeroshot_label_data = get_zero_shot_label(model = zero_shot_ob_model, buffer = img_buffer, processor = processor, device = device)


        non_llm_input_bboxes = []
        llm_input_bboxes = []

        for i in range(len(yolo_label_data)):
            all_boxes = yolo_label_data[i] + zeroshot_label_data[i]

            # nms_boxes = nms(all_boxes, iou_threshold=0.9)
            nms_boxes, non_nms_boxes = nms_test(yolo_label_data[i], zeroshot_label_data[i], iou_threshold=0.8)

            
            nms_boxes_original_format = [[0, box[0], box[1], box[2] - box[0], box[3] - box[1], box[4]] for box in nms_boxes]
            non_nms_boxes_original_format = [[0, box[0], box[1], box[2] - box[0], box[3] - box[1], box[4]] for box in non_nms_boxes]

            non_llm_input_bboxes.append(nms_boxes_original_format)
            llm_input_bboxes.append(non_nms_boxes_original_format)

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

        
        # save_img_buffer(img_buffer = img_buffer, save_dir = img_save_dir, video_name = video_name)
        # save_label(img_buffer = img_buffer, save_dir = img_save_dir, video_name = video_name)
        
        break


cmd = "chmod 777 -R ./"
os.system(cmd)