import numpy as np

def iou(box1, box2):
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

def merge_boxes(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1[1:5]
    x2_min, y2_min, x2_max, y2_max = box2[1:5]

    new_x_min = min(x1_min, x2_min)
    new_y_min = min(y1_min, y2_min)
    new_x_max = max(x1_max, x2_max)
    new_y_max = max(y1_max, y2_max)
    new_conf = max(box1[5], box2[5])

    return [box1[0], new_x_min, new_y_min, new_x_max, new_y_max, new_conf]

def merge_overlapping_boxes(boxes, iou_threshold):
    if len(boxes) == 0:
        return []

    merged_boxes = []
    while boxes:
        box = boxes.pop(0)
        to_merge = [box]
        for other_box in boxes[:]:
            print(iou(box, other_box))
            if iou(box, other_box) >= iou_threshold:
                to_merge.append(other_box)
                boxes.remove(other_box)

        # Merge all to_merge boxes
        while len(to_merge) > 1:
            box1 = to_merge.pop(0)
            box2 = to_merge.pop(0)
            merged_box = merge_boxes(box1, box2)
            to_merge.append(merged_box)

        merged_boxes.append(to_merge[0])

    return merged_boxes

# Input list
boxes = [
    [0, 0.66, 0.868, 0.033, 0.123, 1],
    [0, 0.588, 0.887, 0.042, 0.162, 1],
    [0, 0.625, 0.887, 0.041, 0.162, 1],
    [0, 0.588, 0.92, 0.042, 0.152, 1]
]

boxes = np.array(boxes)
boxes[:, 3] += boxes[:, 1]
boxes[:, 4] += boxes[:, 2]
boxes = boxes.tolist()

iou_threshold = 0.6
result = merge_overlapping_boxes(boxes, iou_threshold)

print("Merged Boxes:", result)
