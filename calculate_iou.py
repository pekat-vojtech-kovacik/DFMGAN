import cv2
import os
import numpy as np
import json
from pathlib import Path
from enum import Enum
from typing import List
from pekat_send import create_instance, send_to_pekat

IMAGES_FOR_EVAL=r"C:\Users\pekat\VojtasBachelors\BackpackDATA\Scaled512\Test2"
GROUND_TRUTH_JSON=r"C:\Users\pekat\PekatVisionProjects\BP Backpack\merged_annotations_TEST2.json"
PORT=7970
SIZE=512,512

class IOU_Type(Enum):
    RECTANGLES = 0
    MASKS = 1


def json_loader(path):
    with open(path, 'rb') as handle:
        data = json.loads(handle.read())

    return data


def read_images_calculate_iou(path_ground_truth, list_evaluated):
    iou_values = []
    
    for eval_path in list_evaluated:
        print(eval_path.name)
        truth_mask = cv2.imread(str(Path(path_ground_truth) / Path(eval_path.stem + ".png")), 0)
        eval_mask = cv2.imread(str(eval_path), 0)

        tmp_iou = calculate_iou_masks(truth_mask, eval_mask)
        iou_values.append(tmp_iou)
    
    average_iou = sum(iou_values) / len(iou_values)
    return average_iou

    
def calculate_iou_masks(truth_mask, eval_mask):
    _, truth_mask = cv2.threshold(truth_mask, 0, 255, cv2.THRESH_BINARY)
    _, eval_mask = cv2.threshold(eval_mask, 0, 255, cv2.THRESH_BINARY)

    truth_mask, eval_mask = truth_mask / 255, eval_mask / 255

    intersection = cv2.bitwise_and(truth_mask, eval_mask)
    union = cv2.bitwise_or(truth_mask, eval_mask)

    iou = np.sum(intersection) / np.sum(union)
    return iou if np.sum(union) != 0 else 0

def mask_from_rectangles(image_rectangles, mask_size, normalized=False):
    mask = np.zeros(mask_size, dtype="uint8")

    for rectangle in image_rectangles:
        # calculate coordinates if they are normalized to [0, 1] - multiply by mask size
        x1 = int(rectangle['x'] * mask_size[0]) if normalized else int(rectangle['x'])
        y1 = int(rectangle['y'] * mask_size[0]) if normalized else int(rectangle['y'])
        x2 = int((rectangle['x'] + rectangle['width']) * mask_size[0]) if normalized else int(rectangle['x'] + rectangle['width'])
        y2 = int((rectangle['y'] + rectangle['height']) * mask_size[0]) if normalized else int(rectangle['y'] + rectangle['height'])
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    return mask

def mask_from_rectangle(rectangle, mask_size, normalized=False):
    mask = np.zeros(mask_size, dtype="uint8")

    if not rectangle:
        return mask
    # calculate coordinates if they are normalized to [0, 1] - multiply by mask size
    x1 = int(rectangle['x'] * mask_size[0]) if normalized else int(rectangle['x'])
    y1 = int(rectangle['y'] * mask_size[0]) if normalized else int(rectangle['y'])
    x2 = int((rectangle['x'] + rectangle['width']) * mask_size[0]) if normalized else int(rectangle['x'] + rectangle['width'])
    y2 = int((rectangle['y'] + rectangle['height']) * mask_size[0]) if normalized else int(rectangle['y'] + rectangle['height'])
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    return mask


def create_masks_by_classes(truth_rectangles, eval_rectangles, mask_size):
    truth_masks = {}
    eval_masks = {}

    for rect in truth_rectangles:
        className = 4 if rect['className'] >= 5 else rect['className']
        tmp_mask = mask_from_rectangle(rect, mask_size, True)
        if className in truth_masks:
            truth_masks[className] = cv2.bitwise_or(truth_masks[className], tmp_mask)
        else:
            truth_masks[className] = tmp_mask

    for rect in eval_rectangles:
        className = 4 if rect['classNames'][0]['id'] >= 5 else rect['classNames'][0]['id']
        tmp_mask = mask_from_rectangle(rect, mask_size)
        if className in eval_masks:
            eval_masks[className] = cv2.bitwise_or(eval_masks[className], tmp_mask)
        else:
           eval_masks[className] = tmp_mask
    
    return truth_masks, eval_masks


def main():
    ground_truth_json = json_loader(GROUND_TRUTH_JSON)
    path = Path(IMAGES_FOR_EVAL)
    p = create_instance(PORT)

    writable_strings = []
    iou = []
    class_ious = {}
    incorrect_class_count = 0

    for image_annotations in ground_truth_json:
        image_path = path / image_annotations['label']
        context = send_to_pekat(p, image_path)
        truth_mask = mask_from_rectangles(image_annotations['rectangles'], SIZE, True)
        eval_mask = mask_from_rectangles(context['detectedRectangles'], SIZE)
        tmp_iou = calculate_iou_masks(truth_mask, eval_mask)
        iou.append(tmp_iou)
        tmp_string = f"Image: {image_annotations['label']}, IoU: {tmp_iou}"

        # truth_mask_per_class, eval_mask_per_class = create_masks_by_classes(image_annotations['rectangles'], context['detectedRectangles'], SIZE)
        
        # tmp_class_ious = {}
        # for i in truth_mask_per_class.keys():
        #     t_mask = truth_mask_per_class.get(i, np.zeros(SIZE))
        #     e_mask = eval_mask_per_class.get(i, np.zeros(SIZE))
        #     tmp_iou = calculate_iou_masks(t_mask, e_mask)
        #     tmp_class_ious[i] = tmp_iou

        # tmp_string = ""
        # for i, iou in tmp_class_ious.items():
        #     tmp_string += f"Image: {image_annotations['label']}, Class: {i} IoU: {iou}\n"
        #     class_ious[i] = class_ious.get(i, [])
        #     class_ious[i].append(tmp_class_ious[i])
        
        writable_strings.append(tmp_string)

        print(tmp_string)



    for i in class_ious.keys():
        average = sum(class_ious[i]) / len(class_ious[i])
        result_string = f"Average IOU in {i}: {average}"
        print(result_string)
        writable_strings.append(result_string)

    result_string = f"Average IOU is: {sum(iou) / len(iou)}"
    print(result_string)
    writable_strings.append(result_string)

    with open(path / 'result_synthetic+test1.txt', 'w') as file:
        for line in writable_strings:
            file.write(line + '\n')

def main_masks():
    GROUND_TRUTH=r"C:\Users\pekat\VojtasBachelors\DFMGAN\evaluation\Bread_512\Ground truth\Masks"
    EVALUATED_VERSIONS=[
        r"C:\Users\pekat\VojtasBachelors\DFMGAN\evaluation\Bread_512\Real_1",
        r"C:\Users\pekat\VojtasBachelors\DFMGAN\evaluation\Bread_512\Real_2",
        r"C:\Users\pekat\VojtasBachelors\DFMGAN\evaluation\Bread_512\Real_3",
        r"C:\Users\pekat\VojtasBachelors\DFMGAN\evaluation\Bread_512\Real_4",
        r"C:\Users\pekat\VojtasBachelors\DFMGAN\evaluation\Bread_512\Real_5"
    ]
    gt = Path(GROUND_TRUTH)
    iou = 0
    for test in EVALUATED_VERSIONS:
        test_path = Path(test).iterdir()
        tmp_iou = read_images_calculate_iou(gt, test_path)
        print(f"IOU: {tmp_iou}")
        iou += tmp_iou
    
    print(iou / len(EVALUATED_VERSIONS))

if __name__ == "__main__":
    #main()
    main_masks()
    # list_ground_truth = sorted(list(Path(GROUND_TRUTH).iterdir()))
    # list_evaluated = []
    
    # for test_version in EVALUATED_VERSIONS:
    #     list_evaluated.extend(list(Path(test_version).iterdir()))

    # list_evaluated = sorted(list_evaluated, key=lambda d: d.name)

    # iou = calculate_iou_masks(list_ground_truth, list_evaluated)

    # print(f"{NAME} dataset IOU: {iou}")