import time
import copy
from dataclasses import dataclass
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import alexnet
import torchvision.transforms as transforms

import util

#
# Hardcoded configuration for detection. These values are used regardless of
# external parameters passed by callers to ensure a deterministic setup.
#
@dataclass(frozen=True)
class DetectorConfig:
    # Paths and model
    model_path: str = './models/best_linear_svm_alexnet_car.pth'

    # Preprocessing
    input_size: tuple = (227, 227)
    normalize_mean: tuple = (0.5, 0.5, 0.5)
    normalize_std: tuple = (0.5, 0.5, 0.5)

    # Selective search
    selective_search_mode: str = 'Fast'  # One of: 'Single', 'Fast', 'Quality'

    # Classification threshold for positive car class (index 1)
    svm_score_threshold: float = 0.70

    # Non-maximum suppression IoU threshold
    nms_iou_threshold: float = 0.10

    # Drawing settings
    box_color_bgr: tuple = (255, 0, 0)
    box_thickness: int = 2
    text: str = 'CAR'
    text_font: int = cv2.FONT_HERSHEY_SIMPLEX
    text_scale: float = 0.5
    text_color_bgr: tuple = (255, 255, 255)
    text_thickness: int = 1


CFG = DetectorConfig()

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')

def get_transform():
    # Note: Keep a deterministic transform for inference. Horizontal flip is
    # typically an augmentation for training; omitting it stabilizes outputs.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(CFG.input_size),
        transforms.ToTensor(),
        transforms.Normalize(CFG.normalize_mean, CFG.normalize_std)
    ])
    return transform

def get_model(device=None):
    model = alexnet()
    num_classes = 2
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    model.load_state_dict(
        torch.load(
            CFG.model_path,
            map_location=device
        )
    )
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    if device: model = model.to(device)
    return model

def draw_box_with_text(img, rect_list, score_list):
    for i in range(len(rect_list)):
        xmin, ymin, xmax, ymax = rect_list[i]
        score = score_list[i]

        cv2.rectangle(
            img,
            (xmin, ymin),
            (xmax, ymax),
            color=CFG.box_color_bgr,
            thickness=CFG.box_thickness
        )
        cv2.putText(
            img,
            f"{CFG.text}: {score:.3f}",
            (xmin, ymin),
            CFG.text_font,
            CFG.text_scale,
            CFG.text_color_bgr,
            CFG.text_thickness
        )

def nms(rect_list, score_list):
    nms_rects = list()
    nms_scores = list()

    rect_array = np.array(rect_list)
    score_array = np.array(score_list)

    idxs = np.argsort(score_array)[::-1]
    rect_array = rect_array[idxs]
    score_array = score_array[idxs]

    thresh = CFG.nms_iou_threshold
    while len(score_array) > 0:
        nms_rects.append(rect_array[0])
        nms_scores.append(score_array[0])
        rect_array = rect_array[1:]
        score_array = score_array[1:]

        length = len(score_array)
        if (length <= 0): break

        iou_scores = util.iou(
            np.array(nms_rects[-1]),
            rect_array
        )
        idxs = np.where(iou_scores < thresh)[0]
        rect_array = rect_array[idxs]
        score_array = score_array[idxs]

    return nms_rects, nms_scores

def preds(img, svm_thresh, rects, callback=None):
    # Allow thresholds from UI and allow rects=None to trigger internal selective search
    score_list = list()
    positive_list = list()
    device = get_device()
    transform = get_transform()
    model = get_model(device=device)
    dst = copy.deepcopy(img)

    # If rects are not provided, compute them using the hardcoded selective search mode
    if rects is None:
        rects = get_ss(img, CFG.selective_search_mode)

    nRects = len(rects)

    start = time.time()
    percent = 0
    for i, rect in enumerate(rects):

        if (percent < (i * 10 // max(1, nRects))):
            percent = i / max(1, nRects)
            if (callback is not None): callback(percent)

        xmin, ymin, xmax, ymax = rect
        rect_img = img[ymin:ymax, xmin:xmax]

        rect_transform = transform(rect_img).to(device)
        output = model(rect_transform.unsqueeze(0))[0]

        if torch.argmax(output).item() == 1:
            probs = torch.softmax(output, dim=0).cpu().numpy()

            # Use provided threshold from caller, fallback to CFG if missing
            threshold = svm_thresh if svm_thresh is not None else CFG.svm_score_threshold
            if probs[1] >= threshold:
                score_list.append(probs[1])
                positive_list.append(rect)
    end = time.time()
    print('detect time: %d s' % (end - start))

    nms_rects, nms_scores = nms(positive_list, score_list)
    draw_box_with_text(dst, nms_rects, nms_scores)

    return dst


def detect_boxes(img, svm_thresh=None, rects=None, callback=None):
    """
    Return detected car boxes and scores after NMS without drawing.

    Args:
        img: numpy array (H, W, 3)
        svm_thresh: float or None (falls back to CFG)
        rects: optional selective search rects; if None, computed internally
        callback: optional progress callback in [0,1]

    Returns:
        (nms_rects, nms_scores): lists with shapes (N, 4) and (N,)
    """
    score_list = list()
    positive_list = list()
    device = get_device()
    transform = get_transform()
    model = get_model(device=device)

    if rects is None:
        rects = get_ss(img, CFG.selective_search_mode)

    nRects = len(rects)
    start = time.time()
    percent = 0
    for i, rect in enumerate(rects):
        if (percent < (i * 10 // max(1, nRects))):
            percent = i / max(1, nRects)
            if (callback is not None): callback(percent)

        xmin, ymin, xmax, ymax = rect
        rect_img = img[ymin:ymax, xmin:xmax]
        rect_transform = transform(rect_img).to(device)
        output = model(rect_transform.unsqueeze(0))[0]

        if torch.argmax(output).item() == 1:
            probs = torch.softmax(output, dim=0).cpu().numpy()
            threshold = svm_thresh if svm_thresh is not None else CFG.svm_score_threshold
            if probs[1] >= threshold:
                score_list.append(probs[1])
                positive_list.append(rect)

    end = time.time()
    print('detect time: %d s' % (end - start))

    nms_rects, nms_scores = nms(positive_list, score_list)
    return nms_rects, nms_scores

def _ss_backend_create():
    # Local selective search backend using OpenCV ximgproc, avoiding external package name conflicts
    if not hasattr(cv2, 'ximgproc') or not hasattr(cv2.ximgproc, 'segmentation'):
        raise RuntimeError('OpenCV contrib (ximgproc) is required for selective search')
    return cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()


def _ss_backend_config(gs, img, strategy='q'):
    gs.setBaseImage(img)
    if strategy == 's':
        gs.switchToSingleStrategy()
    elif strategy == 'f':
        gs.switchToSelectiveSearchFast()
    elif strategy == 'q':
        gs.switchToSelectiveSearchQuality()
    else:
        gs.switchToSelectiveSearchQuality()


def _ss_backend_rects(gs):
    rects = gs.process()
    rects[:, 2] += rects[:, 0]
    rects[:, 3] += rects[:, 1]
    return rects


def get_ss(img, ss_mode):
    # Use provided mode if given; otherwise fall back to CFG
    gs = _ss_backend_create()
    chosen_mode = ss_mode if ss_mode is not None else CFG.selective_search_mode
    strategy = ''
    if (chosen_mode == "Single"): strategy = 's'
    elif (chosen_mode == "Fast"): strategy = 'f'
    elif (chosen_mode == "Quality"): strategy = 'q'
    _ss_backend_config(gs, img, strategy=strategy)
    rects = _ss_backend_rects(gs)
    return rects


def detect_cars(img, callback=None):
    """
    Convenience pipeline with hardcoded configuration:
      1) Run selective search with CFG.selective_search_mode
      2) Classify regions and draw boxes using CFG.svm_score_threshold and NMS

    This function can be used directly without passing thresholds or rects.
    """
    rects = get_ss(img, CFG.selective_search_mode)
    return preds(img, CFG.svm_score_threshold, rects, callback=callback)