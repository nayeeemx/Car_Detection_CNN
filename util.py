import numpy as np
import cv2


def _ensure_2d(boxes: np.ndarray) -> np.ndarray:
    """Ensure boxes array is (N, 4). If (4,), expand to (1, 4)."""
    if len(boxes.shape) == 1:
        boxes = boxes[np.newaxis, :]
    return boxes


def _area(boxes: np.ndarray) -> np.ndarray:
    """Compute area for boxes shaped (N, 4) with [xmin, ymin, xmax, ymax]."""
    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return widths * heights


def iou(pred_box: np.ndarray, target_box: np.ndarray) -> np.ndarray:
    """
    Intersection-over-Union between a single predicted box and 1 or many target boxes.

    Args:
        pred_box: shape (4,) - [xmin, ymin, xmax, ymax]
        target_box: shape (4,) or (N, 4)

    Returns:
        IoU scores: shape (N,)
    """
    target_box = _ensure_2d(target_box)

    xA = np.maximum(pred_box[0], target_box[:, 0])
    yA = np.maximum(pred_box[1], target_box[:, 1])
    xB = np.minimum(pred_box[2], target_box[:, 2])
    yB = np.minimum(pred_box[3], target_box[:, 3])

    intersection = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)
    boxAArea = _area(np.asarray([pred_box]))[0]
    boxBArea = _area(target_box)

    denom = (boxAArea + boxBArea - intersection)
    # Avoid division by zero by adding a tiny epsilon
    denom = np.where(denom <= 0.0, 1e-12, denom)
    scores = intersection / denom
    return scores


def clip_box_to_image(box: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clip a single box [xmin, ymin, xmax, ymax] to image boundaries."""
    xmin = float(np.clip(box[0], 0, width - 1))
    ymin = float(np.clip(box[1], 0, height - 1))
    xmax = float(np.clip(box[2], 0, width - 1))
    ymax = float(np.clip(box[3], 0, height - 1))
    return np.array([xmin, ymin, xmax, ymax], dtype=np.float32)


def valid_box(box: np.ndarray) -> bool:
    """Return True if a box has positive width and height."""
    return (box[2] > box[0]) and (box[3] > box[1])


def denoise_colored(image_bgr: np.ndarray, h: int = 7, hColor: int = 7, templateWindowSize: int = 7, searchWindowSize: int = 21) -> np.ndarray:
    """Color image denoising using Non-local Means."""
    return cv2.fastNlMeansDenoisingColored(image_bgr, None, h=h, hColor=hColor, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)


def unsharp_mask(image_bgr: np.ndarray, ksize: int = 5, sigma: float = 1.0, amount: float = 1.5) -> np.ndarray:
    """Simple unsharp masking for sharpening."""
    blur = cv2.GaussianBlur(image_bgr, (ksize, ksize), sigma)
    sharpened = cv2.addWeighted(image_bgr, 1 + amount, blur, -amount, 0)
    return sharpened


def clahe_bgr(image_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Apply CLAHE on the L channel in LAB space for contrast enhancement."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def restore_cars_in_image(image_bgr: np.ndarray, rects: list) -> tuple:
    """
    Apply denoise + unsharp + CLAHE to each car crop and paste back.

    Returns:
        restored_bgr: composited image with restored car regions
        crops_before: list of original crops (BGR)
        crops_after: list of restored crops (BGR)
    """
    restored = image_bgr.copy()
    crops_before = []
    crops_after = []

    h, w = image_bgr.shape[:2]
    for rect in rects:
        xmin, ymin, xmax, ymax = [int(v) for v in rect]
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(w - 1, xmax); ymax = min(h - 1, ymax)
        if xmax <= xmin or ymax <= ymin:
            continue
        crop = restored[ymin:ymax, xmin:xmax]
        crops_before.append(crop.copy())

        # Pipeline: denoise -> sharpen -> CLAHE
        step1 = denoise_colored(crop)
        step2 = unsharp_mask(step1)
        step3 = clahe_bgr(step2)

        restored[ymin:ymax, xmin:xmax] = step3
        crops_after.append(step3)

    return restored, crops_before, crops_after