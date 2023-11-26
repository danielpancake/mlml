import numpy as np


def jaccard_coefficient(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    """
    Calculates the Jaccard coefficient (IoU) between two bounding boxes.
    """
    x0 = max(bbox_a[0], bbox_b[0])
    y0 = max(bbox_a[1], bbox_b[1])
    x1 = min(bbox_a[2], bbox_b[2])
    y1 = min(bbox_a[3], bbox_b[3])

    intersection = float(max(0, x1 - x0) * max(0, y1 - y0))
    if intersection == 0:
        return 0

    bbox_a_ww = bbox_a[2] - bbox_a[0]
    bbox_a_hh = bbox_a[3] - bbox_a[1]
    bbox_b_ww = bbox_b[2] - bbox_b[0]
    bbox_b_hh = bbox_b[3] - bbox_b[1]

    union = float(bbox_a_ww * bbox_a_hh + bbox_b_ww * bbox_b_hh - intersection)

    return intersection / union


def argmax_equal(x: np.ndarray, y: np.ndarray) -> bool:
    assert x.shape == y.shape, "x and y must have the same shape"
    return np.argmax(x) == np.argmax(y)
