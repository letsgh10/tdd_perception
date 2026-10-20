"""Geometria skrzynek (x1, y1, x2, y2). Czysty NumPy, bez modeli."""

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    def area(r: np.ndarray) -> np.ndarray:
        return (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])

    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Zwraca indeksy skrzynek do zachowania (od najwyższego score), jak torchvision.ops.nms."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while len(order):
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        ious = iou_matrix(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return np.array(keep, dtype=int)
