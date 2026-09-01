"""Panoptic quality and instance metrics.

Paper, Sec. 4.3: all organisms are evaluated as instances of a single class.
Predicted and reference instances are matched one-to-one at an IoU threshold
of 0.5. PQ accounts for both mask and recognition errors, whereas SQ measures
mask overlap among matched instances; PQ = SQ x RQ.

Precision and recall come from the same matching. Recall is particularly
relevant because an unmatched organism is unavailable to downstream analysis.

The over-segmentation ratio ``OSR = N_pred / N_gt`` equals one for equal
counts, above/below one for over- and under-segmentation. Because equal counts
can conceal simultaneous false positives and false negatives, OSR must be read
together with precision and recall.

Evaluation is class-agnostic: every annotated shape is one ground-truth
object, whatever its label. All metrics are computed per test crop and
macro-averaged across the held-out crops, so a crowded crop does not dominate
a sparse one.

Note that RQ is algebraically identical to F1 — reporting both would be
reporting the same number twice.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

METRIC_NAMES = ("pq", "sq", "rq", "precision", "recall", "osr")


def instance_masks(labels: np.ndarray) -> list[np.ndarray]:
    """Split an int label image into a list of boolean masks, skipping 0."""
    return [labels == i for i in np.unique(labels) if i > 0]


def iou_matrix(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]) -> np.ndarray:
    """Pairwise IoU, shape ``(n_pred, n_gt)``."""
    matrix = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    for i, pred in enumerate(pred_masks):
        for j, gt in enumerate(gt_masks):
            union = np.logical_or(pred, gt).sum()
            if union:
                matrix[i, j] = np.logical_and(pred, gt).sum() / union
    return matrix


def match_instances(
    matrix: np.ndarray, iou_threshold: float = 0.5
) -> list[tuple[int, int, float]]:
    """Optimal one-to-one matching, then threshold at ``iou_threshold``.

    Hungarian assignment maximises total IoU across the whole crop. At IoU
    above 0.5 the assignment is provably unique so greedy matching would agree,
    but exactly at 0.5 it need not, and solving it optimally costs nothing at
    these instance counts.
    """
    if matrix.size == 0:
        return []
    rows, cols = linear_sum_assignment(-matrix)
    return [
        (int(r), int(c), float(matrix[r, c]))
        for r, c in zip(rows, cols)
        if matrix[r, c] >= iou_threshold
    ]


def panoptic_quality(
    pred_labels: np.ndarray, gt_masks: list[np.ndarray], iou_threshold: float = 0.5
) -> dict[str, float]:
    """Compute PQ, SQ, RQ, precision, recall and OSR for a single crop."""
    pred_masks = instance_masks(pred_labels)
    n_pred, n_gt = len(pred_masks), len(gt_masks)

    if n_pred == 0 or n_gt == 0:
        return {
            "pq": 0.0,
            "sq": 0.0,
            "rq": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "osr": (n_pred / n_gt) if n_gt else float("nan"),
            "tp": 0,
            "fp": n_pred,
            "fn": n_gt,
        }

    matches = match_instances(iou_matrix(pred_masks, gt_masks), iou_threshold)
    tp = len(matches)
    fp, fn = n_pred - tp, n_gt - tp

    sq = float(np.mean([iou for _, _, iou in matches])) if matches else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) else 0.0

    return {
        "pq": sq * rq,
        "sq": sq,
        "rq": rq,
        "precision": tp / n_pred,
        "recall": tp / n_gt,
        "osr": n_pred / n_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def macro_average(per_crop: list[dict[str, float]]) -> dict[str, float]:
    """Macro-average metrics across crops (unweighted mean, per the paper)."""
    return {
        name: float(np.nanmean([crop[name] for crop in per_crop]))
        for name in METRIC_NAMES
    }


def load_labelme_instances(json_path: Path, height: int, width: int) -> list[np.ndarray]:
    """Rasterise every shape in a LabelMe annotation into a boolean mask.

    Labels are ignored — evaluation is class-agnostic, so an annotated shape
    counts as one organism regardless of what species it was tagged with.
    """
    data = json.loads(Path(json_path).read_text())
    masks = []
    for shape in data.get("shapes", []):
        points = np.asarray(shape["points"], dtype=np.int32)
        canvas = np.zeros((height, width), dtype=np.uint8)
        kind = shape.get("shape_type", "polygon")

        if kind == "circle":
            # LabelMe stores a circle as [centre, a point on the circumference],
            # NOT as two bounding-box corners.
            centre = (int(points[0][0]), int(points[0][1]))
            radius = int(round(float(np.linalg.norm(points[1] - points[0]))))
            cv2.circle(canvas, centre, max(radius, 1), 1, -1)
        elif kind == "rectangle":
            cv2.rectangle(
                canvas,
                (int(points[0][0]), int(points[0][1])),
                (int(points[1][0]), int(points[1][1])),
                1,
                -1,
            )
        else:
            cv2.fillPoly(canvas, [points], 1)

        if canvas.any():
            masks.append(canvas.astype(bool))
    return masks


def load_ground_truth(path: Path, shape: tuple[int, int]) -> list[np.ndarray]:
    """Load ground-truth instances from a LabelMe JSON or an ``.npy`` label map."""
    path = Path(path)
    if path.suffix == ".json":
        return load_labelme_instances(path, *shape)
    return instance_masks(np.load(path))


def format_markdown(results: dict[str, float], name: str = "ours") -> str:
    """Render one result row as a README-ready Markdown table."""
    header = "| Method | PQ | SQ | RQ | Precision | Recall | OSR |"
    rule = "|---|---|---|---|---|---|---|"
    row = f"| {name} | " + " | ".join(
        f"{results[metric]:.3f}" for metric in METRIC_NAMES
    ) + " |"
    return "\n".join([header, rule, row])
