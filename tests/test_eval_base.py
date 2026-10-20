"""Testy czystych funkcji ewaluatora, bez ładowania modeli."""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture(scope="module")
def eval_base() -> ModuleType:
    """Ładuje skrypt z atrapą opcjonalnego modułu envfile."""
    path = Path(__file__).parents[1] / "scripts" / "eval_base.py"
    spec = spec_from_file_location("eval_base_under_test", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    envfile = ModuleType("envfile")
    envfile.load_env = lambda: None  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"envfile": envfile}):
        spec.loader.exec_module(module)
    return module


def test_read_gt_converts_normalized_boxes_and_polygon_to_pixels(tmp_path, eval_base):
    label = tmp_path / "labels.txt"
    label.write_text("2 0.5 0.5 0.4 0.2\n3 0.1 0.2 0.4 0.2 0.3 0.6\n")

    boxes, classes = eval_base.read_gt(label, w=200, h=100)

    np.testing.assert_allclose(boxes, [[60, 40, 140, 60], [20, 20, 80, 60]])
    np.testing.assert_array_equal(classes, [2, 3])


def test_read_gt_missing_label_returns_empty_arrays(tmp_path, eval_base):
    boxes, classes = eval_base.read_gt(tmp_path / "missing.txt", w=20, h=10)

    assert boxes.shape == (0, 4)
    assert boxes.dtype.kind == "f"
    assert classes.shape == (0,)
    assert classes.dtype.kind in "iu"


@pytest.mark.parametrize(
    ("rgb", "expected"),
    [
        ((255, 0, 0), "Red-Traffic-Light"),
        ((0, 255, 0), "Green-Traffic-Light"),
    ],
)
def test_light_color_classifies_saturated_rgb_crop(eval_base, rgb, expected):
    image = np.full((8, 8, 3), rgb, dtype=np.uint8)

    assert eval_base.light_color(image, np.array([1, 1, 7, 7])) == expected


def test_iou_matrix_returns_pairwise_overlap(eval_base):
    first = np.array([[0, 0, 4, 4], [10, 10, 12, 12]], dtype=float)
    second = np.array([[2, 2, 6, 6]], dtype=float)

    np.testing.assert_allclose(eval_base.iou_matrix(first, second), np.array([[1 / 7], [0.0]]), atol=1e-6)


def test_prf_greedily_matches_highest_scoring_prediction_once(eval_base):
    preds = [
        (
            np.array([[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30]], dtype=float),
            np.array([0.9, 0.8, 0.4]),
            np.array([1, 1, 1]),
        )
    ]
    gts = [(np.array([[0, 0, 10, 10]], dtype=float), np.array([1]))]

    result = eval_base.prf(preds, gts, classes=[1], thr=0.5)[1]

    assert result == {"precision": 0.5, "recall": 1.0, "f1": 2 / 3, "tp": 1, "fp": 1, "fn": 0}
