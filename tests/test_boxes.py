import numpy as np

from perception.boxes import iou_matrix, nms


def test_iou_matrix_returns_one_for_identical_boxes_and_zero_for_disjoint_boxes() -> None:
    boxes = np.array(
        [
            [0, 0, 2, 2],
            [3, 3, 5, 5],
        ],
        dtype=float,
    )

    result = iou_matrix(boxes, boxes)

    np.testing.assert_allclose(result, [[1.0, 0.0], [0.0, 1.0]], rtol=1e-12, atol=0)


def test_iou_matrix_calculates_partial_overlap() -> None:
    first = np.array([[0, 0, 2, 2]], dtype=float)
    second = np.array([[1, 1, 3, 3]], dtype=float)

    result = iou_matrix(first, second)

    np.testing.assert_allclose(result, [[1.0 / 7.0]], rtol=1e-12, atol=0)


def test_iou_matrix_handles_empty_inputs_and_preserves_matrix_shape() -> None:
    empty = np.empty((0, 4), dtype=float)
    boxes = np.array([[0, 0, 1, 1], [2, 2, 3, 3]], dtype=float)

    assert iou_matrix(empty, boxes).shape == (0, 2)
    assert iou_matrix(boxes, empty).shape == (2, 0)
    assert iou_matrix(empty, empty).shape == (0, 0)


def test_nms_keeps_highest_scoring_box_and_removes_overlapping_box() -> None:
    boxes = np.array(
        [
            [0, 0, 2, 2],
            [0.25, 0.25, 2.25, 2.25],
            [5, 5, 7, 7],
        ],
        dtype=float,
    )
    scores = np.array([0.9, 0.8, 0.7])

    kept = nms(boxes, scores, 0.5)

    np.testing.assert_array_equal(kept, [0, 2])


def test_nms_handles_empty_boxes() -> None:
    boxes = np.empty((0, 4), dtype=float)
    scores = np.empty((0,), dtype=float)

    kept = nms(boxes, scores, 0.5)

    assert len(kept) == 0
