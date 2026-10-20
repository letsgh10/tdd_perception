"""Testy Detector i pick_device na atrapach torcha/transformers (bez wag)."""

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from perception.detect import ROAD, Detector, pick_device


def fake_torch(mps: bool = False, cuda: bool = False) -> ModuleType:
    mod = ModuleType("torch")
    mod.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps))  # type: ignore[attr-defined]
    mod.cuda = SimpleNamespace(is_available=lambda: cuda)  # type: ignore[attr-defined]
    mod.no_grad = nullcontext  # type: ignore[attr-defined]
    mod.tensor = lambda x: x  # type: ignore[attr-defined]
    return mod


class FakeTensor:
    def __init__(self, arr):
        self.arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class FakeInputs(dict):
    def to(self, device):
        self.device = device
        return self


class FakeProcessor:
    result: dict = {}

    @classmethod
    def from_pretrained(cls, model_id):
        inst = cls()
        inst.model_id = model_id
        return inst

    def __call__(self, images, return_tensors):
        return FakeInputs(pixel_values=images)

    def post_process_object_detection(self, out, threshold, target_sizes):
        self.threshold = threshold
        self.target_sizes = target_sizes
        return [type(self).result]


class FakeModel:
    config = SimpleNamespace(id2label={1: "person", 3: "car", 99: "toaster"})
    kwargs: dict = {}

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        inst = cls()
        inst.model_id = model_id
        inst.kwargs = kwargs
        inst.evaluated = False
        return inst

    def to(self, device):
        self.device = device

    def eval(self):
        self.evaluated = True

    def __call__(self, **inputs):
        return SimpleNamespace(inputs=inputs)


@pytest.fixture
def fakes(monkeypatch):
    torch = fake_torch()
    transformers = ModuleType("transformers")
    transformers.DetrImageProcessor = FakeProcessor  # type: ignore[attr-defined]
    transformers.DetrForObjectDetection = FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    FakeProcessor.result = {}
    return torch


@pytest.mark.parametrize(
    ("mps", "cuda", "expected"),
    [(True, True, "mps"), (False, True, "cuda"), (False, False, "cpu")],
)
def test_pick_device_priority(monkeypatch, mps, cuda, expected):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(mps=mps, cuda=cuda))
    assert pick_device() == expected


def test_init_loads_model_in_eval_mode_on_requested_device(fakes):
    det = Detector(device="cpu", model_id="custom/model")
    assert det.device == "cpu"
    assert det.processor.model_id == "custom/model"
    assert det.model.model_id == "custom/model"
    assert det.model.kwargs == {"attn_implementation": "eager"}
    assert det.model.device == "cpu"
    assert det.model.evaluated


def test_init_autopicks_device_when_none(fakes, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda=True))
    assert Detector().device == "cuda"


def test_init_road_ids_only_include_road_labels(fakes):
    det = Detector(device="cpu")
    assert det.id2label == {1: "person", 3: "car", 99: "toaster"}
    assert det.road_ids == {1, 3}
    assert {det.id2label[i] for i in det.road_ids} <= ROAD


def test_init_handles_missing_id2label(fakes, monkeypatch):
    monkeypatch.setattr(FakeModel, "config", SimpleNamespace(id2label=None))
    det = Detector(device="cpu")
    assert det.id2label == {} and det.road_ids == set()


def test_call_returns_filtered_numpy_detections(fakes):
    FakeProcessor.result = {
        "boxes": FakeTensor([[100, 100, 160, 160], [10, 10, 60, 60], [0, 0, 600, 400]]),
        "scores": FakeTensor([0.9, 0.8, 0.95]),
        "labels": FakeTensor([3, 99, 3]),  # toaster i gigantyczny boks odpadają
    }
    det = Detector(device="cpu")
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    boxes, scores, labels = det(img, thr=0.5)

    assert boxes.shape == (1, 4) and boxes[0].tolist() == [100, 100, 160, 160]
    assert scores.tolist() == [0.9] and labels.tolist() == [3]
    assert det.processor.threshold == 0.5
    assert det.processor.target_sizes == [(480, 640)] or list(
        det.processor.target_sizes
    ) == [480, 640]


def test_call_uses_default_threshold_and_handles_no_detections(fakes):
    FakeProcessor.result = {
        "boxes": FakeTensor(np.zeros((0, 4))),
        "scores": FakeTensor(np.zeros(0)),
        "labels": FakeTensor(np.zeros(0, dtype=int)),
    }
    det = Detector(device="cpu")

    boxes, scores, labels = det(np.zeros((120, 160, 3), dtype=np.uint8))

    assert len(boxes) == len(scores) == len(labels) == 0
    assert det.processor.threshold == 0.3
