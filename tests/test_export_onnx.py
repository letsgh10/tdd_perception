"""Eksport ONNX na malutkim modelu (bez wag DETR-a): kontrakt wejść i wyjść."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from scripts.export_onnx import export  # noqa: E402


class TinyDetr(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 5)

    def forward(self, pixel_values):
        feats = self.proj(pixel_values.mean(dim=(2, 3)))[:, None, :]
        return SimpleNamespace(
            logits=feats.expand(-1, 7, -1), pred_boxes=feats[..., :4].expand(-1, 7, -1)
        )


def test_export_produces_runnable_model(tmp_path):
    path = export(TinyDetr(), tmp_path / "m.onnx", size=(32, 48))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["pixel_values"]
    x = np.zeros((2, 3, 64, 96), dtype=np.float32)  # inny rozmiar niż przy eksporcie
    logits, boxes = sess.run(None, {"pixel_values": x})
    assert logits.shape == (2, 7, 5) and boxes.shape == (2, 7, 4)
