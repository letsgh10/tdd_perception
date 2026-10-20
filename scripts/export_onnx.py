"""Eksport DETR-a (facebook/detr-resnet-50) do ONNX.

  python scripts/export_onnx.py --out detr.onnx

Wejście: pixel_values (N, 3, H, W) po normalizacji DetrImageProcessor, wyjścia:
logits (N, 100, 92) i pred_boxes (N, 100, 4) w formacie cxcywh, znormalizowane.
Postprocessing (progi, filter_detections) zostaje po stronie Pythona.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

from perception.detect import MODEL_ID


def export(model: Any, out: Path, size: tuple[int, int] = (480, 640)) -> Path:
    """Eksportuje model przyjmujący pixel_values; zwraca ścieżkę pliku."""
    import torch

    class Wrapper(torch.nn.Module):
        def __init__(self, inner: Any):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values: Any) -> tuple[Any, Any]:
            res = self.inner(pixel_values=pixel_values)
            return res.logits, res.pred_boxes

    wrapped = Wrapper(model).eval()
    dummy = torch.zeros(1, 3, *size)
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (dummy,),
            str(out),
            input_names=["pixel_values"],
            output_names=["logits", "pred_boxes"],
            dynamic_axes={"pixel_values": {0: "batch", 2: "height", 3: "width"}},
            opset_version=17,
            dynamo=False,
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="detr.onnx")
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()

    from transformers import DetrForObjectDetection

    model = DetrForObjectDetection.from_pretrained(
        args.model_id, attn_implementation="eager"
    )
    out = export(model.eval(), Path(args.out), (args.height, args.width))
    print(f"Zapisano {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
