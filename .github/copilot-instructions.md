# Copilot instructions for this repository

## Project overview

This project is a Python video-perception pipeline for dashcam footage. The core flow is:

- `perception/video.py` loads RGB frames from a video file, resizes them, and keeps a sparse frame sample using `stride`.
- `perception/detect.py` runs a DETR object detector, filters to road-relevant classes, removes oversized boxes, and applies NMS.
- `perception/tracker.py` performs lightweight multi-object tracking with a custom ByteTrack-like state estimator and keeps persistent track IDs across frames.
- `perception/depth.py` estimates relative depth and computes a `near_score` for each tracked box.
- `perception/pipeline.py` orchestrates detection, tracking, optional depth, and exposes a compact `RunResult` structure for downstream outputs.
- `perception/cli.py` produces annotated output videos and JSON records; `perception/app.py` provides a Gradio UI wrapper around the same logic.

The important mental model is: frames -> detections -> track IDs -> per-track `near` values. Most feature work touches a small set of these files together rather than a single module in isolation.

## Build, test, and lint commands

This repo is configured for `uv` and Python 3.11.

Use the project environment and install dev dependencies:

```bash
uv sync --group dev
```

Run the full test suite:

```bash
uv run pytest
```

Run a single test file:

```bash
uv run pytest tests/test_file.py -q
```

Run a single test function:

```bash
uv run pytest tests/test_file.py::test_name -q
```

Run linting:

```bash
uv run ruff check .
```

Run type checking:

```bash
uv run mypy perception
```

The repo currently defines pytest, Ruff, and mypy in `pyproject.toml`; there are no committed tests yet, so new test coverage should be added under `tests/` and invoked with pytest directly.

## High-level architecture notes

### Detection and filtering

`Detector.__call__` in `perception/detect.py` loads a Hugging Face DETR model, runs inference on a single RGB frame, and returns `(boxes, scores, labels)` where boxes are `xyxy` pixel coordinates. The `filter_detections()` helper trims the result to road-related labels and filters out:

- boxes whose object area is too large,
- boxes whose center is too low in the image,
- duplicate detections via NMS.

This filtering is intentionally part of the detection boundary, not a post-processing step elsewhere.

### Tracking

`ByteTrackLite` in `perception/tracker.py` maintains track state across frames using a constant-velocity Kalman-style model (`Track`) and uses IoU-based assignment for high/low confidence detections. Track IDs are stable across frames and only confirmed tracks that meet `min_hits` are emitted as live results.

If you modify tracker behavior, validate the assumptions in the `Record` object and the frame-by-frame pipeline that consumes it.

### Depth and "near" score

`DepthEstimator` in `perception/depth.py` uses a Hugging Face depth-estimation pipeline and returns a normalized depth map. The `near_score()` helper computes the median depth in the lower half of a detection box, which is then stored on each `Record` as a relative proximity indicator.

The depth output is optional in the pipeline; when disabled, `near` remains `None` and the downstream outputs should still work.

### CLI and UI entrypoints

- `perception/cli.py` is the primary command-line interface for generating annotated MP4 output and JSON records.
- `perception/app.py` wraps the same run logic in Gradio and is intended for interactive use in a browser.

Both entrypoints use the same `run(frames, detect, depth)` orchestration in `perception/pipeline.py`; new behavior should usually be implemented once in the pipeline and then surfaced via the CLI/UI as needed.

## Key conventions specific to this codebase

- Image arrays are kept in RGB for model inputs and drawing, while OpenCV video I/O uses BGR internally. Conversions are explicit (`cv2.cvtColor(..., cv2.COLOR_BGR2RGB)` / `COLOR_RGB2BGR`).
- Bounding boxes are always `xyxy` arrays in pixel coordinates, not normalized coordinates.
- `Record` is the canonical per-object carrier: each item contains `frame`, `id`, `label`, `box`, and `near`, and is the format used by JSON export and visualization.
- The project uses narrow, typed NumPy/array interfaces and avoids large abstractions around the core pipeline. Most logic stays in a few plain Python modules instead of framework-heavy layers.
- Road filter semantics are codified in `ROAD` and `road_ids` in `perception/detect.py`; if detection logic is changed, keep the road-object scope consistent with the rest of the repository.
- Video output functions intentionally account for browser compatibility and `ffmpeg` availability: when `ffmpeg` is present, H.264 output is preferred; otherwise the project falls back to `mp4v` encoding.
- `pyproject.toml` is the canonical source of tooling and dependency groups. If you add a dependency or tool config, update the relevant section there instead of creating a separate local config.

## Working style in this repo

When making changes:

- For behavior changes and bug fixes, follow `.github/agents/tdd.agent.md`: do not inspect or edit `perception/` until a focused test under `tests/` has run and failed for the expected behavior.
- keep the pipeline contract intact (`frames -> detect -> tracker -> Record`),
- prefer updating the core module that owns the behavior instead of patching UI/CLI code separately,
- add new tests under `tests/` for detection/tracking behavior changes, especially around filtering, track IDs, and depth scoring.
