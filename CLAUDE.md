# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`perception` is a dashcam perception pipeline: object detection (DETR) → multi-object tracking (ByteTrack-lite with a Kalman filter) → optional relative depth estimation (Depth Anything V2), with a CLI and a Gradio UI on top. Source comments and CLI/UI strings are in Polish; code identifiers are in English.

Tests live in `tests/` (one `test_<module>.py` per module, plus `test_models.py` marked `model`). They use injected fakes instead of real models, so they run without weights or network. `tests/fixtures/clip.mp4` is a sample dashcam clip used as fixture input (`sample_video_path` in `conftest.py`) and as the Gradio example. Coverage requires `fail_under = 60`.

## Commands

This project uses `uv` (there is a `uv.lock`). Prefix Python commands with `uv run`, or activate `.venv` first.

```bash
uv sync --all-extras             # install all optional dependency groups (ml, ui, label, train, export) + dev tools
uv sync --extra ml --extra ui    # install only what's needed for running the pipeline + Gradio UI

uv run pytest                    # run tests
uv run pytest tests/test_x.py::test_name   # run a single test
uv run pytest -m model           # run tests marked `model` (needs downloaded HF weights; slow)
uv run pytest -m "not model"     # skip model-dependent tests

make mutation-test              # mutation tests for perception/cli.py (mutmut) -> mutation-report.md

uv run ruff check .              # lint
uv run ruff check --fix .        # lint + autofix
uv run mypy perception           # type-check

uv run perception VIDEO.mp4 --out out.mp4 --json out.json   # run CLI pipeline on a video
uv run perception-ui                                        # launch Gradio UI (localhost:7860)
```

Coverage is configured in `[tool.coverage.*]` in `pyproject.toml`: source is `perception`, `perception/app.py` (the Gradio UI) is omitted, and `fail_under = 60`.

## Mutation testing

`mutmut` (dev dependency) mutates only `perception/cli.py` and runs only `tests/test_cli.py`; see `[tool.mutmut]` in `pyproject.toml`. `scripts/mutation_report.py` turns the results into a Markdown report (score, status table, diffs of survivors with notes on equivalent mutants in `SURVIVOR_NOTES`) and fails below `--min-score`. The CI job `mutation-job` (in `.github/workflows/ci.yaml`, after `quality`) writes the report to the Job Summary with `--min-score 90`. Current state: ~98%, 8 survivors, all equivalent mutants. Generated output goes to `mutants/` (git-ignored). When changing `cli.py`, add tests that kill new survivors or add a note to `SURVIVOR_NOTES`.

## Architecture

The pipeline is deliberately decoupled from any specific model via dependency injection — `pipeline.run()` takes `detect`/`depth` as plain callables, so tests can inject fakes without loading DETR or Depth Anything.

**Data flow** (`perception/pipeline.py: run()`):
1. `video.load_frames()` decodes an mp4 into a list of RGB frames (stride-decimated, resized to a target width) + effective fps.
2. For each frame: `detect(frame) -> (boxes, scores, labels)` (xyxy boxes), then `ByteTrackLite.update(...)` assigns/updates track IDs, then optionally `depth(frame) -> depth_map` (H×W, 0..1, higher = closer).
3. For each confirmed track in the frame, `depth.near_score()` reduces the depth map under the box's lower half to a single "nearness" scalar (feet/bumper area, less background/sky) — `None` when no depth model is used.
4. Output is a flat `RunResult`: `records: list[Record]` (frame, track id, label, box, near) plus per-frame detection counts and timings, and (if depth was used) the depth maps themselves.

**Detection** (`perception/detect.py`): `Detector` wraps a HF `DetrForObjectDetection` model (default `facebook/detr-resnet-50`). `filter_detections()` is the pure-numpy postprocessing step, separated from the model so it's independently testable: restricts to a fixed set of road-relevant classes (`ROAD`), drops oversized boxes (likely the ego vehicle's own hood/mirror) and boxes too low in frame, then runs class-agnostic NMS (`boxes.py: nms`).

**Tracking** (`perception/tracker.py`): `ByteTrackLite` implements the ByteTrack two-stage association idea — match high-confidence detections to existing tracks first (stricter IoU threshold), then try to recover unmatched tracks using low-confidence detections (looser threshold) — via Hungarian assignment (`scipy.optimize.linear_sum_assignment`) on an IoU cost matrix, gated so boxes can only match tracks of the same class. Each `Track` holds its own constant-velocity Kalman filter over state `[cx, cy, w, h, vcx, vcy, vw, vh]`. A track is only returned as "active" once `hits >= min_hits`, and is dropped after `since > max_age` frames without an update.

**Depth** (`perception/depth.py`): `DepthEstimator` wraps a HF depth-estimation pipeline (`depth-anything/Depth-Anything-V2-Small-hf`), bicubic-upsampled back to the input resolution and min-max normalized per frame — this is *relative*, not metric, depth, and not comparable across frames/videos.

**Quality gate** (`perception/gate.py`): `compute_metrics()` reduces a `RunResult` to summary stats (mean detections/frame, unique track count, mean track length, median ms/frame); `check()` compares those against a thresholds dict and returns a list of human-readable violations (empty = pass); `markdown_report()` renders both as a Markdown table — intended for CI-style regression gating on pipeline behavior/performance, and `scripts/evaluate.py` runs it in the CI `model` job against `scripts/thresholds.json`.

**Entry points** (`pyproject.toml: [project.scripts]`):
- `perception` (`cli.py`) — batch CLI: runs the full pipeline on a video file, prints per-frame detections, and can write an annotated video, a track-trails video, a colorized depth video, and/or a JSON dump of records.
- `perception-ui` (`app.py`) — Gradio web UI for interactively uploading a clip and tuning frame sampling/depth/device; caches loaded models per-process (`lru_cache` on `_models`) and re-muxes output video through `ffmpeg` (H.264) when available, falling back to OpenCV's `mp4v` writer otherwise.

Device selection (`detect.py: pick_device()`) auto-picks `mps` > `cuda` > `cpu`; CI/Linux installs pull CPU-only torch via the `pytorch-cpu` explicit `uv` index in `pyproject.toml`.
