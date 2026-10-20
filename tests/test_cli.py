"""Testy CLI na atrapach modeli: bez wag i bez sieci."""

import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

import perception.depth
import perception.detect
from perception import cli
from perception.pipeline import Record, RunResult

ID2LABEL = {3: "car"}


class FakeDetector:
    """Jeden samochód jadący w prawo; zapamiętuje argument `device`."""

    last_device: str | None = "unset"

    def __init__(self, device=None):
        FakeDetector.last_device = device
        self.device = device or "fake"
        self.id2label = ID2LABEL
        self.calls = 0

    def __call__(self, img):
        x = 10.0 + 5 * self.calls
        self.calls += 1
        return np.array([[x, 100, x + 80, 160.0]]), np.array([0.9]), np.array([3])


class FakeDepth:
    last_device: str | None = "unset"
    instances = 0

    def __init__(self, device=None):
        FakeDepth.last_device = device
        FakeDepth.instances += 1

    def __call__(self, img):
        return np.full(img.shape[:2], 0.7)


@pytest.fixture
def fake_models(monkeypatch):
    monkeypatch.setattr(perception.detect, "Detector", FakeDetector)
    monkeypatch.setattr(perception.depth, "DepthEstimator", FakeDepth)


def _argv(video: Path, *extra: str) -> list[str]:
    return [str(video), "--max-frames", "8", "--stride", "2", "--width", "320", *extra]


def _rec(id_=1, label=3, box=(10, 20, 60, 80), near=None, frame=0) -> Record:
    return Record(frame, id_, label, np.array(box, float), near)


def _n_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


# --- main: ścieżki błędów ---------------------------------------------------


def test_missing_video_returns_2(capsys, tmp_path):
    assert cli.main([str(tmp_path / "nie_ma.mp4")]) == 2
    assert "Brak pliku" in capsys.readouterr().err


def test_unreadable_video_returns_2(capsys, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"to nie jest wideo")
    assert cli.main([str(bad)]) == 2
    assert "Nie udało się wczytać klatek" in capsys.readouterr().err


def test_out_depth_with_no_depth_is_argparse_error(sample_video_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(_argv(sample_video_path, "--no-depth", "--out-depth", "d.mp4"))
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert err.endswith("error: --out-depth wymaga głębi (usuń --no-depth)\n")


# --- main: ścieżka szczęśliwa -------------------------------------------------


def test_main_prints_per_frame_and_summary(fake_models, sample_video_path, capsys):
    assert cli.main(_argv(sample_video_path, "--no-depth")) == 0
    cap = capsys.readouterr()
    lines = cap.out.splitlines()
    frame_lines = [ln for ln in lines if ln.startswith("klatka ")]
    assert len(frame_lines) == 8
    assert frame_lines[0] == "klatka    0: -"  # ścieżka jeszcze niepotwierdzona
    assert frame_lines[-1] == "klatka    7: #1 car"
    assert "Unikalne tory: 1, mediana" in cap.out
    assert "ms/klatkę (fake)" in cap.out
    assert "Klatek: 8, ładuję modele..." in cap.err


def test_main_passes_device_to_detector(fake_models, sample_video_path):
    cli.main(_argv(sample_video_path, "--no-depth", "--device", "cpu"))
    assert FakeDetector.last_device == "cpu"
    cli.main(_argv(sample_video_path, "--no-depth"))
    assert FakeDetector.last_device is None


def test_no_depth_does_not_load_depth_model(fake_models, sample_video_path):
    before = FakeDepth.instances
    cli.main(_argv(sample_video_path, "--no-depth"))
    assert FakeDepth.instances == before
    cli.main(_argv(sample_video_path))
    assert FakeDepth.instances == before + 1


def test_main_passes_device_to_depth(fake_models, sample_video_path):
    cli.main(_argv(sample_video_path, "--device", "cuda"))
    assert FakeDetector.last_device == "cuda"
    assert FakeDepth.last_device == "cuda"


def test_main_writes_json(fake_models, sample_video_path, tmp_path, capsys):
    out = tmp_path / "r.json"
    assert cli.main(_argv(sample_video_path, "--json", str(out))) == 0
    text = out.read_text()
    assert text.endswith("]\n")
    assert text.startswith("[\n  {")  # indent=2
    rows = json.loads(text)
    assert rows
    assert set(rows[0]) == {"frame", "id", "label", "box", "near"}
    assert rows[0]["label"] == "car"
    assert rows[0]["id"] == 1
    assert rows[0]["box"] == [rows[0]["box"][0], 100.0, rows[0]["box"][0] + 80, 160.0]
    assert rows[0]["near"] == pytest.approx(0.7, abs=1e-3)
    assert f"JSON: {out}" in capsys.readouterr().out


def test_json_near_is_null_without_depth(fake_models, sample_video_path, tmp_path):
    out = tmp_path / "r.json"
    cli.main(_argv(sample_video_path, "--no-depth", "--json", str(out)))
    assert all(r["near"] is None for r in json.loads(out.read_text()))


def test_json_rounds_box_and_near(monkeypatch, sample_video_path, tmp_path):
    class Odd(FakeDetector):
        def __call__(self, img):
            return (
                np.array([[10.123456, 100.987654, 90.55555, 160.11111]]),
                np.array([0.9]),
                np.array([3]),
            )

    class OddDepth(FakeDepth):
        def __call__(self, img):
            return np.full(img.shape[:2], 0.123456789)

    monkeypatch.setattr(perception.detect, "Detector", Odd)
    monkeypatch.setattr(perception.depth, "DepthEstimator", OddDepth)
    out = tmp_path / "r.json"
    cli.main(_argv(sample_video_path, "--json", str(out)))
    row = json.loads(out.read_text())[0]
    assert all(round(v, 1) == v for v in row["box"])
    assert row["near"] == round(row["near"], 4)


def test_json_unknown_label_falls_back_to_number(
    monkeypatch, sample_video_path, tmp_path
):
    class NoNames(FakeDetector):
        def __init__(self, device=None):
            super().__init__(device)
            self.id2label = {}

    monkeypatch.setattr(perception.detect, "Detector", NoNames)
    out = tmp_path / "r.json"
    cli.main(_argv(sample_video_path, "--no-depth", "--json", str(out)))
    assert json.loads(out.read_text())[0]["label"] == "3"


def test_main_writes_all_videos(fake_models, sample_video_path, tmp_path, capsys):
    o, t, d = tmp_path / "o.mp4", tmp_path / "t.mp4", tmp_path / "d.mp4"
    argv = _argv(
        sample_video_path,
        "--out",
        str(o),
        "--out-tracks",
        str(t),
        "--out-depth",
        str(d),
    )
    assert cli.main(argv) == 0
    for p in (o, t, d):
        assert p.stat().st_size > 0
        assert _n_frames(p) == 8
    out = capsys.readouterr().out
    for p in (o, t, d):
        assert f"Wideo: {p}" in out


def test_out_tracks_only(fake_models, sample_video_path, tmp_path):
    t = tmp_path / "t.mp4"
    assert cli.main(_argv(sample_video_path, "--no-depth", "--out-tracks", str(t))) == 0
    assert t.exists()


def test_bad_out_extension_returns_2(fake_models, sample_video_path, tmp_path, capsys):
    assert (
        cli.main(
            _argv(sample_video_path, "--no-depth", "--out", str(tmp_path / "o.avi"))
        )
        == 2
    )
    assert "Nie udało się zapisać wideo" in capsys.readouterr().err


def test_bad_out_tracks_extension_returns_2(
    fake_models, sample_video_path, tmp_path, capsys
):
    argv = _argv(
        sample_video_path, "--no-depth", "--out-tracks", str(tmp_path / "t.mkv")
    )
    assert cli.main(argv) == 2
    assert "Nie udało się zapisać wideo" in capsys.readouterr().err


def test_unwritable_out_depth_returns_2(
    fake_models, sample_video_path, tmp_path, capsys
):
    argv = _argv(sample_video_path, "--out-depth", str(tmp_path / "brak" / "d.mp4"))
    assert cli.main(argv) == 2
    assert "Nie udało się zapisać wideo" in capsys.readouterr().err


# --- draw -------------------------------------------------------------------


def test_draw_returns_copy_and_marks_box():
    frame = np.zeros((100, 100, 3), np.uint8)
    vis = cli.draw(frame, [_rec()], ID2LABEL)
    assert vis is not frame
    assert not frame.any()
    assert vis[20, 30].any()  # górna krawędź skrzynki
    assert not vis[50, 30].any()  # wnętrze skrzynki nie jest wypełnione


def test_draw_without_records_is_unchanged():
    frame = np.full((50, 50, 3), 7, np.uint8)
    assert np.array_equal(cli.draw(frame, [], ID2LABEL), frame)


def test_draw_label_text_changes_pixels():
    frame = np.zeros((100, 200, 3), np.uint8)
    with_near = cli.draw(frame, [_rec(near=0.5)], ID2LABEL)
    without = cli.draw(frame, [_rec(near=None)], ID2LABEL)
    assert not np.array_equal(with_near, without)


def test_draw_label_text_depends_on_id2label():
    frame = np.zeros((100, 200, 3), np.uint8)
    a = cli.draw(frame, [_rec()], {3: "car"})
    b = cli.draw(frame, [_rec()], {})
    assert not np.array_equal(a, b)


def test_draw_text_clamped_at_top_edge():
    frame = np.zeros((100, 200, 3), np.uint8)
    vis = cli.draw(frame, [_rec(box=(10, 0, 60, 50))], ID2LABEL)
    assert vis[:12].any()  # tekst narysowany w y=12, nie poza obrazem


def test_draw_uses_color_of_track():
    frame = np.zeros((100, 100, 3), np.uint8)
    vis = cli.draw(frame, [_rec(id_=5)], ID2LABEL)
    assert tuple(int(c) for c in vis[20, 30]) == cli._color(5)


def _reference_draw(frame, recs, id2label):
    vis = frame.copy()
    for r in recs:
        x1, y1, x2, y2 = (int(v) for v in r.box)
        col = cli._color(r.id)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        text = f"#{r.id} {id2label.get(r.label, r.label)}"
        if r.near is not None:
            text += f" {r.near:.2f}"
        cv2.putText(
            vis, text, (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2
        )
    return vis


@pytest.mark.parametrize("near", [None, 0.456])
@pytest.mark.parametrize("box", [(40, 50, 120, 90), (10, 0, 60, 50), (10, 14, 60, 50)])
@pytest.mark.parametrize("id2label", [ID2LABEL, {}])
def test_draw_matches_reference_rendering(near, box, id2label):
    frame = np.zeros((120, 200, 3), np.uint8)
    recs = [_rec(box=box, near=near), _rec(id_=2, box=(150, 60, 190, 110))]
    assert np.array_equal(
        cli.draw(frame, recs, id2label), _reference_draw(frame, recs, id2label)
    )


def test_draw_tracks_matches_reference_polyline():
    frame = np.zeros((100, 200, 3), np.uint8)
    boxes = [(0, 0, 20, 40), (60, 0, 80, 60), (100, 0, 120, 30), (150, 0, 190, 20)]
    by_frame = {i: [_rec(box=b, frame=i)] for i, b in enumerate(boxes)}
    expected = _reference_draw(frame, by_frame[3], ID2LABEL)
    pts = [(int((b[0] + b[2]) / 2), int(b[3])) for b in boxes]
    cv2.polylines(expected, [np.array(pts, dtype=np.int32)], False, cli._color(1), 2)
    assert np.array_equal(cli.draw_tracks(frame, by_frame, 3, ID2LABEL), expected)


# --- _color -----------------------------------------------------------------


def test_color_deterministic_in_range_and_distinct():
    assert cli._color(1) == cli._color(1)
    assert cli._color(1) != cli._color(2)
    for i in range(50):
        c = cli._color(i)
        assert len(c) == 3 and all(60 <= v < 255 for v in c)
        assert all(type(v) is int for v in c)
    values = {v for i in range(300) for v in cli._color(i)}
    assert min(values) == 60 and max(values) == 254  # [60, 255)


# --- draw_tracks ------------------------------------------------------------


def test_draw_tracks_draws_trail_through_bottom_centers():
    frame = np.zeros((100, 200, 3), np.uint8)
    by_frame = {
        0: [_rec(box=(0, 0, 20, 40), frame=0)],
        1: [_rec(box=(100, 0, 120, 40), frame=1)],
    }
    vis = cli.draw_tracks(frame, by_frame, 1, ID2LABEL)
    # ślad: (10, 40) -> (110, 40); punkt pośrodku leży tylko na śladzie
    assert tuple(int(c) for c in vis[40, 60]) == cli._color(1)


def test_draw_tracks_ignores_future_frames():
    frame = np.zeros((100, 200, 3), np.uint8)
    by_frame = {
        0: [_rec(box=(0, 0, 20, 40))],
        1: [_rec(box=(100, 0, 120, 40))],
    }
    vis = cli.draw_tracks(frame, by_frame, 0, ID2LABEL)
    assert not vis[40, 60].any()
    assert np.array_equal(vis, cli.draw(frame, by_frame[0], ID2LABEL))


def test_draw_tracks_single_point_has_no_trail():
    frame = np.zeros((100, 200, 3), np.uint8)
    by_frame = {0: [_rec(box=(0, 0, 20, 40))]}
    vis = cli.draw_tracks(frame, by_frame, 0, ID2LABEL)
    assert np.array_equal(vis, cli.draw(frame, by_frame[0], ID2LABEL))


def test_draw_tracks_separate_trail_per_id():
    frame = np.zeros((100, 200, 3), np.uint8)
    by_frame = {
        0: [_rec(id_=1, box=(0, 0, 20, 40)), _rec(id_=2, box=(0, 50, 20, 90))],
        1: [_rec(id_=1, box=(100, 0, 120, 40)), _rec(id_=2, box=(100, 50, 120, 90))],
    }
    vis = cli.draw_tracks(frame, by_frame, 1, ID2LABEL)
    assert tuple(int(c) for c in vis[40, 60]) == cli._color(1)
    assert tuple(int(c) for c in vis[90, 60]) == cli._color(2)


# --- depth_frame ------------------------------------------------------------


def test_depth_frame_shape_dtype_and_ordering():
    d = np.tile(np.arange(256, dtype=np.uint8), (4, 1))
    out = cli.depth_frame(d)
    assert out.shape == (4, 256, 3) and out.dtype == np.uint8
    # INFERNO: 0 -> prawie czarny, 255 -> jasnożółty; po konwersji do RGB R > B
    assert out[0, 0].sum() < out[0, 255].sum()
    r, _, b = (int(v) for v in out[0, 255])
    assert r > b
    expected = cv2.cvtColor(
        cv2.applyColorMap(d, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB
    )
    assert np.array_equal(out, expected)


# --- write_video ------------------------------------------------------------


def _frames(n=3):
    return [np.full((32, 48, 3), 40 * i, np.uint8) for i in range(n)]


def test_write_video_roundtrip(tmp_path):
    p = cli.write_video(tmp_path / "v.mp4", _frames(), 10.0)
    assert p == tmp_path / "v.mp4"
    cap = cv2.VideoCapture(str(p))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 48
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 32
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(10.0)
    cap.release()


def test_write_video_adds_mp4_suffix(tmp_path):
    p = cli.write_video(tmp_path / "bez_rozszerzenia", _frames(), 5.0)
    assert p.suffix == ".mp4" and p.exists()


def test_write_video_accepts_uppercase_suffix(tmp_path):
    p = cli.write_video(tmp_path / "V.MP4", _frames(), 5.0)
    assert p.exists()


def test_write_video_rejects_other_suffix(tmp_path):
    with pytest.raises(
        ValueError, match="^Plik wyjściowy musi mieć rozszerzenie .mp4$"
    ):
        cli.write_video(tmp_path / "v.avi", _frames(), 5.0)
    assert not (tmp_path / "v.avi").exists()


def test_write_video_unopenable_path_raises(tmp_path):
    with pytest.raises(RuntimeError, match="^Nie można otworzyć pliku wyjściowego: "):
        cli.write_video(tmp_path / "brak" / "v.mp4", _frames(), 5.0)


def test_write_video_converts_rgb_to_bgr(tmp_path):
    red = [np.zeros((32, 48, 3), np.uint8) for _ in range(3)]
    for f in red:
        f[..., 0] = 255  # RGB: czerwony
    p = cli.write_video(tmp_path / "r.mp4", red, 5.0)
    cap = cv2.VideoCapture(str(p))
    ok, bgr = cap.read()
    cap.release()
    assert ok
    b, _, r = bgr[16, 24]
    assert r > 150 and b < 100


def test_write_video_single_frame(tmp_path):
    p = cli.write_video(tmp_path / "one.mp4", _frames(1), 5.0)
    assert _n_frames(p) == 1


# --- CLI: parametry, pomoc, składanie wyjścia --------------------------------


def test_help_lists_options_and_descriptions(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = " ".join(capsys.readouterr().out.split())
    assert out.startswith("usage: perception ")
    assert "CLI: uruchamia potok" in out
    for text in (
        "plik wideo (dashcam)",
        "zapisz wideo z naniesionymi skrzynkami (mp4)",
        "zapisz wideo z torami obiektów (mp4)",
        "zapisz wideo z mapą głębi (mp4)",
        "zapisz rekordy (klatka, id, klasa, box, near) do JSON",
        "bierz co N-tą klatkę",
        "szerokość po przeskalowaniu",
        "pomiń estymację głębi (szybciej)",
        "cpu/mps/cuda; domyślnie auto",
    ):
        # cały tekst jako osobne słowa (wyklucza dopiski typu "XX...XX")
        assert re.search(rf"(?<!\S){re.escape(text)}(?!\S)", out), text


def test_defaults_passed_to_load_frames(monkeypatch, sample_video_path):
    seen = {}

    def spy(*args):
        seen["args"] = args
        return [], 1.0

    monkeypatch.setattr(cli, "load_frames", spy)
    assert cli.main([str(sample_video_path)]) == 2
    video, max_frames, stride, width = seen["args"]
    assert isinstance(video, Path) and video == sample_video_path
    assert (max_frames, stride, width) == (180, 4, 960)


def test_cli_options_passed_to_load_frames(monkeypatch, sample_video_path):
    seen = {}

    def spy(*args):
        seen["args"] = args
        return [], 1.0

    monkeypatch.setattr(cli, "load_frames", spy)
    cli.main(
        [str(sample_video_path), "--max-frames", "7", "--stride", "3", "--width", "99"]
    )
    assert seen["args"][1:] == (7, 3, 99)


def test_summary_uses_median_ms_per_frame(monkeypatch, sample_video_path, capsys):
    res = RunResult(seconds_per_frame=[0.2504, 5.0, 0.0001])
    monkeypatch.setattr(perception.detect, "Detector", FakeDetector)
    monkeypatch.setattr(cli, "run", lambda *a, **k: res)
    assert cli.main(_argv(sample_video_path, "--no-depth")) == 0
    assert "mediana 250 ms/klatkę" in capsys.readouterr().out


def test_per_frame_lines_join_several_objects_and_unknown_labels(
    monkeypatch, sample_video_path, capsys
):
    class Two(FakeDetector):
        def __call__(self, img):
            return (
                np.array([[10.0, 100, 90, 160], [200, 100, 280, 160]]),
                np.array([0.9, 0.9]),
                np.array([3, 7]),  # 7 nie ma w id2label
            )

    monkeypatch.setattr(perception.detect, "Detector", Two)
    cli.main(_argv(sample_video_path, "--no-depth"))
    out = capsys.readouterr().out
    assert "klatka    7: #1 car, #2 7" in out


def test_json_box_and_near_rounded_exactly(monkeypatch, sample_video_path, tmp_path):
    class Odd(FakeDetector):
        def __call__(self, img):
            return (
                np.array([[10.123456, 101.0, 90.55555, 160.11111]]),
                np.array([0.9]),
                np.array([3]),
            )

    class OddDepth(FakeDepth):
        def __call__(self, img):
            return np.full(img.shape[:2], 0.123456789)

    monkeypatch.setattr(perception.detect, "Detector", Odd)
    monkeypatch.setattr(perception.depth, "DepthEstimator", OddDepth)
    out = tmp_path / "r.json"
    cli.main(_argv(sample_video_path, "--json", str(out)))
    row = json.loads(out.read_text())[0]
    assert row["box"] == [10.1, 101.0, 90.6, 160.1]
    assert "101.0" in out.read_text()
    assert row["near"] == 0.1235


@pytest.fixture
def written(monkeypatch):
    """Przechwytuje (ścieżka, klatki) przekazane do write_video."""
    calls: dict[str, list[np.ndarray]] = {}
    real = cli.write_video

    def spy(path, frames, fps):
        calls[Path(path).name] = frames
        return real(path, frames, fps)

    monkeypatch.setattr(cli, "write_video", spy)
    return calls


def test_out_videos_contain_boxes_tracks_and_depth(
    monkeypatch, sample_video_path, tmp_path, written
):
    class Diagonal(FakeDetector):
        def __call__(self, img):
            x = 10.0 + 5 * self.calls
            y = 20.0 + 8 * self.calls
            self.calls += 1
            return np.array([[x, y, x + 80, y + 60]]), np.array([0.9]), np.array([3])

    monkeypatch.setattr(perception.detect, "Detector", Diagonal)
    monkeypatch.setattr(perception.depth, "DepthEstimator", FakeDepth)
    argv = _argv(
        sample_video_path,
        "--out", str(tmp_path / "o.mp4"),
        "--out-tracks", str(tmp_path / "t.mp4"),
        "--out-depth", str(tmp_path / "d.mp4"),
    )  # fmt: skip
    cli.main(argv)
    raw, _ = cli.load_frames(sample_video_path, 8, 2, 320)
    boxed, tracks, depth = written["o.mp4"], written["t.mp4"], written["d.mp4"]
    assert not np.array_equal(boxed[7], raw[7])  # skrzynka narysowana w ostatniej
    assert np.array_equal(boxed[0], raw[0])  # a w 0. jeszcze nie (brak toru)
    assert not np.array_equal(tracks[7], boxed[7])  # ślad ponad skrzynkami
    expected = cli.depth_frame(np.full(raw[0].shape[:2], 178, np.uint8))
    assert all(np.array_equal(d, expected) for d in depth)
