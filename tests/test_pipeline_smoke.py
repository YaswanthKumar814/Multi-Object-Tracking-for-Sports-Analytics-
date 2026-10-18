"""End-to-end smoke test on a 40-frame synthetic clip, no model weights needed."""

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from soccer_tracker.cli import main as cli_main
from soccer_tracker.errors import ModelLoadError, VideoOpenError
from soccer_tracker.pipeline import Pipeline
from soccer_tracker.video_io import VideoReader


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_pipeline_end_to_end(synthetic_video, scene, test_config, gt_factory):
    summary = Pipeline(test_config, detector_factory=gt_factory).run(synthetic_video)
    out_video = Path(test_config.io.output_video)

    # annotated video: same size, same number of frames
    with VideoReader(out_video) as r:
        assert (r.info.width, r.info.height) == (scene.width, scene.height)
        assert len(list(r.frames())) == scene.num_frames

    rows = read_csv(test_config.io.output_csv)
    players = [r for r in rows if r["object_type"] == "player"]
    balls = [r for r in rows if r["object_type"] == "ball"]

    # every frame appears (ByteTrack assigns ids from the second frame a
    # track is seen, so frame 0 has no confirmed players), timestamps = frame / fps
    assert {int(r["frame"]) for r in players} == set(range(1, scene.num_frames))
    r10 = next(r for r in players if r["frame"] == "10")
    assert float(r10["timestamp_s"]) == pytest.approx(10 / scene.fps, abs=1e-3)

    # four players, mostly stable ids (allow a couple of switches where boxes cross)
    per_id = Counter(r["track_id"] for r in players)
    assert 4 <= len(per_id) <= 6
    assert sum(1 for n in per_id.values() if n >= 0.8 * scene.num_frames) >= 3, per_id

    # ball: detections dropped on frames 10-12 must be bridged by predictions
    ball_status = {int(r["frame"]): r["status"] for r in balls}
    assert all(ball_status.get(f) == "predicted" for f in (10, 11, 12))
    assert ball_status.get(13) == "detected"
    assert len({r["track_id"] for r in balls}) == 1

    # the predicted ball position stays close to the hidden ground truth
    for f in (10, 11, 12):
        row = next(r for r in balls if int(r["frame"]) == f)
        gx1, gy1, gx2, gy2 = scene.ball_boxes[f]
        assert abs(float(row["center_x"]) - (gx1 + gx2) / 2) < 8
        assert abs(float(row["center_y"]) - (gy1 + gy2) / 2) < 8

    s = json.loads(Path(test_config.io.output_summary).read_text())
    assert s["frames_processed"] == scene.num_frames == summary["frames_processed"]
    assert s["ball"]["frames_predicted"] == 3
    assert s["ball"]["coverage"] > 0.9


def test_pipeline_without_ball(synthetic_video, test_config, gt_factory):
    test_config.ball.enabled = False
    test_config.io.max_frames = 10
    Pipeline(test_config, detector_factory=gt_factory).run(synthetic_video)
    rows = read_csv(test_config.io.output_csv)
    assert rows and all(r["object_type"] == "player" for r in rows)
    assert max(int(r["frame"]) for r in rows) == 9


def test_missing_input_fails_before_creating_outputs(tmp_path, test_config, gt_factory):
    with pytest.raises(VideoOpenError):
        Pipeline(test_config, detector_factory=gt_factory).run(tmp_path / "nope.mp4")
    assert not Path(test_config.io.output_video).exists()
    assert not Path(test_config.io.output_csv).exists()


def test_model_failure_fails_before_creating_outputs(synthetic_video, test_config):
    def broken_factory(cfg):
        raise ModelLoadError("weights missing")

    with pytest.raises(ModelLoadError):
        Pipeline(test_config, detector_factory=broken_factory).run(synthetic_video)
    assert not Path(test_config.io.output_video).exists()


def test_detector_crash_is_wrapped_with_frame_index(synthetic_video, test_config, scene):
    from soccer_tracker.errors import ProcessingError

    class Crashy:
        def __init__(self):
            self.i = 0

        def detect(self, frame):
            self.i += 1
            if self.i == 5:
                raise RuntimeError("CUDA out of memory")
            from soccer_tracker.detection import Detections
            return Detections.empty()

    with pytest.raises(ProcessingError, match="frame 4"):
        Pipeline(test_config, detector_factory=lambda c: (Crashy(), None)).run(synthetic_video)


def test_refuses_to_overwrite_input(synthetic_video, test_config, gt_factory):
    from soccer_tracker.errors import VideoWriteError

    with pytest.raises(VideoWriteError, match="same as the input"):
        Pipeline(test_config, detector_factory=gt_factory).run(synthetic_video, output_video=synthetic_video)


def test_cli_reports_missing_input_cleanly(tmp_path, caplog):
    code = cli_main(["--input", str(tmp_path / "missing.mp4")])
    assert code == 2
    assert "not found" in caplog.text


def test_cli_reports_bad_config_cleanly(tmp_path, synthetic_video, caplog):
    bad = tmp_path / "bad.yaml"
    bad.write_text("players:\n  detector:\n    confidance: 0.2\n")
    assert cli_main(["--input", str(synthetic_video), "--config", str(bad)]) == 2
    assert "Unknown key" in caplog.text
