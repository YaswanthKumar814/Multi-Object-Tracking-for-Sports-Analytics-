import numpy as np
import pytest

from soccer_tracker.errors import VideoOpenError, VideoWriteError
from soccer_tracker.video_io import VideoReader, VideoWriter


def test_reads_synthetic_video(synthetic_video, scene):
    with VideoReader(synthetic_video) as r:
        assert r.info.width == scene.width and r.info.height == scene.height
        assert r.info.fps == pytest.approx(scene.fps, rel=0.01)
        frames = list(r.frames())
    assert len(frames) == scene.num_frames
    idxs = [f[0] for f in frames]
    assert idxs == list(range(scene.num_frames))
    assert frames[10][1] == pytest.approx(10 / scene.fps)
    assert frames[0][2].shape == (scene.height, scene.width, 3)


def test_max_frames_limits_iteration(synthetic_video):
    with VideoReader(synthetic_video, max_frames=5) as r:
        assert len(list(r.frames())) == 5


def test_capture_released_on_exit_even_after_error(synthetic_video):
    reader = VideoReader(synthetic_video)
    with pytest.raises(RuntimeError):
        with reader:
            for _ in reader.frames():
                raise RuntimeError("boom")
    assert reader._cap is None


def test_missing_file(tmp_path):
    with pytest.raises(VideoOpenError, match="not found"):
        VideoReader(tmp_path / "missing.mp4").open()


def test_directory_is_rejected(tmp_path):
    d = tmp_path / "clip.mp4"
    d.mkdir()
    with pytest.raises(VideoOpenError, match="not a file"):
        VideoReader(d).open()


def test_unsupported_extension(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hello")
    with pytest.raises(VideoOpenError, match="Unsupported video extension"):
        VideoReader(p).open()


def test_empty_file(tmp_path):
    p = tmp_path / "empty.mp4"
    p.write_bytes(b"")
    with pytest.raises(VideoOpenError, match="empty"):
        VideoReader(p).open()


def test_corrupt_file(tmp_path):
    p = tmp_path / "corrupt.mp4"
    p.write_bytes(np.random.default_rng(0).integers(0, 255, 4096, dtype=np.uint8).tobytes())
    with pytest.raises(VideoOpenError):
        VideoReader(p).open()


def test_writer_rejects_wrong_frame_size(tmp_path):
    with VideoWriter(tmp_path / "o.mp4", 25, (64, 48)) as w:
        with pytest.raises(VideoWriteError, match="does not match"):
            w.write(np.zeros((10, 10, 3), np.uint8))
    assert w._writer is None


def test_writer_creates_parent_dirs_and_roundtrips(tmp_path):
    out = tmp_path / "a" / "b" / "o.mp4"
    with VideoWriter(out, 25, (64, 48)) as w:
        for _ in range(5):
            w.write(np.full((48, 64, 3), 128, np.uint8))
    assert out.is_file() and w.frames_written == 5
    with VideoReader(out) as r:
        assert len(list(r.frames())) == 5


def test_writer_respects_overwrite_flag(tmp_path):
    out = tmp_path / "o.mp4"
    out.write_bytes(b"x")
    with pytest.raises(VideoWriteError, match="already exists"):
        VideoWriter(out, 25, (64, 48), overwrite=False).open()


def test_writer_reports_when_opencv_cannot_open(tmp_path, monkeypatch):
    # Whether OpenCV rejects a given codec depends on the platform and build
    # (Windows builds often fall back to another codec instead of failing), so
    # simulate a writer that fails to open and check our error handling.
    import soccer_tracker.video_io as vio

    class FailingWriter:
        released = False

        def __init__(self, *args, **kwargs):
            pass

        def isOpened(self):
            return False

        def release(self):
            FailingWriter.released = True

    monkeypatch.setattr(vio.cv2, "VideoWriter", FailingWriter)
    with pytest.raises(VideoWriteError, match="codec"):
        VideoWriter(tmp_path / "o.mp4", 25, (64, 48), fourcc="ZZZZ").open()
    assert FailingWriter.released  # the handle is released even on failure