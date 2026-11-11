"""Dataset adapters, checked against tiny fixtures written in each dataset's real layout."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from soccer_tracker.datasets import DatasetError, InFactoryClip, SportsMOTSequence, UVYTrackSequence, list_clips
from soccer_tracker.datasets.infactory import parse_yolo_label
from soccer_tracker.synthetic import generate_scene, write_video


def _img(path, w=64, h=48, value=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((h, w, 3), value, np.uint8))


# --------------------------------------------------------------------------- #
# SportsMOT
# --------------------------------------------------------------------------- #
@pytest.fixture
def sportsmot_root(tmp_path):
    seq = tmp_path / "val" / "v_demo_c001"
    (seq / "gt").mkdir(parents=True)
    (seq / "seqinfo.ini").write_text(
        "[Sequence]\nname=v_demo_c001\nimDir=img1\nframeRate=25\nseqLength=3\nimWidth=64\nimHeight=48\nimExt=.jpg\n")
    for f in (1, 2, 3):
        _img(seq / "img1" / f"{f:06d}.jpg", value=10 * f)
    (seq / "gt" / "gt.txt").write_text(
        "1, 0, 10, 5, 8, 20, 1, 1, 1\n1, 1, 30, 5, 8, 20, 1, 1, 1\n2, 0, 11, 5, 8, 20, 1, 1, 1\n"
        "3, 0, 12, 5, 8, 20, 0, 1, 1\n")  # last row is flagged "ignore"
    return tmp_path


def test_sportsmot_reads_info_frames_and_ground_truth(sportsmot_root):
    seq = SportsMOTSequence(sportsmot_root, "v_demo_c001", split="val", role="dev")
    assert (seq.info.dataset, seq.info.fps, seq.info.num_frames) == ("SportsMOT", 25.0, 3)
    assert (seq.info.width, seq.info.height, seq.info.role) == (64, 48, "dev")
    frames = list(seq.frames())
    assert [f for f, _ in frames] == [1, 2, 3] and frames[0][1].shape == (48, 64, 3)
    gt = seq.player_gt()
    assert gt.shape == (3, 6)  # ignored row dropped
    assert gt[0].tolist() == [1, 0, 10, 5, 8, 20]
    assert seq.has_player_ids and not seq.has_ball_boxes and seq.ball_gt() == {}
    assert seq.describe()["player_gt_tracks"] == 2


def test_sportsmot_max_frames_and_missing_sequence(sportsmot_root):
    seq = SportsMOTSequence(sportsmot_root, "v_demo_c001", max_frames=2)
    assert seq.info.num_frames == 2 and len(list(seq.frames())) == 2
    assert seq.player_gt()[:, 0].max() == 2
    with pytest.raises(DatasetError, match="not found"):
        SportsMOTSequence(sportsmot_root, "v_missing")


# --------------------------------------------------------------------------- #
# UVY-Track
# --------------------------------------------------------------------------- #
@pytest.fixture
def uvy_root(tmp_path):
    seq = tmp_path / "soccer_V99"
    (seq / "gt").mkdir(parents=True)
    write_video(generate_scene(width=240, height=200, num_frames=12, num_players=2), seq / "clip.mp4")
    (seq / "gt" / "labels.txt").write_text("player\nsports ball\ngoal\nreferee\ngoalkeeper\npcamera\n")
    lines = []
    for f in range(3, 9):  # only frames 3-8 are annotated
        lines += [f"{f},1,{10 + f},20,10,30,1,1,1.0",       # player
                  f"{f},2,{60 + f},20,10,30,1,5,1.0",       # goalkeeper
                  f"{f},3,100,20,10,30,1,4,1.0",            # referee
                  f"{f},4,0,0,80,40,1,3,1.0",               # goal (ignored)
                  f"{f},-1,5,5,5,5,1,1,1.0"]                # leftover row with a negative id
        if f != 6:
            lines.append(f"{f},9,{40 + 2 * f},50,6,6,1,2,1.0")  # ball, not visible in frame 6
    (seq / "gt" / "gt.txt").write_text("\n".join(lines) + "\n")
    (seq / "video_info.txt").write_text("seqname=wrong_name\nFPS=30\n")
    return tmp_path


def test_uvy_merges_person_classes_and_renumbers_frames(uvy_root):
    seq = UVYTrackSequence(uvy_root, "soccer_V99")
    assert seq.info.num_frames == 6 and seq.info.source_first_frame == 3
    assert (seq.info.width, seq.info.height) == (240, 200)  # read from the video, not video_info.txt
    gt = seq.player_gt()
    assert sorted(np.unique(gt[:, 1]).tolist()) == [1, 2, 3]  # player, goalkeeper, referee; no goal, no id -1
    assert gt[:, 0].min() == 1 and gt[:, 0].max() == 6
    assert len(gt) == 18
    assert [f for f, _ in seq.frames()] == [1, 2, 3, 4, 5, 6]


def test_uvy_ball_boxes_ids_and_labelled_frames(uvy_root):
    seq = UVYTrackSequence(uvy_root, "soccer_V99")
    balls = seq.ball_gt()
    assert sorted(balls) == [1, 2, 3, 5, 6]  # source frame 6 (ball hidden) is evaluated frame 4
    np.testing.assert_allclose(balls[1], [[46, 50, 52, 56]])  # xywh (46, 50, 6, 6) -> xyxy
    assert seq.ball_track_ids() == {f: 9 for f in balls}
    assert seq.ball_labelled_frames() == {1, 2, 3, 4, 5, 6}
    assert seq.has_ball_ids and seq.has_ball_boxes and seq.has_player_ids


def test_uvy_explicit_range_and_errors(uvy_root):
    seq = UVYTrackSequence(uvy_root, "soccer_V99", first_frame=5, last_frame=7)
    assert seq.info.num_frames == 3 and sorted(seq.ball_gt()) == [1, 3]
    assert len(list(seq.frames())) == 3
    with pytest.raises(DatasetError):
        UVYTrackSequence(uvy_root, "soccer_V00")
    with pytest.raises(DatasetError, match="ended"):
        list(UVYTrackSequence(uvy_root, "soccer_V99", first_frame=3, last_frame=40).frames())


# --------------------------------------------------------------------------- #
# InFactory
# --------------------------------------------------------------------------- #
@pytest.fixture
def infactory_root(tmp_path):
    rows = ["file_path,video_source,frame_index,visibility,bboxes_count"]
    for i in range(4):
        vis = i != 2
        folder = "visible" if vis else "not_visible"
        rel = f"data/{folder}/clipA_{i:06d}.jpg"
        _img(tmp_path / rel, w=200, h=100)
        if vis:
            (tmp_path / rel).with_suffix(".txt").write_text("0 0.5 0.5 0.1 0.2\n")
        rows.append(f"{rel},clipA,{i:06d},{folder},{int(vis)}")
    rel = "data/visible/clipB_000007.jpg"
    _img(tmp_path / rel, w=200, h=100)
    (tmp_path / rel).with_suffix(".txt").write_text("0 0.25 0.5 0.1 0.2\n0 0.75 0.5 0.1 0.2\n")
    rows.append(f"{rel},clipB,000007,visible,2")
    (tmp_path / "metadata.csv").write_text("\n".join(rows) + "\n")
    return tmp_path


def test_parse_yolo_label_converts_normalised_to_pixels():
    boxes = parse_yolo_label("0 0.5 0.5 0.1 0.2\n\n0 0.1 0.1 0.2 0.2\n", width=200, height=100)
    np.testing.assert_allclose(boxes, [[90, 40, 110, 60], [0, 0, 40, 20]])
    assert parse_yolo_label("", 10, 10).shape == (0, 4)


def test_infactory_clip_frames_and_ball_boxes(infactory_root):
    assert list_clips(infactory_root) == ["clipA", "clipB"]
    clip = InFactoryClip(infactory_root, "clipA")
    assert clip.info.num_frames == 4 and (clip.info.width, clip.info.height) == (200, 100)
    assert [f for f, _ in clip.frames()] == [1, 2, 3, 4]
    balls = clip.ball_gt()
    assert sorted(balls) == [1, 2, 4]  # frame 3 is a "not visible" negative
    np.testing.assert_allclose(balls[1], [[90, 40, 110, 60]])
    assert clip.ball_labelled_frames() is None  # every frame has a known ball status
    assert clip.has_ball_boxes and not clip.has_ball_ids and not clip.has_player_ids
    assert len(clip.player_gt()) == 0
    assert InFactoryClip(infactory_root, "clipB").ball_gt()[1].shape == (2, 4)


def test_infactory_errors(infactory_root, tmp_path):
    with pytest.raises(DatasetError, match="not listed"):
        InFactoryClip(infactory_root, "nope")
    with pytest.raises(DatasetError, match="metadata"):
        list_clips(tmp_path / "missing")
