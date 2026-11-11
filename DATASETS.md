# Datasets

Three public datasets are used. Each has its own adapter in
`src/soccer_tracker/datasets/`; annotations from different datasets are never
mixed, and every result is reported per dataset.

None of the data is stored in Git (`data/` is ignored). Only small subsets are
downloaded: **about 0.73 GB in total instead of roughly 11 GB** for the full
archives. The exact sequences and frame ranges are fixed in
[`configs/experiments.yaml`](configs/experiments.yaml).

| | SportsMOT | UVY-Track | InFactory Soccer Ball Tracking |
|---|---|---|---|
| Source | <https://github.com/MCG-NJU/SportsMOT> (frames mirrored by the Roboflow `trackers` package) | <https://zenodo.org/records/21303900> | <https://huggingface.co/datasets/infactory-ai/ball-tracking> |
| Reference | Cui et al., ICCV 2023 | Alencar & de Freitas (Universidade Federal do Amazonas), 2025-2026 | Constantinou, Iranmanesh & Kanalakis (Infactory), 2026 |
| Content | Broadcast basketball, volleyball, football; 240 clips | 20 user-generated YouTube sport videos, 7 sports (4 soccer) | Soccer broadcast frames from 10 clips |
| Licence | CC BY-NC 4.0, research use | Videos collected under CC BY | CC BY-NC 4.0 |
| Annotation format | MOT Challenge text | MOT-style text with class column + `labels.txt` | YOLO text (normalised `cx cy w h`), one file per frame |
| What is annotated | Players on the pitch only (no referees, no ball) | player, goalkeeper, referee, sports ball, goal, camera | Ball only; frames sorted into `visible/` and `not_visible/` |
| Persistent track IDs | Yes (players) | Yes (people **and** ball) | No |
| Used here for | Player tracking: 1 dev + 3 test sequences | Player tracking and ball detection/tracking: 3 test sequences | Ball detection only: 10 clips |
| Frames used | 1,831 | 1,569 | 1,450 |
| Size on disk | 408 MB of a 6.5 GB archive | 37 MB of a 3.3 GB archive | 283 MB (the 1,450 frames of the published sample; the full repository is 1.3 GB) |

## 1. SportsMOT

**Layout** (verified on the downloaded files):

```
data/raw/sportsmot/val/<sequence>/seqinfo.ini      name, frameRate=25, seqLength, imWidth=1280, imHeight=720
data/raw/sportsmot/val/<sequence>/img1/000001.jpg  1-based frame images
data/raw/sportsmot/val/<sequence>/gt/gt.txt        frame, id, x, y, w, h, 1, 1, 1
```

`x, y` is the top-left corner in pixels. Only athletes on the playing area are
annotated; referees, coaches, stewards and spectators have **no** boxes, and
there are no ball annotations. A generic `person` detector therefore produces
"false positives" on real people that the benchmark does not count as targets
(this matters for MOTA; see the report).

**Subset.** The validation split has 45 sequences. The repository does not
ship per-sport lists, so football sequences were identified from the ground
truth itself (19-22 identities per sequence, versus 10 for basketball and 12 for
volleyball) and confirmed visually. Four football sequences from four different
source videos are used:

| Sequence | Role | Frames | GT tracks | GT boxes |
|---|---|---|---|---|
| `v_2QhNRucNC7E_c017` | **dev** (the only sequence used for tuning) | 450 | 21 | 6,622 |
| `v_dw7LOz17Omg_c067` | test | 450 | 22 | 6,959 |
| `v_G-vNjfx1GGc_c008` | test | 480 | 20 | 4,573 |
| `v_i2_L4qquVg0_c007` | test | 451 | 21 | 4,828 |

**Download.** The full-split command is

```bash
trackers download --name sportsmot --split val --asset frames,annotations --output data/raw/sportsmot
```

but it fetches a single 6.5 GB ZIP. To get only the four sequences, the same
archive is read with HTTP range requests:

```bash
python scripts/fetch_zip_subset.py https://storage.googleapis.com/com-roboflow-marketing/trackers/datasets/sportsmot-v1/sportsmot-val-frames.zip --match "val/v_2QhNRucNC7E_c017/" --match "val/v_dw7LOz17Omg_c067/" --match "val/v_G-vNjfx1GGc_c008/" --match "val/v_i2_L4qquVg0_c007/" --out data/raw/sportsmot --strip 1
```

```bash
python scripts/fetch_zip_subset.py https://storage.googleapis.com/com-roboflow-marketing/trackers/datasets/sportsmot-v1/sportsmot-val-annotations.zip --match "val/(v_2QhNRucNC7E_c017|v_dw7LOz17Omg_c067|v_G-vNjfx1GGc_c008|v_i2_L4qquVg0_c007)/" --out data/raw/sportsmot --strip 1
```

## 2. UVY-Track

> **Record number.** The link originally noted for this dataset
> (`zenodo.org/records/15859365`) resolves to an unrelated biology record. The
> dataset is published as *"UVY: a Dataset of Sport User-Generated Videos from
> YouTube for Multi-Object Tracking Evaluation"* (related identifier
> "UVY-Track") at **<https://zenodo.org/records/21303900>**, a single 3.3 GB
> `UVY.zip`.

**Layout** (verified on the downloaded files):

```
data/raw/uvy_track/<sequence>/<video>.mp4       source clip; video frame i == img1/00000i.jpg
data/raw/uvy_track/<sequence>/video_info.txt    key=value metadata
data/raw/uvy_track/<sequence>/gt/labels.txt     class names, one per line
data/raw/uvy_track/<sequence>/gt/gt.txt         frame, id, x, y, w, h, conf, class, visibility
```

This is **not** YOLO format. `gt.txt` is MOT-style: pixel `x, y, w, h` with a
top-left origin, a persistent `id` per object, and `class` as a 1-based index
into `labels.txt` (1 player, 2 sports ball, 3 goal, 4 referee, 5 goalkeeper,
6 pcamera). The ball has a persistent track id, so identity statistics for the
ball are possible on this dataset.

Findings from inspecting the files, and how the adapter handles them:

* The archive also contains every frame as a JPEG (`img1/`). The MP4 decodes to
  the same frames (checked at frames 1, 50 and 200 of all four soccer clips), so
  only the MP4 and the annotations are downloaded.
* Clips are annotated only for part of their length (for example soccer_V03:
  frames 73-987 of 2,492). Evaluation is restricted to explicit annotated frame
  ranges; frames outside are never scored.
* `video_info.txt` of soccer_V01 names another sequence, so image size and frame
  rate are read from the video.
* A few rows carry track id -1; they are dropped.
* The detector reports the generic COCO class `person`, so player, goalkeeper
  and referee are merged into one "person" target for player tracking. `goal`
  and `pcamera` are ignored.
* **Only part of the people are labelled.** In the fan-filmed penalty-kick clips
  (soccer_V01, soccer_V03) the 4-6 people involved in the kick have boxes, while
  dozens of visible photographers, staff and other players do not. A detector
  that finds those people is not wrong, so metrics that count false positives
  (precision, MOTA, DetA, HOTA, IDF1) are not valid on this dataset. The report
  uses OWTA, detection recall, AssA, identity switches and fragmentation instead.
* The ball boxes are loose: about twice the size of the detector's boxes
  (20x26 px against 12x11 px on soccer_V03), so IoU >= 0.5 is rarely reached even
  by a perfectly centred detection. A centre-distance criterion is used as well.

**Subset.** No parameter is tuned on UVY-Track; all three sequences are test data.

| Sequence | Frames evaluated | Resolution | Person tracks | Visible-ball frames |
|---|---|---|---|---|
| `soccer_V04` | 1-219 (all annotated frames) | 1280x720 | 19 | 168 |
| `soccer_V01` | 1-750 of 1,509 | 636x360 | 6 | 750 |
| `soccer_V03` | 82-681 of the annotated 82-987 | 1280x720 | 5 | 600 |

`soccer_V02` was downloaded but not evaluated, to keep CPU time bounded (its
ball is annotated in only 116 frames).

**Download** (37 MB on disk instead of 3.3 GB):

```bash
python scripts/fetch_zip_subset.py "https://zenodo.org/records/21303900/files/UVY.zip?download=1" --match "UVY/soccer_V0[1-4]/(gt/|video_info\.txt|[^/]+\.mp4)" --match "UVY/(metrics_per_video\.csv|dataset_summary\.json)" --out data/raw/uvy_track --strip 1
```

## 3. InFactory Soccer Ball Tracking

**Layout** (verified on the downloaded files):

```
data/raw/infactory_ball/metadata.csv                        file_path, video_source, frame_index, visibility, bboxes_count
data/raw/infactory_ball/data/visible/<uuid>_<frame>.jpg     frame with a visible ball
data/raw/infactory_ball/data/visible/<uuid>_<frame>.txt     YOLO label: 0 cx cy w h (normalised)
data/raw/infactory_ball/data/not_visible/<uuid>_<frame>.jpg frame without a visible ball
```

`metadata.csv` defines the published sample: **1,450 frames from 10 clips, 884
with a visible ball and 566 without**. Frame indices inside a clip are
contiguous, so each clip is replayed in order as a short video (frames are
1280x720 or 720x608). The repository currently holds more image files than the
metadata lists (about 1.3 GB); only the 1,450 listed frames are used.

The labels are ball boxes only. There are no player annotations and **no
persistent ball identity**, so the dataset is used for ball **detection**
evaluation only (precision, recall, F1, AP@0.5, coverage, centre error). No
identity-based tracking metric is reported for it.

**Download.** The plain command fetches the whole repository:

```bash
hf download infactory-ai/ball-tracking --repo-type dataset --local-dir data/raw/infactory_ball
```

The experiments only need the files listed in `metadata.csv`, which can be
fetched with:

```bash
python -c "import csv,io,urllib.request as u;from huggingface_hub import snapshot_download as s;b='https://huggingface.co/datasets/infactory-ai/ball-tracking/resolve/main/metadata.csv';ids=sorted({r['video_source'] for r in csv.DictReader(io.StringIO(u.urlopen(b).read().decode()))});s('infactory-ai/ball-tracking',repo_type='dataset',local_dir='data/raw/infactory_ball',allow_patterns=[f'data/*/{i}_*' for i in ids]+['metadata.csv','README.md','LICENSE','dataset_info.json'])"
```

## Tuning versus testing

| Data | Used to choose parameters? | Reported as |
|---|---|---|
| SportsMOT `v_2QhNRucNC7E_c017` | Yes: tracker hyper-parameter grid (same grid for all three trackers) | "dev", shown separately, never in test averages |
| SportsMOT, the three other sequences | No | test |
| UVY-Track, three soccer sequences | No | test |
| InFactory, ten clips | No (the ball tracker runs with its default configuration) | test |

The dev sequence comes from a different source video than every test sequence.
