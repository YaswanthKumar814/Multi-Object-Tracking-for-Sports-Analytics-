# Multi-Object Tracking for Sports Analytics (Player and Ball Tracking)

**Proposed system: YOLO11n + ByteTrack (players) + Kalman filter (ball)**, compared
against YOLO11n + SORT and YOLO11n + OC-SORT on three public datasets
(SportsMOT, UVY-Track, InFactory Soccer Ball Tracking).

| Document | Contents |
|---|---|
| [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md) | Full report: method, datasets, baseline comparison, metrics, plots, limitations |
| [DATASETS.md](DATASETS.md) | Dataset sources, formats, the exact subsets used, download commands |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design |
| [presentation.pptx](presentation.pptx), [docs/PRESENTATION_NOTES.md](docs/PRESENTATION_NOTES.md) | Slides and speaker notes |
| [results/](results/) | Metrics (CSV/JSON), plots (PNG), SQLite database, demo outputs |

The application detects players and the ball in a local soccer video, tracks player
identities across frames, tracks the ball as a separate component, and writes:

- an **annotated video** (boxes, track IDs, player trails, ball marker and trail, status overlay),
- a **CSV** with one row per tracked object per frame (timestamp, ID, class, confidence, box, centre),
- a **JSON run summary** (counts, ball coverage, processing speed, the config used).

Built on pretrained models only: Ultralytics YOLO for detection, ByteTrack for player
tracking, and a Kalman-filter single-target tracker for the ball. No training is required.
Ball positions that come from the motion model are always marked `predicted` (orange in
the video, `status=predicted` and an empty confidence in the CSV); they are never
presented as detections.

---

## 1. Installation

Requires Python 3.10+ (tested on 3.13). A GPU is optional; CPU works but is slow.

```bash
cd soccer-tracker
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Optional but recommended: install the PyTorch build that matches your hardware
# FIRST, from https://pytorch.org/get-started/locally/ (CUDA, CPU-only, or Apple MPS).
# If you skip this, ultralytics installs the default torch wheel for your platform.

pip install -r requirements-dev.txt
pip install -e .            # makes `python -m soccer_tracker` / `soccer-tracker` available
```

`requirements.txt` covers the application. The experiments, plots and slides
additionally use `matplotlib`, `scipy`, `huggingface_hub` and `python-pptx`, which
`requirements-dev.txt` installs.

Check the install:

```bash
pytest                      # unit + smoke tests; YOLO tests are skipped until weights exist
```

## 2. Running

```bash
python -m soccer_tracker --input path/to/match.mp4
```

On first run the default weights (`weights/yolo11n.pt`, ~5 MB) are downloaded
automatically from the Ultralytics GitHub releases. Outputs go to `outputs/`.

Useful options:

| Option | Meaning |
|---|---|
| `-c configs/default.yaml` | Use a config file (all keys optional) |
| `-o out/video.mp4`, `--output-csv`, `--output-summary` | Override output paths |
| `--max-frames 300` | Process only the first N frames (fast check) |
| `--tracker bytetrack` / `sort` / `ocsort` | Player tracker: ByteTrack (proposed, default) or a baseline |
| `--device cpu` / `--device 0` / `--device mps` | Force CPU, first CUDA GPU, or Apple GPU |
| `--player-weights`, `--ball-weights` | Override weights without editing YAML |
| `--ball-tiling` | Sliced inference for the ball (better for tiny balls, slower) |
| `--no-ball` | Players only |
| `-v` | Debug logging, full cause on errors |

Exit codes: `0` success, `2` expected error (message printed), `130` interrupted.

**Start with a short clip.** Try `--max-frames 200` on a 720p clip first. On CPU
expect roughly 1–3 FPS at the default `imgsz: 1280`; see Troubleshooting for speed.

### Outputs

`tracks.csv` columns:

| Column | Description |
|---|---|
| `frame` | 0-based frame index |
| `timestamp_s` | `frame / fps` (seconds) |
| `object_type` | `player` or `ball` |
| `track_id` | Player: tracker ID (ByteTrack by default). Ball: ball-track segment ID (increments when the ball is re-acquired after being lost) |
| `class_name` | Detector class (e.g. `person`, `sports ball`, or your model's names) |
| `confidence` | Detector confidence; **empty** for predicted ball positions |
| `status` | `detected`, or `predicted` (ball position from the motion model while undetected) |
| `x1,y1,x2,y2` | Box in pixels |
| `center_x,center_y` | Box centre in pixels |

All coordinates are **image pixels**, not pitch metres.

`summary.json` includes frames processed, processing FPS, unique player IDs, median
track length, count of tracks shorter than 10 frames (a rough fragmentation signal),
ball frames detected/predicted/lost and coverage, and the full effective config.

### Demo run used in the report

```bash
python -m soccer_tracker --input data/raw/uvy_track/soccer_V04/vid__fkPwtg7ozU.mp4 -o results/demo/uvy_soccer_V04_annotated.mp4 --output-csv results/demo/uvy_soccer_V04_tracks.csv --output-summary results/demo/uvy_soccer_V04_summary.json --device cpu
```

```bash
python scripts/visualize_run.py --csv results/demo/uvy_soccer_V04_tracks.csv --video data/raw/uvy_track/soccer_V04/vid__fkPwtg7ozU.mp4 --annotated results/demo/uvy_soccer_V04_annotated.mp4 --out-dir results/demo --prefix uvy_soccer_V04
```

The first command writes the annotated MP4, the tracks CSV and the JSON summary; the
second draws the player-trajectory, heatmap and ball-trajectory figures for that run.
`scripts/frames_to_video.py` converts a folder of frames (for example a SportsMOT
sequence) into an MP4 the application can read.

## 2b. Experiments: baselines, metrics, plots

Get the data subsets first (commands in [DATASETS.md](DATASETS.md), about 0.73 GB), then:

```bash
python scripts/run_experiments.py all
```

That runs, in order (each stage can also be run on its own by name):

| Stage | What it does | Output |
|---|---|---|
| `detect` | Runs YOLO11n once per sequence and caches the detections, so every tracker sees identical boxes. The slow step: 0.3-1 s per frame on a CPU, 4,850 frames. Resumable | `results/cache/` |
| `runtime` | Times every pipeline stage on 150 frames (run on an idle machine) | `results/metrics/runtime_*` |
| `players` | SORT vs OC-SORT vs ByteTrack on the cached detections: parameter grid on one dev sequence, then HOTA / DetA / AssA / MOTA / IDF1 / ID switches / FP / FN / fragmentation on the test sequences | `results/metrics/player_*`, `results/tracks/` |
| `ball` | Detection-only vs Kalman ball tracking: precision, recall, F1, AP@0.5, coverage, centre error, continuity | `results/metrics/ball_*` |
| `models` | Optional YOLOv8n vs YOLO11n comparison on a subset | `results/metrics/model_comparison.csv` |
| `plots` | All figures | `results/plots/*.png` |
| `db` | Rebuilds the SQLite database from the result files | `results/sports_analytics.db` |

```bash
python scripts/make_report_tables.py
```

regenerates `results/tables.md` and refreshes the tables in the report from the metrics
files. Which sequences and frame ranges are used is fixed in
[`configs/experiments.yaml`](configs/experiments.yaml).

**Evaluators.** Player-tracking metrics come from `trackers.eval` (Roboflow `trackers`
package), an established re-implementation built to match TrackEval. Ball and detector
metrics come from this project's own evaluator (`src/soccer_tracker/evaluation/`), which
is documented and unit-tested but is **not** an official benchmark evaluator. All numbers
are computed on small subsets with a generic COCO detector and are not comparable to
leaderboard results.

### Headline results

**Player tracking, SportsMOT test sequences** (3 sequences, 1,381 frames; parameters
chosen on one separate dev sequence; identical YOLO11n detections for every tracker;
best value per column in bold):

<!-- TABLE:tracking_sportsmot -->
| Method | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN | Frag | MT | ML | Prec. | Rec. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11n + SORT | **65.90** | **64.83** | **67.03** | **64.90** | **70.56** | 60 | **4898** | 784 | **158** | 59 | **0** | **76.08** | 95.21 |
| YOLO11n + OC-SORT | 59.27 | 60.63 | 57.97 | 55.35 | 61.21 | 106 | 6382 | 817 | 230 | 61 | **0** | 70.89 | 95.01 |
| YOLO11n + ByteTrack (proposed) | 62.63 | 59.35 | 66.13 | 52.63 | 66.84 | **51** | 7029 | **670** | 168 | **62** | **0** | 69.06 | **95.91** |
<!-- /TABLE:tracking_sportsmot -->

**Player tracking, UVY-Track test sequences** (3 sequences, 1,569 frames). UVY-Track
labels only part of the people in each clip, so false positives cannot be measured;
OWTA, detection recall (DetRe) and AssA are the valid metrics, and the columns marked *
are shown only for completeness:

<!-- TABLE:tracking_uvy_partial -->
| Method | OWTA | DetRe | AssA | IDSW | Frag | MT | ML | FN | HOTA* | MOTA* | IDF1* | FP* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11n + SORT | 55.75 | 56.44 | **57.41** | 25 | **147** | **16** | 1 | 2504 | 38.50 | -31.96 | 46.85 | 9956 |
| YOLO11n + OC-SORT | 54.49 | 57.54 | 54.22 | 44 | 193 | **16** | **0** | 2366 | 33.46 | -85.50 | 38.05 | 15140 |
| YOLO11n + ByteTrack (proposed) | **56.80** | **58.67** | 57.34 | **23** | 165 | **16** | **0** | **2243** | 34.31 | -93.25 | 39.06 | 16017 |
<!-- /TABLE:tracking_uvy_partial -->

**Ball: detection only vs Kalman tracker** (13 sequences of UVY-Track and InFactory;
"dist" = centre within one ball diameter of the ground truth):

<!-- TABLE:ball_summary -->
| Data | Method | Visible-ball frames | P (dist) | R (dist) | F1 (dist) | P (IoU) | R (IoU) | F1 (IoU) | AP@0.5 | Coverage | Centre error of correct outputs (px) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| InFactory | Detection only (YOLO11n) | 884 | 74.51 | 34.39 | 47.06 | 66.91 | 30.88 | 42.26 | 25.95 | 28.14 | 2.11 |
| InFactory | Kalman tracker, detected positions only | 884 | 85.13 | 30.43 | 44.83 | 74.68 | 26.70 | 39.33 | n/a | 21.79 | 3.29 |
| InFactory | Kalman tracker: detected + predicted (proposed) | 884 | 62.24 | 43.44 | 51.17 | 46.19 | 32.24 | 37.98 | n/a | 42.55 | 4.59 |
| UVY-Track | Detection only (YOLO11n) | 1518 | 62.43 | 34.59 | 44.51 | 25.45 | 14.10 | 18.14 | 12.02 | 53.60 | 1.99 |
| UVY-Track | Kalman tracker, detected positions only | 1518 | 61.11 | 28.99 | 39.32 | 30.42 | 14.43 | 19.57 | n/a | 45.89 | 2.00 |
| UVY-Track | Kalman tracker: detected + predicted (proposed) | 1518 | 54.85 | 43.22 | 48.34 | 19.48 | 15.35 | 17.17 | n/a | 76.23 | 3.15 |
| All ball sequences | Detection only (YOLO11n) | 2402 | 66.37 | 34.51 | 45.41 | 38.99 | 20.27 | 26.68 | 16.29 | 41.37 | 2.03 |
| All ball sequences | Kalman tracker, detected positions only | 2402 | 68.44 | 29.52 | 41.24 | 43.92 | 18.94 | 26.47 | n/a | 34.32 | 2.49 |
| All ball sequences | Kalman tracker: detected + predicted (proposed) | 2402 | 57.36 | 43.30 | 49.35 | 28.57 | 21.57 | 24.58 | n/a | 60.05 | 3.68 |
<!-- /TABLE:ball_summary -->

In short: ByteTrack (proposed) gives the fewest identity switches and the highest recall
on both tracking datasets and the best OWTA on UVY-Track, but tuned SORT has the higher
HOTA / MOTA / IDF1 on SportsMOT. The Kalman ball tracker raises ball recall, F1 and track
continuity over detection alone at the cost of precision. The generic COCO detector is
the main limit on both accuracy and speed (about 3 FPS on a laptop CPU). Full results,
plots and discussion: [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md).

## 3. Configuration

`configs/default.yaml` lists every option with its default and a comment. You only
need to put the keys you change in your own YAML; unknown keys are rejected so typos
fail loudly. Relative paths resolve against the directory you run the command from.

Most important knobs:

| Key | Default | Why you'd change it |
|---|---|---|
| `players.detector.weights` | `weights/yolo11n.pt` | `yolo11s.pt`/`yolo11m.pt` = more accurate, slower |
| `players.detector.imgsz` | 1280 | 640 is ~3× faster but misses distant players |
| `players.filter.max_height_frac` | 0.6 | Raise to 1.0 for close-up footage |
| `players.filter.use_pitch_mask` | false | `true` drops detections whose feet are not on grass (crowd, bench) |
| `players.tracker.algorithm` | `bytetrack` | `sort` or `ocsort` to run a baseline tracker |
| `players.tracker.lost_track_buffer` | 30 | Higher keeps IDs through longer occlusions, risks wrong re-association |
| `ball.detector.weights` / `classes` | COCO `sports ball` | Point to soccer-specific weights (see §4) |
| `ball.detector.tiling.enabled` | false | Helps with tiny balls; costs several extra inferences per frame |
| `ball.tracker.gate_px` | 80 | Larger for high-resolution or fast-paced footage |
| `ball.tracker.max_missed_frames` | 15 | How long to show a predicted ball when undetected |
| `io.fourcc` | `mp4v` | `avc1` for browser-playable H.264 (if your OpenCV supports it) |

`configs/soccer_finetuned.example.yaml` shows the setup for a soccer-specific model.

## 4. Model weights and licensing

| Weights | Source | Licence / access | Used for |
|---|---|---|---|
| `yolo11n.pt` (default), `yolo11s/m/l/x.pt`, `yolov8*.pt`, `yolo26*.pt` | Downloaded automatically by Ultralytics from `github.com/ultralytics/assets/releases` (set `allow_download: false` to forbid this) | **AGPL-3.0** (Ultralytics). Fine for coursework; commercial/closed use needs an Ultralytics Enterprise licence | Players (`person`) and baseline ball (`sports ball`), COCO-pretrained |
| Soccer-specific YOLO weights (optional) | You supply the `.pt` file. Options below | Depends on the source; models trained with Ultralytics inherit AGPL-3.0 | Better players/ball |

The libraries: Ultralytics (AGPL-3.0), Roboflow `trackers` (Apache-2.0), `supervision`
(MIT), OpenCV (Apache-2.0), PyTorch (BSD-style).

### Ball detection: what to expect and how to improve it

The COCO `sports ball` class was trained mostly on close-up balls. In wide broadcast
shots the ball is often 5–15 pixels, blurred, or hidden, so **expect frequent misses
and occasional false positives** with the default weights. The project handles this
in layers, each replaceable independently:

1. **Detection** (`ball.detector`): low confidence threshold, high `imgsz`, optional
   tiling, and any weights/class name you configure.
2. **Filtering** (`ball.filter`): rejects boxes that are too large, not square, or
   (optionally) off the pitch.
3. **Tracking** (`ball.tracker`): picks the candidate consistent with the ball's
   motion, bridges short gaps with predictions (drawn in orange, `status=predicted`
   in the CSV), and re-acquires after long passes.

To substantially improve ball recall you need soccer-specific weights:

- **Pretrained, ready to use:** Roboflow's open-source `sports` repository
  (github.com/roboflow/sports, soccer example) distributes YOLOv8 weights fine-tuned
  for soccer players and the ball; follow its setup script for download links and
  check its licence. Then point `ball.detector.weights` at the file and set
  `classes: [ball]` (use whatever class name the model reports; a wrong name prints
  the valid ones).
- **Fine-tune yourself (if needed):** public soccer datasets on Roboflow Universe
  (search "football players detection"; many are CC BY 4.0 and include a `ball`
  class) can be exported in YOLO format and trained with
  `yolo detect train data=data.yaml model=yolo11s.pt imgsz=1280`. Roboflow's
  ball-tracking write-up used footage from the Kaggle *DFL Bundesliga Data Shootout*
  competition; that data is subject to the competition's rules, so check them before use.

## 5. Tests

```bash
pytest                    # everything; YOLO tests skip if weights/yolo11n.pt is absent
pytest -m "not integration"   # fast tests only, no PyTorch needed at run time
pytest -m integration     # real-model tests (needs weights; run the CLI once to download)
```

| File | What it checks |
|---|---|
| `test_config.py` | Defaults valid; `default.yaml` matches code defaults; typos, wrong types and out-of-range values rejected |
| `test_video_io.py` | Reading/writing, FPS and timestamps, missing/empty/corrupt/unsupported files, bad codec, handle release on error |
| `test_detection_utils.py` | IoU, NMS, tile coverage, player/ball filters, pitch mask, shared inference pass |
| `test_ball_tracker.py` | Confirmation, linear motion, coasting through gaps, false-positive rejection, re-acquisition, dropping |
| `test_player_tracker.py` | Stable IDs, survival across short gaps, activation threshold, class names |
| `test_analytics.py` | CSV columns/values, summary statistics |
| `test_pipeline_smoke.py` | Full pipeline on a 40-frame synthetic clip (ground-truth detector, no weights): video, CSV, summary, gap bridging, error paths, CLI exit codes |
| `test_yolo_integration.py` | Real YOLO loading, class lookup, download policy errors, tiled inference, pipeline with real model |
| `test_datasets.py` | SportsMOT / UVY-Track / InFactory adapters on small fixtures in each real layout: frame ranges, class merging, YOLO label conversion, error paths |
| `test_evaluation.py` | Evaluators against hand-computed cases: precision/recall/F1, AP, ball metrics (predicted vs detected, continuity, centre error, ID switches), MOT file conversion with a known ID switch |
| `test_experiments.py` | Tracker factory (SORT / OC-SORT / ByteTrack), `--tracker` option, detection cache round trip, deterministic tracker replay, equal tuning budget, CSV round trip, database schema |

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `Model weights not found ... not an official Ultralytics asset` | Custom weights must exist at the configured path; check spelling and working directory |
| Download fails / offline machine | Download `yolo11n.pt` from the Ultralytics assets releases page and place it in `weights/` |
| `Class name(s) [...] not found in model` | The error lists the model's class names; copy the right one into `classes` |
| `OpenCV could not open ...` / no frame decoded | Re-encode: `ffmpeg -i in.mkv -c:v libx264 -pix_fmt yuv420p out.mp4` |
| Output video won't play in a browser/VS Code | `mp4v` is MPEG-4 Part 2. Play it in VLC, set `io.fourcc: avc1`, or re-encode with ffmpeg as above |
| `Could not open a video writer ... codec` | Match codec and extension: `mp4v`/`avc1` → `.mp4`, `XVID`/`MJPG` → `.avi` |
| Very slow | Use a GPU (`--device 0`); lower `imgsz` to 640–960; keep `ball.share_player_inference: true` and tiling off; test with `--max-frames` |
| No players drawn on close-up footage | `players.filter.max_height_frac` (0.6) removes boxes taller than 60% of the frame; raise it |
| Crowd/coaches tracked as players | `players.filter.use_pitch_mask: true`; tune `pitch_mask.hsv_*` if your grass colour differs |
| Many short tracks / ID switches | Raise `lost_track_buffer`; use a larger model (`yolo11s.pt`); lower `players.detector.confidence` so ByteTrack's second stage has candidates |
| Ball rarely found | Expected with COCO weights; enable tiling and/or use soccer-specific weights (§4) |
| Ball jumps to a wrong object | Lower `ball.tracker.gate_px`, raise `reinit_confidence`, enable `ball.filter.use_pitch_mask` |
| CUDA out of memory | Lower `imgsz`, disable tiling, or use `--device cpu` |
| `No module named soccer_tracker` | Run `pip install -e .` in the project folder (or `PYTHONPATH=src`) |

## 7. Features, limitations, next steps

**Implemented**
- Video input validation (existence, extension, emptiness, decodability, FPS fallback) and safe output handling
- YOLO player detection with configurable weights, classes, thresholds, resolution and device
- ByteTrack player tracking with configurable association parameters; SORT and OC-SORT selectable as baselines (`--tracker`)
- Dataset adapters for SportsMOT, UVY-Track and InFactory, and a tool that downloads only selected sequences from a remote ZIP
- Reproducible experiments: cached detections, three-tracker comparison with HOTA / MOTA / IDF1, ball evaluation, runtime benchmark, plots, SQLite database
- Separate ball pipeline: detection (optional tiling) → filtering → Kalman tracking with gating, coasting, re-acquisition
- Annotated video: per-ID colours, IDs, foot-point trails, ball marker and trail (detected vs. predicted), status overlay
- Streaming CSV export and JSON run summary
- Single shared inference pass for players and ball when possible
- YAML config with strict validation; CLI overrides; clear errors and exit codes
- 109 automated tests including an end-to-end smoke test without model weights

**Known limitations**
- Ball detection with COCO weights is weak on broadcast footage (see §4)
- ID switches during long occlusions and crowded set pieces; IDs are not kept across camera cuts (no re-identification)
- ByteTrack reports a new track from its second frame onward, so frame 0 has no player rows
- `person` includes referees, coaches and ball boys unless filtered or a soccer model is used
- Trajectories are in image pixels; camera panning and zoom distort them (no homography/camera-motion compensation)
- No team assignment, jersey numbers, speed/distance in metres, or possession
- Processing is offline; CPU throughput is far below real time

**Recommended next improvements (roughly in order of value)**
1. Soccer-specific detector weights for players and ball (largest accuracy gain)
2. Camera-motion compensation (BoT-SORT's CMC) for steadier trails
3. Team classification by jersey-colour clustering of player crops
4. Pitch keypoint detection + homography → positions in metres, speeds, heatmaps
5. Offline ball-trajectory smoothing/interpolation over the full video

## Project layout

```
soccer-tracker/
├── configs/            default.yaml, soccer_finetuned.example.yaml, experiments.yaml (experiment manifest)
├── docs/               FINAL_REPORT.md, PRESENTATION_NOTES.md, ARCHITECTURE.md
├── scripts/            run_experiments.py, make_report_tables.py, make_presentation.py,
│                       fetch_zip_subset.py, visualize_run.py, frames_to_video.py, make_synthetic_video.py
├── src/soccer_tracker/
│   ├── cli.py, __main__.py, config.py, errors.py
│   ├── video_io.py, pipeline.py, visualization.py, analytics.py, synthetic.py
│   ├── detection/      base.py, yolo.py, filters.py
│   ├── tracking/       player_tracker.py, factory.py (SORT / OC-SORT / ByteTrack), ball_tracker.py, types.py
│   ├── datasets/       sportsmot.py, uvy_track.py, infactory.py (one adapter per dataset)
│   ├── evaluation/     mot.py (trackers.eval wrapper), detection.py, ball.py (custom evaluators)
│   └── experiments/    manifest, detections (cache), players, ball, models, runtime, plots, database
├── tests/              unit, smoke, and integration tests
├── results/            metrics/, plots/, tracks/, demo/, tables.md, sports_analytics.db
├── data/               datasets (git-ignored; see DATASETS.md)
├── weights/            model files (git-ignored)
├── DATASETS.md, presentation.pptx
├── requirements.txt, requirements-dev.txt, pyproject.toml
```
