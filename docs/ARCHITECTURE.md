# System architecture

## Data flow (one frame)

```
                 ┌──────────────┐
 input video ──▶ │ VideoReader  │  validates file, decodes frames, computes timestamps
                 └──────┬───────┘
                        │ (frame_idx, t, frame)
          ┌─────────────┴──────────────────────────────┐
          ▼                                            ▼
 ┌──────────────────┐                        ┌──────────────────┐
 │ Player detector  │  YOLO "person"         │ Ball detector    │  YOLO "sports ball" or custom
 │ (YoloDetector or │  (one shared pass when │ (+ optional      │  weights; optional tiled
 │  ClassView)      │   configs are equal)   │   tiling)        │  inference for tiny balls
 └────────┬─────────┘                        └────────┬─────────┘
          ▼                                           ▼
 ┌──────────────────┐   optional PitchMasker  ┌──────────────────┐
 │ filter_players   │◀──── (HSV grass mask) ─▶│ filter_balls     │  size / aspect / on-pitch
 └────────┬─────────┘                         └────────┬─────────┘
          ▼                                            ▼
 ┌──────────────────┐                        ┌──────────────────┐
 │ PlayerTracker    │  ByteTrack (Roboflow   │ BallTracker      │  constant-velocity Kalman,
 │                  │  `trackers`)           │                  │  gating, coasting, re-acquire
 └────────┬─────────┘                        └────────┬─────────┘
          └──────────────┬─────────────────────────────┘
                         ▼  list[TrackedObject]
          ┌──────────────┴──────────────┐
          ▼                             ▼
 ┌──────────────────┐          ┌──────────────────┐
 │ Visualizer       │          │ TrackCsvWriter   │  streamed rows; RunStats -> summary.json
 └────────┬─────────┘          └──────────────────┘
          ▼
 ┌──────────────────┐
 │ VideoWriter      │ ──▶ annotated video
 └──────────────────┘
```

`Pipeline` (src/soccer_tracker/pipeline.py) owns this loop and nothing else.

## Modules

| Module | Responsibility | Depends on |
|---|---|---|
| `config.py` | Typed dataclass config, YAML loading, strict validation (unknown keys rejected) | PyYAML |
| `errors.py` | Exception hierarchy; every expected failure is a `SoccerTrackerError` | – |
| `video_io.py` | `VideoReader` / `VideoWriter` context managers; input validation; FPS fallback | OpenCV |
| `detection/base.py` | `Detections` container, `Detector` protocol, IoU, NMS, tiling geometry | NumPy |
| `detection/yolo.py` | Ultralytics wrapper, weight resolution and download policy, class-name lookup, tiled inference, shared inference pass | Ultralytics (lazy import) |
| `detection/filters.py` | Size/aspect filters, HSV pitch mask | OpenCV |
| `tracking/player_tracker.py` | ByteTrack adapter (the only file touching the `trackers` API) | trackers, supervision |
| `tracking/ball_tracker.py` | Single-target Kalman tracker and trajectory buffer | NumPy |
| `visualization.py` | Boxes, IDs, trails, ball marker, HUD | OpenCV |
| `analytics.py` | CSV export, run statistics, JSON summary | stdlib |
| `pipeline.py` | Orchestration, resource lifetime, error wrapping | all of the above |
| `cli.py` | Argument parsing, CLI overrides, exit codes | config, pipeline |
| `synthetic.py` | Deterministic synthetic clip with ground truth (tests and I/O checks) | OpenCV |

## Key design decisions

**Detector-agnostic core.** Trackers, filters, visualisation, and export only see
`Detections` / `TrackedObject`. PyTorch and Ultralytics are imported lazily inside
`detection/yolo.py`, so most tests run without them and a different detector
(RT-DETR, a Roboflow model, ONNX) only needs a class with `detect(frame) -> Detections`.

**Detectors are injected.** `Pipeline(cfg, detector_factory=...)` takes a factory.
Production uses `build_yolo_detectors`; the smoke test passes a ground-truth replay
detector, which is how the full pipeline is tested without weights.

**Class names, not class ids.** Configs list `classes: [person]` or `[ball]`. The
detector maps names through `model.names`, so swapping COCO weights for a
fine-tuned soccer model is a config change. A wrong name fails at start-up with the
list of valid names.

**Two different trackers.** Players are many similar objects that need global
assignment → ByteTrack. The ball is a single, small, frequently missed object →
a dedicated tracker that decides *which candidate is the ball* using a motion model:

- predict with a constant-velocity Kalman filter;
- accept the best candidate within a gate around the prediction
  (score = confidence × (1 − distance/gate)); the gate widens while the ball is missing;
- otherwise coast for up to `max_missed_frames`, emitting `status=predicted`;
- re-initialise when a confident detection appears outside the gate after a few
  misses (long passes, camera cuts), or drop the track if the prediction leaves the frame;
- report only after `min_hits` detections, to suppress one-frame false positives.

**Shared inference.** With default COCO weights the player and ball detectors would
run the same network twice. When weights, `imgsz`, device and tiling allow it, one
pass serves both (`ClassView` objects filter by class and per-detector confidence).
Measured ≈2× faster on CPU in this environment.

**Fail early, fail clearly, never leave handles open.** Order of operations in
`Pipeline.run`: validate output paths → open and decode the first input frame →
load models → only then create output files. All OpenCV handles live in context
managers. Unexpected exceptions inside the loop are re-raised as `ProcessingError`
carrying the frame index; the CLI prints one line and exits with code 2.

## Extension points

| Change | Where |
|---|---|
| Different detector / weights | config `weights` + `classes`; or a new `Detector` class and factory |
| Better ball detection | `ball.detector` weights, `tiling`; ball tracker is unaffected |
| Different MOT algorithm (BoT-SORT, OC-SORT) | `tracking/player_tracker.py` only (`trackers` ships several) |
| Pitch coordinates (homography) | new stage after tracking, before analytics |
| Team assignment (jersey colour clustering) | new stage after player tracking |
| Extra analytics (speed, distance, possession) | consume `tracks.csv` or extend `analytics.py` |
