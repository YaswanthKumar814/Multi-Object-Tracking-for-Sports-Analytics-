# Presentation notes

Speaker notes for `presentation.pptx` (11 slides, about 10-12 minutes). The slides
are rebuilt from the stored results with `python scripts/make_presentation.py`, so
the numbers below match the slides as long as the experiments are not re-run.
If you re-run the experiments, read the numbers off the slides.

Before presenting: put your name, roll number and team members on slide 1
(or rebuild with `python scripts/make_presentation.py --author "Name 1, Name 2"`).
Have `results/demo/uvy_soccer_V04_annotated.mp4` open in a video player (VLC plays
it reliably) to show for 10 seconds on slide 9.

---

## Slide 1 - Title

"Our project is multi-object tracking for sports analytics: tracking every player
and the ball in soccer video. The system we propose is YOLO11n for detection,
ByteTrack for the players and a Kalman filter for the ball. We compare it with two
baseline trackers, SORT and OC-SORT, on three public datasets, about 4,850 labelled
frames in total, all processed on a laptop CPU."

## Slide 2 - Problem statement and objectives

* The problem has two parts: give every player a box and keep the *same ID* on the
  same player; and locate the ball, which is tiny, blurred and often hidden.
* A point we care about: when the ball is not detected and we estimate its
  position, we say so. A predicted position is never shown as a detection.
* Objectives: build the working pipeline, export video + CSV + JSON, compare with
  baselines fairly, evaluate with standard metrics, keep it reproducible.

## Slide 3 - Datasets used

* **SportsMOT**: professional broadcast, players labelled with track IDs. We use 4
  football sequences: 1 for tuning and 3 for testing.
* **UVY-Track**: user-generated YouTube sports video; players, referees,
  goalkeepers and the ball are labelled with track IDs. We use 3 soccer clips.
* **InFactory**: soccer broadcast frames with only the ball labelled, plus frames
  where the ball is not visible. 10 clips. No track IDs, so we use it for ball
  *detection* only.
* Each dataset has a different annotation format, so each has its own loader, and
  we never mix them.
* Honest detail if asked: we downloaded only the sequences we needed (0.73 GB
  instead of 11 GB) by reading the remote archives selectively.
* If asked about the UVY-Track link: the dataset is at Zenodo record 21303900.

## Slide 4 - Proposed system architecture

Walk left to right: video in, one YOLO11n pass per frame that finds both people
and the ball, simple size filters, then two different trackers. "Players are many
similar objects, so they need multi-object association: ByteTrack. The ball is a
single small object that is often missed, so it gets its own single-target Kalman
tracker. That combination is our hybrid proposal. For the baselines we swap only
the player-tracker block."

## Slide 5 - Methodology and tracking algorithms

* All three trackers predict each track with a Kalman filter and match boxes by
  overlap (IoU). They differ in how they use detection confidence:
  SORT is one matching stage; OC-SORT uses only confident boxes and corrects the
  filter after occlusion; ByteTrack matches confident boxes first and then uses
  the low-confidence boxes to keep existing tracks alive.
* Ball tracker: constant-velocity Kalman filter. It only accepts a candidate near
  where the ball should be, needs 2 detections before reporting, and predicts for
  up to 15 frames when the ball is not detected.
* **Fairness** (stress this, it is the 5-mark item): YOLO was run once and the
  detections were saved, so all three trackers got exactly the same boxes, frames
  and ground truth. Each tracker was tuned with the same 48-setting grid on one
  separate sequence.

## Slide 6 - Baseline comparison

Top table, SportsMOT (3 test sequences, 1,381 frames):

| | HOTA | MOTA | IDF1 | ID switches |
|---|---|---|---|---|
| SORT | 65.9 | 64.9 | 70.6 | 60 |
| OC-SORT | 59.3 | 55.3 | 61.2 | 106 |
| ByteTrack (proposed) | 62.6 | 52.6 | 66.8 | 51 |

What to say:

* "ByteTrack keeps identities best: 51 ID switches against 60 and 106, and it
  misses the fewest players. That is what it was designed for."
* "It is **not** the best on HOTA. Tuned SORT is higher, 65.9 against 62.6. The
  reason is false positives: ByteTrack keeps low-confidence boxes, and with a
  generic detector many of those are real people who are not labelled in the
  dataset, such as staff at the side of the pitch."
* "OC-SORT is the weakest here: about twice the ID switches."
* Bottom table, UVY-Track: "This dataset labels only some of the people in each
  clip, so false positives cannot be measured. We therefore use OWTA, a metric
  built for partially labelled data, and recall. On these ByteTrack is best:
  OWTA 56.8."
* If asked "then why is ByteTrack your proposed method?": it gives the most stable
  identities and highest recall, which matter most for per-player analytics, at
  the same speed; and we report honestly that SORT scores higher on HOTA here.
  With untuned default parameters the three are within 0.8 HOTA of each other.

## Slide 7 - Performance metrics and evaluation

* Player metrics: HOTA (overall), DetA and AssA (its detection and association
  parts), MOTA, IDF1, ID switches, FP, FN, fragmentation. Computed with the
  `trackers.eval` library, which reproduces the standard TrackEval tool.
* Ball metrics: precision, recall, F1, AP@0.5, coverage, centre error, continuity.
  These come from our own evaluator, which is unit-tested but is not an official
  benchmark tool. A ball output counts as correct when its centre is within one
  ball diameter of the true centre.
* Ball result: "Detection alone finds the ball in about a third of the frames
  where it is visible. Adding the Kalman tracker raises recall from 34.5 to 43.3
  percent and F1 from 45.4 to 49.3, and the ball track becomes much more
  continuous: 168 short pieces become 47 longer ones. The price is precision,
  from 66.4 down to 57.4, because only about 43 percent of the predicted positions
  are right."
* We report no identity metrics on InFactory because it has no ball IDs.

## Slide 8 - Graphs

Four plots; point at each briefly:

1. Detection vs association accuracy: SORT and ByteTrack have similar association
   accuracy; OC-SORT is lower.
2. ID switches and fragmentation: ByteTrack has the fewest switches on both datasets.
3. Per-sequence results: the ranking changes from sequence to sequence, which is
   why we do not over-claim.
4. Runtime: the trackers take 2-7 ms per frame; the detector takes about 290 ms.
   The full pipeline runs at about 3 FPS on this laptop CPU (about 1 FPS on battery).

## Slide 9 - Visual results

* Top left: a frame of the annotated output video: boxes, IDs, trails, yellow ball
  marker and the status line. **Play the demo video here for a few seconds.**
* Top right: player trajectories. Bottom left: player-position heatmap. Both are in
  image pixels, not pitch metres.
* Bottom right: ball trajectory against ground truth. Blue dots are detections,
  orange triangles are Kalman predictions, red marks are frames with no output.
  "You can see predictions that follow the ball and predictions that fly off in a
  straight line after a kick. That is exactly why we label them as predictions."

## Slide 10 - Limitations and future improvements

* The generic COCO detector is the main limit: it detects unlabelled people and
  misses most small balls.
* The proposed tracker is not best on every metric.
* Small evaluation subset, CPU only, about 3 FPS, no camera-motion compensation,
  no team identification.
* Future: a soccer-specific detector first (biggest gain), tiled inference for the
  ball, a better ball motion model, camera-motion compensation, team and jersey
  recognition, pitch homography for real distances, GPU for real time.

## Slide 11 - Conclusion

"We built a complete, tested player-and-ball tracking pipeline and evaluated it
on three datasets with a fair protocol. ByteTrack gave the most stable identities
and the best result on the partially labelled dataset, tuned SORT scored higher on
HOTA on SportsMOT, and OC-SORT was weakest. The Kalman ball tracker improved
recall, F1 and continuity at some cost in precision, and we always mark predicted
positions. The clearest next step is a soccer-specific detector. Everything is
reproducible with one command."

---

## Likely questions

**Why is MOTA negative in some of your result files?** On UVY-Track only a few
people per clip are labelled. The detector correctly finds the others, and MOTA
counts each of them as a false positive, so it drops below zero. It measures the
labelling, not the tracker, so we use OWTA there.

**Did you train anything?** No. All weights are pretrained (COCO). Only tracker
thresholds were chosen, on one separate sequence.

**Is this comparable with the SportsMOT leaderboard?** No. We used 3 test
sequences of the validation split and a generic detector. Leaderboard methods use
detectors fine-tuned on SportsMOT and the full test set.

**How do you know the evaluator is right?** Player metrics use an established
library. Our ball evaluator has unit tests with hand-computed answers
(`tests/test_evaluation.py`), and the tracking pipeline is checked with a case
that has exactly one known ID switch.

**Why is the Kalman prediction often wrong?** It assumes constant velocity. When
the ball is kicked or bounces during a detection gap, the filter keeps
extrapolating the old velocity. Fixes are listed under future work.

**Which YOLO version, and why?** YOLO11n. On our subset it beat YOLOv8n for
players (F1 81.2 against 71.7) and was about 15% faster; for the ball they were equal.

**How fast is it?** About 3 frames per second on an Intel Core 7 150U laptop CPU
on AC power; 87% of the time is the detector. On battery it drops to about 1 FPS.
