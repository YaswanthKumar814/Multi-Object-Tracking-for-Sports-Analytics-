# Individual Contributions

Project: **Multi-Object Tracking for Sports Analytics (Player and Ball Tracking)**
Course: Computer Vision, B.Tech Semester 7

## Team

| Member | Roll number | GitHub account | Primary responsibility |
|---|---|---|---|
| **P Yaswanth Kumar** | `[fill in]` | `YaswanthKumar814` (commits as *Yaswanth* / *P Yaswanth Kumar*) | Core tracking application, baseline comparison, datasets and evaluation framework |
| **Siddhartha** | `[fill in]` | `siddhartharavva` | Ball-in-flight tracking (motion flights), ball-tracker hardening, final results run, IEEE paper |

Faculty guide: Dr. Tripty Singh (listed as co-author in the IEEE paper).

Every row below is backed by the commit history of this repository; the
commit hashes are given so each claim can be checked with `git show <hash>`.
Summary commands: `git shortlog -sn` and `git log --author=<name> --stat`.

## Contribution by component

"Owner" is the member who designed and implemented the component and answers
for it in the viva. "Support" is the member who extended, tested or
re-ran it.

| # | Component | Files | Owner | Support | Evidence (commits) |
|---|---|---|---|---|---|
| 1 | Video input/output, validation, error handling | `src/soccer_tracker/video_io.py`, `errors.py`, `cli.py`, `config.py` | Yaswanth | - | `c60899c` |
| 2 | YOLO detection (shared player/ball pass, tiling, class lookup), detection filters | `detection/yolo.py`, `detection/base.py`, `detection/filters.py` | Yaswanth | - | `c60899c` |
| 3 | Player tracking with ByteTrack | `tracking/player_tracker.py` | Yaswanth | - | `c60899c` |
| 4 | Kalman ball tracker (state, gating, coasting, re-acquisition) | `tracking/ball_tracker.py` | Yaswanth | Siddhartha (gate cap, start confidence, coasting rules) | `c60899c`, `6acff26`, `dae6b41` |
| 5 | Visualization, tracking CSV and JSON summary | `visualization.py`, `analytics.py`, `pipeline.py` | Yaswanth | Siddhartha (delayed output for back-filling flights) | `c60899c`, `6acff26` |
| 6 | Ball in flight: camera-motion-compensated motion detection and flight linking | `detection/motion.py`, `tracking/flight.py` | Siddhartha | - | `6acff26`, `dae6b41` |
| 7 | Baseline trackers (SORT, OC-SORT) and the fair comparison protocol | `tracking/factory.py`, `experiments/players.py` | Yaswanth | - | `42e89e2` |
| 8 | Dataset adapters (SportsMOT, UVY-Track, InFactory) and subset download tool | `datasets/`, `scripts/fetch_zip_subset.py`, `DATASETS.md` | Yaswanth | - | `42e89e2` |
| 9 | Evaluation (HOTA/CLEAR/Identity via `trackers.eval`; custom ball and detection metrics) | `evaluation/` | Yaswanth | Siddhartha (motion-flight method in the ball evaluation) | `42e89e2`, `dae6b41` |
| 10 | Experiment runner, runtime benchmark, plots, SQLite database | `experiments/`, `scripts/run_experiments.py` | Yaswanth | Siddhartha (motion stages in runtime and plots) | `42e89e2`, `dae6b41` |
| 11 | Automated tests | `tests/` | Yaswanth (application, datasets, evaluation, experiments) | Siddhartha (`test_motion_flight.py`, new ball-tracker tests) | `c60899c`, `42e89e2`, `6acff26` |
| 12 | Experiment runs: first full run (Windows, Intel Core 7 150U) | `results/` | Yaswanth | - | `42e89e2` |
| 13 | Experiment runs: final regenerated results (macOS, Apple M1) and result videos | `results/`, `outputs/` | Siddhartha | - | `9240ea8`, `df8147c`, `561d66c`, `0eb22a8` |
| 14 | Report generation scripts (tables, LaTeX, slides) | `scripts/make_report_tables.py`, `make_latex_report.py`, `make_presentation.py` | Yaswanth | Siddhartha (table improvements, motion-flight rows) | `42e89e2`, `5037c1f`, `dae6b41`, `0eb22a8` |
| 15 | Project report (Markdown) and presentation | `docs/FINAL_REPORT.md`, `presentation.pptx`, `docs/PRESENTATION_NOTES.md` | Yaswanth | Siddhartha (regenerated with final results) | `42e89e2`, `5037c1f`, `8a57907`, `9240ea8` |
| 16 | IEEE paper (related work, author block, references, compiled PDF) | `docs/latex/`, `docs/FINAL_REPORT_IEEE.pdf` | Siddhartha | Yaswanth (first LaTeX version) | `5037c1f`, `35926a5`, `fe58ccb`, `58c95d8`, `df8147c` |

## Individual accountability

Each member is accountable in the viva for the components they own:

- **P Yaswanth Kumar**: components 1-3, 5, 7-12, 14 and 15. In particular: why
  ByteTrack and how it differs from SORT and OC-SORT; the Kalman ball tracker
  (state, F/H/Q/R, gating); the fairness protocol of the baseline comparison;
  how HOTA, MOTA, IDF1 and OWTA are computed; the dataset formats and why each
  dataset is used the way it is.
- **Siddhartha**: components 6, 13 and 16, and the changes to component 4. In
  particular: camera-motion compensation (ORB + RANSAC), three-frame
  differencing, how flights are linked and accepted, why a flight must be
  launched from the ball, what the motion stage costs and gains, and how the
  final results were produced.
- **Both** are expected to explain the overall architecture, the headline
  results and the limitations.

## Use of AI tools

An AI coding assistant (Claude, by Anthropic) was used during development.
Seven commits carry its `Co-Authored-By` trailer (`42e89e2`, `6acff26`,
`dae6b41`, `9240ea8`, `35926a5`, `fe58ccb`, `58c95d8`), which can be listed
with `git log --grep="Co-Authored-By: Claude"`. The member who made each of
these commits reviewed, ran and tested the result and takes responsibility
for it as listed above.
