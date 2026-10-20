# FLORA Rewarder: example inference

Example scripts for the three reward prompts of the FLORA Rewarder. One set of weights answers
all three; the prompt selects the reward.

| Script | Reward | Model input | Output |
| --- | --- | --- | --- |
| `terminal_reward.py` | Episode success, 1–5 | Whole episode, up to 64 frames at up to 10 FPS, camera collage | Score, `P(1..5)`, expected score |
| `dense_reward.py` | Action quality, 1–3 per 0.5 s | Overlapping 16 s clips, 64 frames at 4 FPS, camera collage | Score per frame, segments, rationales for bad spans |
| `rubric_reward.py` | Rubric step scores | One video per camera, 2 FPS (up to 166 frames each), rubric | Thinking trace, one score per step |

Scores: terminal `1` = complete failure, `5` = complete success; dense `1` = failure,
`2` = suboptimal, `3` = normal.

## Setup

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use. See [third-party notices](THIRD_PARTY_NOTICES.md).

Validated with Python 3.12 and NVIDIA H100 80 GB GPUs: one GPU for terminal and dense rewards,
two for rubric rewards with the default settings (`--tensor-parallel-size 2`).

```bash
pip install -r batch_utils/requirements.txt
```

Run the scripts from a downloaded copy of this repository, the directory that contains
`batch_utils/`; they load the weights stored next to it. To load the weights from elsewhere,
pass `--model` with a Hub repository id or a local directory, or set `FLORA_REWARDER_REPO`.
The first run compiles GPU kernels and can take several minutes before scoring starts.
`--dry-run` reads the videos and prints the first prompt without loading the model.

## Inputs

Each episode is one line of a JSONL file; video paths are relative to that file. Camera videos
of one episode must be synchronized: same frame rate and frame count, starting together.

```json
{"id": "ep001", "task": "Put the cup on the tray.", "videos": ["left_wrist.mp4", "top.mp4", "right_wrist.mp4"]}
```

`task` is one plain-language instruction for the whole episode.

**Camera order (terminal and dense).** The views of each time step are tiled into one frame:
one view as is, two or three side by side, four as a 2 × 2 grid, in the order given. Use the
order the model was trained with:

| Views | Order |
| --- | --- |
| 2 | left camera, right camera (for example a left exterior camera and the wrist camera) |
| 3 | left, center, right (for example left wrist, top, right wrist) |
| 4 | top, front or bottom, left, right |

A single episode can also be passed on the command line:

```bash
python -m batch_utils.terminal_reward --video left_wrist.mp4 top.mp4 right_wrist.mp4 \
  --task "Put the cup on the tray." --output terminal.jsonl
```

## Terminal reward

```bash
python -m batch_utils.terminal_reward --jobs batch_utils/examples/jobs.jsonl --output terminal.jsonl
```

Frames are sampled uniformly over the whole episode: `min(64, floor(10 × duration in s))`,
rounded down to an even count. Add `--with-rationale` for a one-line explanation before the
score. Each output line has:

- `score`: the generated score, 1–5;
- `probabilities`: `P(1..5)` at the score token, and `expected_score`, their mean;
- `progress`: `(expected_score − 1) / 4`, a value in [0, 1] for ranking, not a calibrated
  success probability.

## Dense reward

```bash
python -m batch_utils.dense_reward --jobs batch_utils/examples/jobs.jsonl --output dense.jsonl
```

The episode is split into 16 s clips (64 frames at 4 FPS) that start every 10 s; the last clip
ends at the episode end. The model scores the 32 frame pairs (0.5 s each) of every clip and
writes a rationale for every span scored 1 or 2. Clips are stitched by dropping the last 4 s
of every clip except the final one and giving each frame the score of the nearest kept pair.

Each output line has `segments` (start and end time, score, rationale), `per_second` (the score
at the middle of each second), `fraction` of time per score, and the raw answer of every clip.
`--frame-scores` adds one score per source frame. Clips whose answer cannot be parsed are
scored normal and counted in `invalid_clips`.

Dense scores are retrospective: a clip also shows the seconds after each scored moment. Use
them for offline labeling and data filtering, not as an online signal.

## Rubric reward

```bash
python -m batch_utils.rubric_reward --jobs batch_utils/examples/rubric_jobs.jsonl --output rubric.jsonl
```

A rubric job also names the robot and describes every camera, in video order:

```json
{"id": "ep001", "task": "Put the cup on the tray.", "robot": "bimanual robot",
 "cameras": ["a top (scene) camera", "a left-wrist camera", "a right-wrist camera"],
 "videos": ["top.mp4", "left_wrist.mp4", "right_wrist.mp4"],
 "rubric": [
   {"action": "Grasp the cup", "grading": [
     {"label": "Not attempted", "points": 0.0},
     {"label": "Grasped after re-tries", "points": 0.1},
     {"label": "Clean grasp", "points": 0.2}]},
   {"action": "Set the cup on the tray", "grading": [
     {"label": "Not attempted", "points": 0.0},
     {"label": "Placed but tipped over", "points": 0.4},
     {"label": "Placed upright", "points": 0.8}]}]}
```

Each step lists the scores that may be awarded; the step maxima should sum to 1. Instead of
`robot` and `cameras`, a job may give `header`, the full opening paragraph of the prompt.

The model thinks before it answers `ANSWER: [...]`. The script samples `--num-samples` answers
(default 9) and reports the medoid: the answer with the smallest summed L1 distance to the
others. Use `--num-samples 1` for a single answer. Traces are often several thousand tokens;
`--keep-thinking` writes them to the output.

## Notes

- Terminal and dense answers are decoded greedily with the answer format enforced; rubric
  answers are sampled at temperature 1.
- Videos are read at their native frame rate, and the model receives absolute episode times.
  Keep the native videos; do not resample them.
- Validate reward mappings on held-out episodes from your own robot before using them for
  training or data selection.
