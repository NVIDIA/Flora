<div align="center">

# FLORA

**An Open Model Suite for Scaling Robotics Data Engines**

[![Project page](https://img.shields.io/badge/Project-page-76B900)](https://research.nvidia.com/labs/gear/flora/)
[![Paper](https://img.shields.io/badge/Paper-PDF-B31B1B)](https://research.nvidia.com/labs/gear/flora/assets/paper/flora.pdf)
[![Models on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Models-Hugging%20Face-FFD21E)](https://huggingface.co/collections/nvidia/flora)

<a href="https://research.nvidia.com/labs/gear/flora/assets/video/flora.mp4"><img src="assets/readme/flora-film-teaser.webp" width="100%" alt="Excerpt from the Flora film: the title, a timestamped instruction written for a guitar-repair video, a robot episode flagged 'Failure: drops the white packet', and an episode scored 1 out of 5 for not picking up the green block."></a>

▶ [Watch the full film](https://research.nvidia.com/labs/gear/flora/assets/video/flora.mp4) (67 s, sound on)

</div>

Flora turns robot and human manipulation videos into labels for robot learning:
**when each action happens, what it is, and how well it went.**

- **Flora-TempSeg** splits a video into actions.
- **Flora-Captioner** writes an instruction for each action.
- **Flora-Rewarder** judges an episode: did it succeed, where did it slip, and why.

This repository holds the inference code. The weights are on
[Hugging Face](https://huggingface.co/collections/nvidia/flora); there is no training code here.

## News

- **2026-10** Flora is released: [weights](https://huggingface.co/collections/nvidia/flora),
  [paper](https://research.nvidia.com/labs/gear/flora/assets/paper/flora.pdf),
  [project page](https://research.nvidia.com/labs/gear/flora/) and this inference code.

## Models and Code

**This repository provides sample vanilla inference code for the released
weights, not the production inference system.** The examples are intended
as a straightforward starting point for running and integrating FLORA.

For large-scale workloads, throughput can be improved with optimized inference
frameworks, dynamic batching, asynchronous video decoding and preprocessing,
and queue-based scheduling across persistent GPU workers. These optimizations
are left to downstream deployments; the sample scripts' runtime does not
represent production throughput.

| Component | Purpose | Weights | Inference guide |
|---|---|---|---|
| Temporal segmentation | Locate action intervals | [Flora-TempSeg](https://huggingface.co/nvidia/Flora-TempSeg) | [Segmentation and captioning](tempseg/batch_utils/README.md) |
| Action captioning | Describe each intended operation | [Flora-Captioner](https://huggingface.co/nvidia/Flora-Captioner) | [Segmentation and captioning](captioner/batch_utils/README.md) |
| Reward scoring | Evaluate outcomes, action quality, and rubric steps | [Flora-Rewarder](https://huggingface.co/nvidia/Flora-Rewarder) | [Rewarder](rewarder/batch_utils/README.md) |

Both `tempseg/` and `captioner/` contain the complete segmentation-to-caption
workflow. The annotation examples below use `captioner/`.

## Quick start

```bash
git clone https://github.com/NVIDIA/Flora.git && cd Flora
```

The annotation pipeline and the Rewarder pin different PyTorch builds, so each gets its own environment.
Weights download from Hugging Face on first use. Validated on Linux (x86_64) with Python 3.12 and NVIDIA
H100 80 GB GPUs.

### Timestamped instructions (TempSeg → Captioner)

```bash
python3.12 -m venv .venv-annotation && source .venv-annotation/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r captioner/batch_utils/requirements.txt

python captioner/infer.py --video my_video.mp4 --profile human --validate-only   # CPU check, no download
python captioner/infer.py --video my_video.mp4 --profile human --output my_video.json
```

The output is one JSON file per video, with times in seconds from the start of the video. For the
guitar-repair video in the film (107 s, 20 actions), two of the actions, both matching the human
annotation (abridged):

```json
{"schema_version": "flora_demo_annotation_v1", "duration_s": 106.933333, "profile": "Human",
 "segments": [
  {"start_s": 56.5, "end_s": 58.1, "caption": "pick up guitar string with both hands", "status": "complete"},
  {"start_s": 58.1, "end_s": 63.6, "caption": "untangle guitar string with both hands", "status": "complete"}
 ]}
```

This sample code favors clarity over speed: it loads both models for every video and reads one window at
a time. The guitar video took 23 minutes on one H100, including the first download.

Robot with three synchronized cameras:

```bash
python captioner/infer.py --video main.mp4 --left-wrist left_wrist.mp4 --right-wrist right_wrist.mp4 \
  --profile yam --output robot.json
```

- **Profiles:** `human` for egocentric human video (one camera), `yam` for parallel grippers, `dex` for
  dexterous hands (`tianji` is a variant for Tianji robots). Robot profiles take one camera or all three.
- **Limits:** at most 120 s, 256 MiB and 4096 px per side per video, at 4 FPS or more. Split longer episodes.
  Wrist videos must share the main video's frame timestamps.
- **Many videos:** `python captioner/infer.py --input-jsonl jobs.jsonl --output-dir annotations/`
  ([job format](captioner/batch_utils/README.md#jsonl-jobs)).
- **Storage:** about 130 GB of weights, in `~/.cache/flora/models` and the Hugging Face cache; set
  `FLORA_MODEL_CACHE` and `HF_HOME` to move them.

### Rewards (Rewarder)

```bash
python3.12 -m venv .venv-rewarder && source .venv-rewarder/bin/activate
pip install -r rewarder/batch_utils/requirements.txt

# Did it succeed? One score from 1 (failure) to 5 (success), with a one-line reason.
python -m rewarder.batch_utils.terminal_reward --model nvidia/Flora-Rewarder \
  --video episode.mp4 --task "Put the block in the silver bowl" --with-rationale --output outcome.jsonl

# Where did it slip? A score every 0.5 s (3 normal, 2 suboptimal, 1 failure) with reasons.
python -m rewarder.batch_utils.dense_reward --model nvidia/Flora-Rewarder \
  --video left_wrist.mp4 top.mp4 right_wrist.mp4 --task "Put the cup on the tray." --output dense.jsonl

# Score each step of your own rubric (2 GPUs).
python -m rewarder.batch_utils.rubric_reward --model nvidia/Flora-Rewarder \
  --jobs rubric_jobs.jsonl --output rubric.jsonl
```

Each command writes one JSON line per episode. The outcome for the [example](#examples) episode
(abridged; the human label is also 5):

```json
{"id": "episode", "task": "Put the block in the silver bowl", "score": 5, "expected_score": 5.0, "progress": 1.0,
 "probabilities": {"1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 1.0},
 "rationale": "The robot grasped the blue block, moved it over the silver bowl, and released it inside the bowl."}
```

- **Cameras:** pass all views of one episode in order; they are tiled into one frame. Two views: left,
  right. Three: left, center, right (for example left wrist, top, right wrist). Four: top, front, left,
  right. Rubric jobs list one description per camera instead
  ([format](rewarder/batch_utils/README.md#rubric-reward), [template](rewarder/batch_utils/examples/rubric_jobs.jsonl)).
- **Many episodes:** every command takes `--jobs jobs.jsonl`, one episode per line:
  `{"id": "ep001", "task": "Put the cup on the tray.", "videos": ["left_wrist.mp4", "top.mp4", "right_wrist.mp4"]}`.
- **First run:** downloads 56 GB and compiles GPU kernels for a few minutes. `--dry-run` prints the
  prompt without loading the model.
- Dense scores look a few seconds past each moment, so use them for offline labeling and filtering, not as
  an online signal.

Full options: `--help` on any command, the [annotation guide](captioner/batch_utils/README.md) and the
[Rewarder guide](rewarder/batch_utils/README.md).

## Examples

Excerpts with Flora's predictions. The outcome, dense and rubric scores agree with the human labels for
these episodes.

<table>
<tr>
<th width="33%">Human video</th>
<th width="33%">Parallel grippers</th>
<th width="33%">Dexterous hands</th>
</tr>
<tr>
<td><a href="https://research.nvidia.com/labs/gear/flora/#explore"><img src="assets/readme/caption-human.gif" width="320" alt="Human cleaning a glass surface while holding a bottle."></a></td>
<td><a href="https://research.nvidia.com/labs/gear/flora/#explore"><img src="assets/readme/caption-parallel.gif" width="320" alt="Robot placing a cucumber on a plate."></a></td>
<td><a href="https://research.nvidia.com/labs/gear/flora/#explore"><img src="assets/readme/caption-dexterous.gif" width="320" alt="Dexterous robot hands peeling a bandage backing."></a></td>
</tr>
<tr valign="top">
<td><b>15.85–37.25 s</b><br>“hold bottle with left hand, scrape glass with utility knife in right hand”</td>
<td><b>8.80–12.80 s</b><br>“Put the cucumber on the plate with the right hand.”</td>
<td><b>11.50–27.73 s</b><br>“Peel the backing from the bandage with the left hand while holding the bandage with the right hand.”</td>
</tr>
<tr>
<th>Outcome reward</th>
<th>Dense reward</th>
<th>Rubric reward</th>
</tr>
<tr>
<td><a href="https://research.nvidia.com/labs/gear/flora/#explore"><img src="assets/readme/reward-outcome.gif" width="320" alt="Robot moving a blue block into a metal bowl."></a></td>
<td><img src="assets/readme/reward-process.gif" width="320" alt="A gripper lowers a toy drill onto a toy car; the bit lands beside the screw head."></td>
<td><a href="https://research.nvidia.com/labs/gear/flora/#explore"><img src="assets/readme/reward-rubric.gif" width="320" alt="Robot placing test tubes in a holder."></a></td>
</tr>
<tr valign="top">
<td><b>Success: 5 / 5</b><br>“Put the block in the silver bowl”<br>Reason: “The robot grasped the blue block, moved it over the silver bowl, and released it inside the bowl.”</td>
<td><b>Failure: 1 / 3</b>, 188.2–199.9 s<br>“… the toy drill is misaligned and contacts the gray screw head without engaging the screw”</td>
<td><b>Total: 1.00 / 1.00</b><br>Four placement steps, each scored 0.25 / 0.25.</td>
</tr>
</table>

## How it works

<a href="assets/readme/figure-7.png"><img src="assets/readme/figure-7.png" width="100%" alt="Figure 7: Video-to-supervision overview, with timestamped captions and dense, outcome, or rubric rewards."></a>

<details>
<summary><b>TempSeg → Captioner: from a video to timestamped instructions</b></summary>

<br>

TempSeg reads overlapping windows of the video (16 s at 4 frames per second, or 32 s with `--yam32`) and
finds the actions in the middle of each window. Fragments from neighboring windows are stitched into
episode-level intervals; concurrent actions keep overlapping intervals. The Captioner then writes one
instruction per interval, looking at the interval itself and at the video just before and after it.

<a href="assets/readme/figure-13.png"><img src="assets/readme/figure-13.png" width="100%" alt="Figure 13: Windowed temporal localization, fragment stitching, contextual captioning, and timestamped action instructions."></a>

With three cameras, the synchronized views of each moment are tiled into one image: main camera above,
left and right wrist below.

<a href="assets/readme/figure-14.png"><img src="assets/readme/figure-14.png" width="100%" alt="Figure 14: Synchronized main and wrist views form mosaics sampled in before, target, and after blocks."></a>

Prompts, sampling and model inputs: [TempSeg card](tempseg/README.md), [Captioner card](captioner/README.md).

</details>

<details>
<summary><b>Rewarder: three rewards from one model</b></summary>

<br>

The prompt selects the reward. All camera views of a moment are tiled into one frame for the outcome
and dense prompts; the rubric prompt reads each camera as its own video.

| Reward | Model input | Output |
|---|---|---|
| Outcome | Whole episode, up to 64 frames (10 per second for short episodes) | Score 1–5, `P(1..5)`, expected score, optional reason |
| Dense | 16 s clips of 64 frames at 4 FPS, one every 10 s, stitched | Score per 0.5 s, a reason for every non-normal span |
| Rubric | Each camera at 2 FPS (up to 166 frames), plus your rubric | Thinking trace, then one score per step (medoid of 9 samples) |

<a href="assets/readme/figure-15.png"><img src="assets/readme/figure-15.png" width="100%" alt="Figure 15: One rewarder produces an episode outcome score, dense quality labels with rationales, or scores for custom rubric steps."></a>

The model reads absolute episode times, so pass the native videos without resampling. Details:
[Rewarder card](rewarder/README.md).

</details>

## Results

From the [paper](https://research.nvidia.com/labs/gear/flora/assets/paper/flora.pdf) (Tables 1, 4 and 5).
Arrows mark the better direction.

| Benchmark | Metric | Flora | GPT-6 Astra | Best open-weight general model |
|---|---|---|---|---|
| WGO-Bench | Segmentation F1@0.75 ↑ | **50.3** | 46.0 | 26.1 |
| WGO-Bench | Gold-window caption accuracy ↑ | 64.7 | **77.4** | 50.6 |
| RoboRewardBench | Outcome MAE ↓ | **40.0** | 55.4 | 68.2 |
| RobometerBench (OOD) | Kendall τa ↑ | 72.0 | **82.3** | 63.5 |
| DenseRewardBench | Failure recall ↑ | **53.3** | 0.0 | 5.7 |
| RubricBench | Pooled L1 error ↓ | 6.5 | **3.6** | 8.9 |

Flora annotates video at an estimated $0.72 per hour of video (eight GB300 GPUs at $10 per GPU-hour),
about 125× less than GPT-6 Astra at its highest reasoning effort.

## Use with coding agents

[AGENTS.md](AGENTS.md) tells Claude Code, Codex, Cursor and other coding agents how to set up the two
environments, check inputs without a GPU, run each model and read the JSON it writes. Try:

> Read AGENTS.md, then add timestamped instructions to every video in `./videos`, one JSON file per video.

## Troubleshooting

- **`decoded fewer frames than its container reports`:** the MP4 hides frames with an edit list, which
  `ffmpeg -ss … -c copy` trims create. Re-encode it: `ffmpeg -i in.mp4 -c:v libx264 -crf 18 -an out.mp4`.
- **`Video exceeds 120 seconds`:** split the video; each part is annotated on its own clock.
- **Arm64 machines (GH200, GB200):** `decord` in the Rewarder requirements has no arm64 wheel; only x86_64
  is validated.

## Intended use

Flora is for non-commercial research. Its outputs are annotations: do not use them to control a robot or
as the only safety monitor of a physical system. Instructions are in English. Check the rewards on
held-out episodes from your own robot before training on them.

## Citation

```bibtex
@article{leng2026flora,
  title   = {Flora: An Open Model Suite for Scaling Robotics Data Engines},
  author  = {Sicong Leng and Yunze Man and Zhiqi Li and Jing Wang and
             Yuhe Zhang and Ryan Punamiya and Siddharth Patki and Kaiyuan Zheng and
             Zi-ang Cao and Zhiding Yu and Sifei Liu and Danfei Xu and Ben Levin and
             Jan Kautz and Yuke Zhu and Johan Bjorck and Linxi "Jim" Fan},
  year    = {2026}
}
```

## License

This repository is governed by the [NVIDIA License](LICENSE) for non-commercial
research and evaluation. Refer to each weight repository for its model terms and
to the third-party notices ([annotation](captioner/batch_utils/THIRD_PARTY_NOTICES.md),
[Rewarder](rewarder/batch_utils/THIRD_PARTY_NOTICES.md)) for dependency licenses.
