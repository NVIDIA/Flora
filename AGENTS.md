# AGENTS.md

Guide for coding agents (Claude Code, Codex, Cursor, Copilot, …) working with this repository.

Flora is inference code for three video models that label robot and human manipulation videos:
Flora-TempSeg (when each action happens), Flora-Captioner (one instruction per action) and Flora-Rewarder
(success score, action quality every 0.5 s, rubric scores). There is no training code. Weights download from
Hugging Face on first use.

## Layout

| Path | Contents |
|---|---|
| `captioner/infer.py`, `captioner/batch_utils/` | TempSeg → Captioner pipeline (CLI and Python API) |
| `tempseg/` | The same pipeline, as shipped in the Flora-TempSeg repository; use `captioner/` |
| `rewarder/batch_utils/` | `terminal_reward.py` (outcome), `dense_reward.py`, `rubric_reward.py` |
| `*/README.md` | Model cards; `*/batch_utils/README.md` are the detailed guides |
| `*/batch_utils/vendor/`, `models.json`, `CODE_MANIFEST.json` | Pinned, checksummed code and model revisions; do not edit |

## Setup

Two environments; never mix them (they pin different PyTorch builds). Linux x86_64, Python 3.12, NVIDIA GPU
with 80 GB (H100 validated); the rubric reward needs two GPUs.

```bash
python3.12 -m venv .venv-annotation && . .venv-annotation/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r captioner/batch_utils/requirements.txt
deactivate
python3.12 -m venv .venv-rewarder && . .venv-rewarder/bin/activate
pip install -r rewarder/batch_utils/requirements.txt
```

Downloads: annotation about 130 GB (to `$FLORA_MODEL_CACHE`, default `~/.cache/flora/models`, and the
Hugging Face cache, `$HF_HOME`); Rewarder 56 GB (Hugging Face cache). The repositories are public; `HF_TOKEN`
is optional.

## Commands

Run from the repository root.

| Goal | Environment | Command |
|---|---|---|
| Check a video on CPU, no download | annotation | `python captioner/infer.py --video v.mp4 --profile human --validate-only` |
| Timestamped instructions | annotation | `python captioner/infer.py --video v.mp4 --profile human --output v.json` |
| Robot, three cameras | annotation | `python captioner/infer.py --video main.mp4 --left-wrist l.mp4 --right-wrist r.mp4 --profile yam --output v.json` |
| Many videos | annotation | `python captioner/infer.py --input-jsonl jobs.jsonl --output-dir out/` |
| Success score 1–5 | rewarder | `python -m rewarder.batch_utils.terminal_reward --model nvidia/Flora-Rewarder --video v.mp4 --task "…" --with-rationale --output o.jsonl` |
| Quality every 0.5 s | rewarder | `python -m rewarder.batch_utils.dense_reward --model nvidia/Flora-Rewarder --video v.mp4 --task "…" --output d.jsonl` |
| Rubric scores (2 GPUs) | rewarder | `python -m rewarder.batch_utils.rubric_reward --model nvidia/Flora-Rewarder --jobs rubric.jsonl --output r.jsonl` |
| Print a Rewarder prompt, no GPU | rewarder | add `--dry-run` to any Rewarder command (downloads 30 MB of processor files) |

Profiles: `human` (one egocentric camera), `yam` (parallel grippers), `dex` (dexterous hands), `tianji`
(dexterous hands, Tianji robots). `--yam32` uses 32 s windows and needs `yam` with all three cameras.

## Inputs

- Pass the native videos. Do not resample, trim or re-encode them unless a command fails (see Troubleshooting).
- Annotation: each video at most 120 s, 256 MiB and 4096 px per side, at least 4 FPS, one video stream. Split
  longer videos and offset each part's times. Wrist videos must have the main video's frame timestamps, frame
  count and duration.
- Annotation jobs (JSONL, paths relative to the file):
  `{"id": "robot01", "video": "main.mp4", "left_wrist": "l.mp4", "right_wrist": "r.mp4", "profile": "yam"}`;
  optional `yam32` and `instruction`.
- Rewarder jobs (JSONL, paths relative to the file): `{"id": "ep001", "task": "Put the cup on the tray.",
  "videos": ["left_wrist.mp4", "top.mp4", "right_wrist.mp4"]}`. `task` is one plain English instruction. Camera
  order: 2 views left, right; 3 views left, center, right; 4 views top, front, left, right. Views must share frame
  rate and frame count.
- Rubric jobs add `robot`, `cameras` (one description per video, same order) and `rubric`: a list of
  `{"action": …, "grading": [{"label": …, "points": …}]}` whose step maxima sum to 1. Template:
  `rewarder/batch_utils/examples/rubric_jobs.jsonl` (its video paths are placeholders).

## Outputs

- `infer.py` writes one JSON file per video. `segments` lists `{start_s, end_s, caption, status,
  segmentation_provenance}` in seconds from the start of the video. Segments may overlap (concurrent actions).
  `status` is `complete` when a caption was written; any other value means `caption` is empty. Also `duration_s`,
  `profile`, `camera_roles`, `model_revisions`, `input_sha256`. Stdout has `FLORA_RUNTIME {json}` event lines and
  a final `{"output": …, "segments": N}` line; errors exit with code 1 and `FLORA inference failed: …` on stderr.
- `terminal_reward`: `score` (1–5), `probabilities` (`P(1..5)`), `expected_score`, `progress` (0–1, for ranking,
  not a success probability), `rationale` (null without `--with-rationale`).
- `dense_reward`: `segments` (`start_s`, `end_s`, `score` 1–3, `label`, `rationale`), `per_second`, `fraction`,
  `worst_score`, `mean_score`, `clips` (raw answers) and `invalid_clips` (clips whose answer did not parse and
  were scored normal).
- `rubric_reward`: `scores` (one per step, null when no sample parsed), `total`, `parsed_samples`, `samples`;
  `medoid_sample` is the reported sample.

## Rules

- Always run `--validate-only` (annotation) or `--dry-run` (Rewarder) on new data before a GPU run.
- Report problems instead of hiding them: segments whose `status` is not `complete`, `invalid_clips > 0`,
  `parsed_samples` below `--num-samples`.
- Ask the user before downloading weights to a new disk, starting multi-GPU jobs, processing more than a few
  hours of video, or re-encoding their videos.
- Never change prompts, frame sampling, image sizes, decoding settings or the pinned revisions in `models.json`.
  The models were trained and evaluated with them; other values give worse results without any error.
- Never commit videos, outputs, weights or tokens. Never put a Hugging Face token in code or command lines.
- Non-commercial research only: the code is under the NVIDIA License in `LICENSE`, the weights under the
  terms on each model's Hugging Face page. Outputs are annotations, not robot controls.

## Troubleshooting

- `decoded fewer frames than its container reports` (Rewarder): the MP4 hides frames with an edit list
  (common after `ffmpeg -ss … -c copy`). Ask, then re-encode: `ffmpeg -i in.mp4 -c:v libx264 -crf 18 -an out.mp4`.
- `Video exceeds 120 seconds or 7200 frames` (annotation): split the video.
- CUDA out of memory: the models need a whole 80 GB GPU; check `nvidia-smi` for other processes.
- The first Rewarder run compiles GPU kernels for several minutes; that is not a hang.
- Annotation is slow by design: TempSeg spends about 4 minutes on the CPU after `Loading weights`, and both
  models load again for every video. A 107 s video took 23 minutes on one H100 (first run, with downloads).
- `pip` cannot find `decord` on arm64: only x86_64 is validated.
