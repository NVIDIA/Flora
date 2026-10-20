# FLORA Data Engine inference

This directory contains the complete **TempSeg → Captioner** workflow, identical
in both model repositories. Use either copy. TempSeg predicts action intervals;
the Captioner describes those predicted intervals. The native windowing,
adaptive retries, stitching, prompts, source timestamps, and camera layouts are
included; no internal source checkout or running demo is required.

## Setup

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use. See [third-party notices](THIRD_PARTY_NOTICES.md).

Python 3.12 and a CUDA GPU with 80 GB VRAM are the validated configuration.
The two models run sequentially. Download/cache space of at least 150 GB is
recommended for the released FP32 Captioner, base model, and adapter.

From a downloaded copy of either repository:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r batch_utils/requirements.txt
hf auth login
```

Both private model repositories must be accessible to the logged-in account.
`HF_TOKEN` is also supported; do not put a token in source code or command-line
arguments. Weights are fetched automatically at the revisions in `models.json`.
Set `FLORA_MODEL_CACHE` to choose the local model directory.

The NVIDIA repositories are selected by default. To override them:

```bash
export FLORA_TEMPSEG_REPO="nvidia/Flora-TempSeg"
export FLORA_CAPTIONER_REPO="nvidia/Flora-Captioner"
```

## Single video

```bash
python infer.py --video videos/human.mp4 --profile human --output human.json
```

Three synchronized robot cameras:

```bash
python infer.py --video videos/top.mp4 \
  --left-wrist videos/left.mp4 --right-wrist videos/right.mp4 \
  --profile yam --output yam.json
```

Use `--profile dex` for Dex or `--profile tianji` for Tianji inputs.
YAM defaults to native 16-second temporal windows; use `--yam32` for the native
32-second YAM profile. Human uses one video. Robot profiles accept one view or
all three; `--yam32` requires all three. Wrist files must share the main video's
relative frame timestamps, frame count, and duration. Inputs are not resampled.

The validated input limits are 120 seconds, 256 MiB per file, 4096 pixels per
side, 7200 frames, and at least 4 source FPS. Variable-rate video is supported
by the single-view Human path. `--instruction` optionally supplies an episode
task; leave it blank for the default workflow.

CPU-only media validation performs no inference or model downloads:

```bash
python -m batch_utils --video videos/human.mp4 --profile human --validate-only
```

## JSONL jobs

Each row specifies one job. Camera paths are relative to the JSONL file:

```json
{"id":"human01","video":"videos/human.mp4","profile":"human"}
{"id":"robot01","video":"videos/top.mp4","left_wrist":"videos/left.mp4","right_wrist":"videos/right.mp4","profile":"yam","yam32":true}
```

```bash
python infer.py --input-jsonl jobs.jsonl --output-dir results
```

Jobs and predicted segments run sequentially; JSONL is not GPU batch scheduling.
Existing outputs are preserved unless `--overwrite` is explicit. An error stops
the command; successfully written earlier jobs remain available.

## Python usage

```python
from batch_utils import DataEnginePipeline

engine = DataEnginePipeline()
result = engine.annotate("videos/human.mp4", profile="human")
for segment in result["segments"]:
    print(segment["start_s"], segment["end_s"], segment["caption"])
```

Outputs contain clip-relative seconds, caption completion status, camera roles,
input SHA-256, and model revisions. Native overlapping intervals are retained.
`include_audit=True` / `--include-audit` adds temporal and caption-input records.
The code preserves FP32 LoRA/GDN values and validates checkpoint manifests.
Full manifest verification happens on first use in the process; reuse requires
unchanged manifest bytes and all file identities/timestamps. FP32 parameter
values are checked on every load. First-time downloads and verification are
substantial; later jobs reuse the verified local files.

Example output (illustrative):

```json
{"segments":[{"start_s":0.8,"end_s":3.2,"caption":"Pick up the cup with the right hand.","status":"complete"}]}
```
