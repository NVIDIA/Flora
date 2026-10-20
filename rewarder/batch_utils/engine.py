"""Shared command-line plumbing and the offline vLLM engine."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable

# Without --model: $FLORA_REWARDER_REPO, else the downloaded copy of the repository that holds
# batch_utils/ (absolute, not resolved: Hub cache files are symlinks).
MODEL_ENV = "FLORA_REWARDER_REPO"
REPOSITORY_COPY = Path(__file__).absolute().parent.parent
# Pixel budget for collage clips (terminal and dense): at least ``shortest_edge`` and at most
# ``longest_edge`` pixels over all frames of one clip. Collages are planned inside this budget,
# so frames reach the model at their planned size.
# Configuration and tokenizer files: enough to render prompts for --dry-run.
PROCESSOR_FILES = ["*.json", "*.jinja", "*.txt"]
COLLAGE_VIDEO_SIZE = {"shortest_edge": 4096, "longest_edge": 29_360_128}


def default_model() -> str | None:
    if os.environ.get(MODEL_ENV):
        return os.environ[MODEL_ENV]
    if (REPOSITORY_COPY / "model.safetensors.index.json").is_file():
        return str(REPOSITORY_COPY)
    return None


def base_parser(description: str, *, tensor_parallel_size: int = 1) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--jobs", type=Path, help="JSONL file, one episode per line (paths relative to the file).")
    source.add_argument("--video", type=Path, nargs="+", help="Camera video(s) of one episode, in order.")
    p.add_argument("--task", help="Task instruction for --video.")
    p.add_argument("--output", type=Path, required=True, help="Output JSONL file.")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    p.add_argument("--model", default=default_model(),
                   help=f"Hub repository id or local directory (default: ${MODEL_ENV}, else this repository's copy).")
    p.add_argument("--revision", default=None, help="Hub revision (branch, tag or commit).")
    p.add_argument("--tensor-parallel-size", type=int, default=tensor_parallel_size)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    p.add_argument("--max-num-seqs", type=int, default=8, help="Concurrent requests in the engine.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="Read the videos and print the first prompt; no model.")
    return p


def read_jobs(args: argparse.Namespace, extra_fields: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Jobs from ``--jobs`` or from ``--video``/``--task``; video paths become absolute."""
    if args.video:
        if not args.task:
            raise SystemExit("--task is required with --video")
        return [{"id": "episode", "task": args.task, "videos": [str(p.expanduser().resolve()) for p in args.video]}]
    root = args.jobs.expanduser().resolve().parent
    allowed = {"id", "task", "videos", *extra_fields}
    jobs, ids = [], set()
    for number, line in enumerate(args.jobs.read_text().splitlines(), 1):
        if not line.strip():
            continue
        job = json.loads(line)
        if not isinstance(job, dict) or set(job) - allowed or not {"task", "videos"} <= set(job):
            raise SystemExit(f"{args.jobs}:{number}: expected keys {sorted(allowed)}")
        job.setdefault("id", f"episode_{number:06d}")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", str(job["id"])) or job["id"] in ids:
            raise SystemExit(f"{args.jobs}:{number}: invalid or duplicate id {job['id']!r}")
        ids.add(job["id"])
        if not isinstance(job["videos"], list) or not job["videos"]:
            raise SystemExit(f"{args.jobs}:{number}: 'videos' must be a non-empty list")
        job["videos"] = [str((root / Path(v).expanduser()).resolve()) for v in job["videos"]]
        jobs.append(job)
    if not jobs:
        raise SystemExit(f"{args.jobs} has no jobs")
    return jobs


def open_output(args: argparse.Namespace):
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists; pass --overwrite to replace it")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return args.output.open("w")


def write_record(stream, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()


def resolve_model(model: str | None, revision: str | None = None, *, weights: bool = True) -> str:
    """A local directory with the model; ``weights=False`` fetches only the processor files."""
    if not model:
        raise SystemExit(f"Pass --model (a Hub repository id or a local directory) or set {MODEL_ENV}.")
    path = Path(model).expanduser()
    if path.is_dir():
        return str(path.resolve())
    from huggingface_hub import snapshot_download

    return snapshot_download(model, revision=revision, allow_patterns=None if weights else PROCESSOR_FILES)


def load_processor(path: str):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(path)


def load_llm(
    path: str,
    args: argparse.Namespace,
    *,
    max_model_len: int,
    videos_per_prompt: int = 1,
    mm_processor_kwargs: dict[str, Any] | None = None,
    **engine_kwargs: Any,
):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from vllm import LLM

    return LLM(
        model=path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=args.max_num_seqs,
        limit_mm_per_prompt={"image": 0, "video": videos_per_prompt},
        mm_processor_kwargs=mm_processor_kwargs,
        # Triton prefill for the linear-attention layers: the configuration the reported
        # results were measured with.
        gdn_prefill_backend="triton",
        enable_prefix_caching=True,
        seed=args.seed,
        **engine_kwargs,
    )


def render(processor, messages, *, thinking: bool, vision_ids: bool = False) -> str:
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
        add_vision_id=vision_ids,
    )


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
