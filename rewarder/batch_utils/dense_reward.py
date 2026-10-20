"""Dense reward: one 1-3 action-quality score per 0.5 s, with rationales for bad spans.

    python -m batch_utils.dense_reward --jobs jobs.jsonl --output dense.jsonl
    python -m batch_utils.dense_reward --video top.mp4 wrist.mp4 --task "Put the cup on the tray." --output dense.jsonl

Each job is ``{"id": ..., "task": ..., "videos": [camera videos, in collage order]}``.
Scores: 1 = failure, 2 = suboptimal, 3 = normal. The episode is scored in overlapping 16 s
clips that are stitched into one score per source frame. Scoring is retrospective (a clip
sees 4 s past each kept moment), so use it for offline labeling, not as an online signal.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from . import core
from .engine import (
    COLLAGE_VIDEO_SIZE,
    base_parser,
    load_llm,
    load_processor,
    log,
    open_output,
    read_jobs,
    render,
    resolve_model,
    write_record,
)


def summarize(job, collage, windows, answers, texts, tail_trim_s, include_frames):
    scores, clip_of_frame, group_of_frame = core.stitch_dense(windows, answers, collage.num_frames, tail_trim_s)
    counts = Counter(scores.tolist())
    record = {
        "id": job["id"],
        "task": job["task"],
        "fps": collage.fps,
        "num_frames": collage.num_frames,
        "duration_s": round(collage.duration_s, 3),
        "worst_score": int(scores.min()),
        "mean_score": round(float(scores.mean()), 4),
        "fraction": {core.LABELS[s]: round(counts.get(s, 0) / collage.num_frames, 4) for s in (1, 2, 3)},
        "segments": core.dense_segments(scores, clip_of_frame, group_of_frame, answers, collage.fps),
        # One score per second of video, read at the middle of each second.
        "per_second": [int(scores[min(collage.num_frames - 1, int((t + 0.5) * collage.fps))])
                       for t in range(int(np.ceil(collage.duration_s)))],
        "clips": [
            {
                "start_s": round(window.start_s, 3),
                "end_s": round(window.end_s, 3),
                "quality": "".join(map(str, answer.scores)),
                "regions": core.window_regions(window, answer),
                "valid": answer.valid,
                "output": text,
            }
            for window, answer, text in zip(windows, answers, texts)
        ],
        "invalid_clips": sum(not a.valid for a in answers),
    }
    if include_frames:
        record["frame_scores"] = scores.tolist()
    return record


def main(argv=None) -> int:
    p = base_parser("FLORA rewarder: dense action-quality scores, 1-3 per 0.5 s.")
    p.add_argument("--tail-trim", type=float, default=core.DENSE_TAIL_TRIM_S,
                   help="Seconds dropped from the end of every clip but the last when stitching.")
    p.add_argument("--frame-scores", action="store_true", help="Also write one score per source frame.")
    p.add_argument("--max-model-len", type=int, default=16384)
    args = p.parse_args(argv)
    jobs = read_jobs(args)
    model_path = resolve_model(args.model, args.revision, weights=not args.dry_run)
    processor = load_processor(model_path)

    if args.dry_run:
        job = jobs[0]
        collage = core.Collage(job["videos"])
        windows = core.dense_windows(collage.num_frames, collage.fps)
        prompt = render(processor, core.dense_messages(job["task"], windows[0]), thinking=False)
        log(f"{job['id']}: {collage.num_frames} frames at {collage.fps:g} FPS, collage "
            f"{collage.width}x{collage.height}, {len(windows)} clips\n{prompt}")
        return 0

    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = load_llm(model_path, args, max_model_len=args.max_model_len,
                   mm_processor_kwargs={"do_sample_frames": False, "size": COLLAGE_VIDEO_SIZE})
    params = SamplingParams(temperature=0.0, max_tokens=core.DENSE_MAX_TOKENS,
                            structured_outputs=StructuredOutputsParams(regex=core.dense_regex()))
    with open_output(args) as stream:
        for number, job in enumerate(jobs, 1):
            collage = core.Collage(job["videos"])
            windows = core.dense_windows(collage.num_frames, collage.fps)
            collage.frames(sorted({i for window in windows for i in window.frame_indices}))  # one decoding pass
            texts = []
            for start in range(0, len(windows), args.max_num_seqs):
                batch = windows[start : start + args.max_num_seqs]
                requests = [
                    {
                        "prompt": render(processor, core.dense_messages(job["task"], window), thinking=False),
                        "multi_modal_data": {
                            "video": (
                                collage.frames(window.frame_indices),
                                core.video_metadata(window.frame_indices, collage.fps, collage.num_frames),
                            )
                        },
                        "mm_processor_kwargs": {"do_sample_frames": False},
                    }
                    for window in batch
                ]
                texts += [out.outputs[0].text for out in llm.generate(requests, params, use_tqdm=False)]
            answers = [core.parse_dense(text) for text in texts]
            write_record(stream, summarize(job, collage, windows, answers, texts, args.tail_trim, args.frame_scores))
            collage.release()
            log(f"dense: {number}/{len(jobs)} episodes ({len(windows)} clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
