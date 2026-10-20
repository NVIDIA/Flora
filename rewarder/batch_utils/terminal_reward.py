"""Terminal reward: one 1-5 success score for each whole episode.

    python -m batch_utils.terminal_reward --jobs jobs.jsonl --output terminal.jsonl
    python -m batch_utils.terminal_reward --video top.mp4 wrist.mp4 --task "Put the cup on the tray." --output terminal.jsonl

Each job is ``{"id": ..., "task": ..., "videos": [camera videos, in collage order]}``.
"""

from __future__ import annotations

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


def main(argv=None) -> int:
    p = base_parser("FLORA rewarder: terminal (episode success) score, 1-5.")
    p.add_argument("--with-rationale", action="store_true", help="Ask for a one-line rationale before the score.")
    p.add_argument("--max-model-len", type=int, default=16384)
    args = p.parse_args(argv)
    jobs = read_jobs(args)
    model_path = resolve_model(args.model, args.revision, weights=not args.dry_run)
    processor = load_processor(model_path)

    prepared = []
    for job in jobs:
        collage = core.Collage(job["videos"])
        indices = core.terminal_frame_indices(collage.num_frames, collage.fps)
        messages = core.terminal_messages(job["task"], args.with_rationale)
        prepared.append((job, collage, indices, render(processor, messages, thinking=False)))
    if args.dry_run:
        job, collage, indices, prompt = prepared[0]
        log(f"{job['id']}: {collage.num_frames} frames at {collage.fps:g} FPS, "
            f"collage {collage.width}x{collage.height}, {len(indices)} sampled frames\n{prompt}")
        return 0

    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = load_llm(model_path, args, max_model_len=args.max_model_len,
                   mm_processor_kwargs={"do_sample_frames": False, "size": COLLAGE_VIDEO_SIZE},
                   max_logprobs=core.TERMINAL_TOP_LOGPROBS)
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(
        temperature=0.0,
        max_tokens=core.terminal_max_tokens(args.with_rationale),
        logprobs=core.TERMINAL_TOP_LOGPROBS,
        structured_outputs=StructuredOutputsParams(regex=core.terminal_regex(args.with_rationale)),
    )
    with open_output(args) as stream:
        for start in range(0, len(prepared), args.max_num_seqs):
            batch = prepared[start : start + args.max_num_seqs]
            requests = [
                {
                    "prompt": prompt,
                    "multi_modal_data": {
                        "video": (collage.frames(indices), core.video_metadata(indices, collage.fps, collage.num_frames))
                    },
                    "mm_processor_kwargs": {"do_sample_frames": False},
                }
                for _, collage, indices, prompt in batch
            ]
            outputs = llm.generate(requests, params, use_tqdm=False)
            for _, collage, _, _ in batch:
                collage.release()
            for (job, collage, indices, _), output in zip(batch, outputs):
                completion = output.outputs[0]
                score, rationale = core.parse_terminal(completion.text, args.with_rationale)
                positions = [
                    (
                        tokenizer.decode([token]),
                        float(candidates[token].logprob),
                        [(tokenizer.decode([i]), float(c.logprob)) for i, c in candidates.items()],
                    )
                    for token, candidates in zip(completion.token_ids, completion.logprobs)
                ]
                probabilities, expected = core.score_probabilities(positions, score)
                write_record(stream, {
                    "id": job["id"],
                    "task": job["task"],
                    "score": score,
                    "expected_score": round(expected, 4),
                    # Expected score rescaled to [0, 1]: a progress value, not a success probability.
                    "progress": round((expected - 1.0) / 4.0, 4),
                    "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
                    "rationale": rationale,
                    "duration_s": round(collage.duration_s, 3),
                    "sampled_frames": len(indices),
                    "output": completion.text,
                })
            log(f"terminal: {min(start + len(batch), len(prepared))}/{len(prepared)} episodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
