"""Rubric reward: a score for every step of a user-defined rubric.

    python -m batch_utils.rubric_reward --jobs rubric_jobs.jsonl --output rubric.jsonl

Each job gives the camera videos (each camera is a separate video, in the order the camera
descriptions list them), the robot, the task instruction and the rubric:

    {"id": "ep01", "task": "Put the cup on the tray.",
     "robot": "bimanual robot",
     "cameras": ["a top (scene) camera", "a left-wrist camera", "a right-wrist camera"],
     "videos": ["top.mp4", "left_wrist.mp4", "right_wrist.mp4"],
     "rubric": [{"action": "Grasp the cup",
                 "grading": [{"label": "Not attempted", "points": 0.0},
                             {"label": "Grasped after re-tries", "points": 0.1},
                             {"label": "Clean grasp", "points": 0.2}]}, ...]}

Instead of ``robot`` and ``cameras``, a job may give ``header``: the full opening paragraph
that describes the robot and Video 1..N (ending with a blank line). The step maxima should sum
to 1. The model thinks before answering. Several answers are sampled per episode and the final
scores are the medoid answer (the answer closest, in summed L1 distance, to all others).
"""

from __future__ import annotations

import os

from . import core
from .engine import base_parser, load_llm, load_processor, log, open_output, read_jobs, render, resolve_model, write_record


def video_inputs(messages):
    """Decode every camera video at the rubric frame rate and pixel budget."""
    if "FORCE_QWENVL_VIDEO_READER" not in os.environ:
        try:
            import decord  # noqa: F401

            os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"
        except ImportError:
            pass
    from qwen_vl_utils import process_vision_info

    _, videos, _ = process_vision_info(
        [messages], return_video_kwargs=True, image_patch_size=16, return_video_metadata=True
    )
    inputs = []
    for frames, metadata in videos:
        metadata = dict(metadata)
        metadata.setdefault("duration", metadata["total_num_frames"] / metadata["fps"])
        metadata["do_sample_frames"] = False
        inputs.append((frames.numpy(), metadata))
    return inputs


def main(argv=None) -> int:
    p = base_parser("FLORA rewarder: rubric scores with thinking.", tensor_parallel_size=2)
    p.add_argument("--num-samples", type=int, default=9, help="Answers sampled per episode; the medoid is reported.")
    p.add_argument("--fps", type=float, default=core.RUBRIC_FPS, help="Frames per second read from each camera.")
    p.add_argument("--max-frames", type=int, default=core.RUBRIC_MAX_FRAMES, help="Frame cap per camera.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=20480)
    p.add_argument("--max-model-len", type=int, default=98304)
    p.add_argument("--max-num-batched-tokens", type=int, default=24576)
    p.add_argument("--keep-thinking", action="store_true", help="Write every sample's thinking trace.")
    p.set_defaults(gpu_memory_utilization=0.85)
    args = p.parse_args(argv)
    jobs = read_jobs(args, extra_fields=("robot", "cameras", "rubric", "task_name", "header"))
    for job in jobs:
        if "rubric" not in job or not ({"robot", "cameras"} <= set(job) or "header" in job):
            raise SystemExit(f"{job['id']}: rubric jobs need 'rubric' and 'robot' + 'cameras' (or 'header')")
        if "cameras" in job and len(job["cameras"]) != len(job["videos"]):
            raise SystemExit(f"{job['id']}: give one camera description per video")
    model_path = resolve_model(args.model, args.revision, weights=not args.dry_run)
    processor = load_processor(model_path)

    def prepare(job):
        header = job.get("header") or core.rubric_header(job["robot"], job["cameras"])
        prompt = core.rubric_prompt(job["task"], job["rubric"], header, job.get("task_name"))
        messages = core.rubric_messages(prompt, job["videos"], fps=args.fps, max_frames=args.max_frames)
        return render(processor, messages, thinking=True, vision_ids=True), messages

    if args.dry_run:
        text, messages = prepare(jobs[0])
        frames = [(tuple(f.shape), m["fps"]) for f, m in video_inputs(messages)]
        log(f"{jobs[0]['id']}: camera frames {frames}\n{text}")
        return 0

    from vllm import SamplingParams

    llm = load_llm(model_path, args, max_model_len=args.max_model_len,
                   videos_per_prompt=max(len(job["videos"]) for job in jobs),
                   mm_processor_kwargs={"do_resize": False},
                   max_num_batched_tokens=args.max_num_batched_tokens)
    params = SamplingParams(n=args.num_samples, temperature=args.temperature, top_p=1.0, top_k=0,
                            max_tokens=args.max_new_tokens)
    with open_output(args) as stream:
        for number, job in enumerate(jobs, 1):
            text, messages = prepare(job)
            request = {"prompt": text, "multi_modal_data": {"video": video_inputs(messages)}}
            output = llm.generate([request], params, use_tqdm=False)[0]
            samples = []
            for completion in output.outputs:
                answer, thinking = core.split_thinking(completion.text)
                scores, ok = core.parse_rubric(answer, job["rubric"])
                sample = {"scores": scores, "parsed": ok, "answer": answer,
                          "generated_tokens": len(completion.token_ids)}
                if args.keep_thinking:
                    sample["thinking"] = thinking
                samples.append(sample)
            parsed = [i for i, s in enumerate(samples) if s["parsed"]]
            best = parsed[core.medoid([samples[i]["scores"] for i in parsed])] if parsed else None
            scores = samples[best]["scores"] if best is not None else None
            write_record(stream, {
                "id": job["id"],
                "task": job["task"],
                "steps": [step["action"] for step in job["rubric"]],
                "scores": scores,
                "total": round(sum(scores), 4) if scores is not None else None,
                "medoid_sample": best,
                "parsed_samples": len(parsed),
                "samples": samples,
                "prompt_tokens": len(output.prompt_token_ids),
            })
            log(f"rubric: {number}/{len(jobs)} episodes, {len(parsed)}/{len(samples)} answers parsed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
