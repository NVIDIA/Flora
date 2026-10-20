"""Inputs, prompts and output parsing for the FLORA rewarder.

One model answers three prompts:

* terminal: one 1-5 score for a whole episode (1 = complete failure, 5 = complete success);
* dense: one 1-3 action-quality score per 0.5 s (1 = failure, 2 = suboptimal, 3 = normal),
  with a short rationale for every non-normal span;
* rubric: a score for every step of a user-defined rubric, after a thinking trace.

This module has no inference backend. It builds the chat messages and the exact video
frames the model expects, and turns the generated text back into rewards.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image

VIDEO_PLACEHOLDER = "__video__"

# ---------------------------------------------------------------------------
# Video reading
# ---------------------------------------------------------------------------

FPS_RELATIVE_TOLERANCE = 0.01


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    num_frames: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.num_frames / self.fps


def probe_video(path: str | Path) -> VideoInfo:
    """Frame rate, frame count and size of the first video stream."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        if not rate or float(rate) <= 0:
            raise ValueError(f"{path}: the video has no usable frame rate")
        width, height = int(stream.codec_context.width), int(stream.codec_context.height)
        frames = sum(1 for packet in container.demux(stream) if packet.size)
    if frames < 1:
        raise ValueError(f"{path}: the video has no frames")
    return VideoInfo(str(path), float(rate), frames, width, height)


def read_frames(
    path: str | Path, indices: Sequence[int], size: tuple[int, int] | None = None
) -> dict[int, np.ndarray]:
    """Decode the requested frame indices as RGB uint8 arrays, optionally resized.

    Resizing uses bilinear interpolation to ``size`` (width, height), with no crop or padding.
    """
    import av

    wanted = sorted({int(i) for i in indices})
    out: dict[int, np.ndarray] = {}
    if not wanted:
        return out
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        pending = iter(wanted)
        target = next(pending)
        for position, frame in enumerate(container.decode(stream)):
            if position != target:
                continue
            image = frame.to_image()
            if size is not None and image.size != tuple(size):
                image = image.resize(tuple(size), resample=Image.Resampling.BILINEAR)
            out[position] = np.asarray(image.convert("RGB"), dtype=np.uint8)
            target = next(pending, None)
            if target is None:
                break
    if len(out) != len(wanted):
        raise ValueError(f"{path}: decoded fewer frames than its container reports")
    return out


def check_synchronized(videos: Sequence[VideoInfo]) -> tuple[float, int]:
    """Camera views must share one frame clock. Returns (fps, frame count)."""
    if not videos:
        raise ValueError("at least one camera video is required")
    fps = videos[0].fps
    for video in videos[1:]:
        if abs(video.fps - fps) > FPS_RELATIVE_TOLERANCE * fps:
            raise ValueError(f"camera videos have different frame rates: {fps} vs {video.fps}")
    counts = [video.num_frames for video in videos]
    if max(counts) - min(counts) > max(1, round(0.01 * max(counts))):
        raise ValueError(f"camera videos have different lengths: {sorted(set(counts))} frames")
    return fps, min(counts)


# ---------------------------------------------------------------------------
# Multi-view collage (terminal and dense prompts)
# ---------------------------------------------------------------------------
# All camera views of one time step are tiled into one frame: one view as is, two or three
# views side by side, four views as a 2 x 2 grid, in the order given. Every view is resized
# to a 32-pixel-aligned cell of about 100k pixels, with no crop, padding or gutter.

ALIGN = 32
TARGET_PIXELS_PER_VIEW = 100_000
MAX_VISUAL_TOKENS = 14_336
MAX_VIEWS = 4


def _aspect_distortion(source_width: int, source_height: int, width: int, height: int) -> float:
    return 100.0 * abs((width / height) / (source_width / source_height) - 1.0)


def _cell_loss(source: tuple[int, int], width: int, height: int) -> float:
    distortion = _aspect_distortion(*source, width, height)
    area_error = abs(math.log((width * height) / TARGET_PIXELS_PER_VIEW))
    over_limit = max(0.0, distortion - 4.0)
    return (distortion / 4.0) ** 2 + (area_error / 0.32) ** 2 + 10_000.0 * over_limit**2


def _candidate_widths(source: tuple[int, int], height: int) -> list[int]:
    center = round(source[0] / source[1] * height / ALIGN) * ALIGN
    return sorted(
        {
            center + offset * ALIGN
            for offset in range(-4, 5)
            if 2 * ALIGN <= center + offset * ALIGN <= 32 * ALIGN
        }
    )


def _heights() -> range:
    return range(4 * ALIGN, 17 * ALIGN, ALIGN)


def _tokens(width: int, height: int) -> int:
    return 32 * (width // ALIGN) * (height // ALIGN)


def _single_cell(source: tuple[int, int]) -> tuple[int, int]:
    candidates = [
        (_cell_loss(source, width, height), width, height)
        for height in _heights()
        for width in _candidate_widths(source, height)
    ]
    _, width, height = min(candidates)
    return width, height


def _row_cells(sources: Sequence[tuple[int, int]]):
    candidates = []
    for height in _heights():
        widths = [
            min(_candidate_widths(s, height), key=lambda w, s=s: _cell_loss(s, w, height))
            for s in sources
        ]
        loss = sum(_cell_loss(s, w, height) for s, w in zip(sources, widths))
        if _tokens(sum(widths), height) <= MAX_VISUAL_TOKENS:
            candidates.append((loss, [(w, height) for w in widths], sum(widths), height))
    if not candidates:
        raise ValueError("no side-by-side layout fits the visual token budget")
    _, cells, total_width, total_height = min(candidates)
    return cells, total_width, total_height


def _row_options(first: tuple[int, int], second: tuple[int, int]):
    options: dict[int, tuple[float, int, int, int]] = {}
    for height in _heights():
        for w1 in _candidate_widths(first, height):
            for w2 in _candidate_widths(second, height):
                loss = _cell_loss(first, w1, height) + _cell_loss(second, w2, height)
                current = options.get(w1 + w2)
                if current is None or loss < current[0]:
                    options[w1 + w2] = (loss, height, w1, w2)
    return options


def _grid_cells(sources: Sequence[tuple[int, int]]):
    top_options = _row_options(sources[0], sources[1])
    bottom_options = _row_options(sources[2], sources[3])
    candidates = []
    for total_width in top_options.keys() & bottom_options.keys():
        top, bottom = top_options[total_width], bottom_options[total_width]
        if _tokens(total_width, top[1] + bottom[1]) <= MAX_VISUAL_TOKENS:
            candidates.append((top[0] + bottom[0], total_width, top, bottom))
    if not candidates:
        raise ValueError("no 2 x 2 layout fits the visual token budget")
    _, total_width, top, bottom = min(candidates)
    (_, top_h, tl_w, tr_w), (_, bottom_h, bl_w, br_w) = top, bottom
    rectangles = [
        (0, 0, tl_w, top_h),
        (tl_w, 0, tr_w, top_h),
        (0, top_h, bl_w, bottom_h),
        (bl_w, top_h, br_w, bottom_h),
    ]
    return rectangles, total_width, top_h + bottom_h


@cache
def plan_collage(sources: tuple[tuple[int, int], ...]):
    """Return ``(rectangles, width, height)``: one ``(x, y, w, h)`` cell per view, in order."""
    if not 1 <= len(sources) <= MAX_VIEWS:
        raise ValueError(f"1 to {MAX_VIEWS} camera views are supported, got {len(sources)}")
    if len(sources) == 1:
        width, height = _single_cell(sources[0])
        return ((0, 0, width, height),), width, height
    if len(sources) in (2, 3):
        cells, width, height = _row_cells(sources)
        rectangles, x = [], 0
        for w, h in cells:
            rectangles.append((x, 0, w, h))
            x += w
        return tuple(rectangles), width, height
    rectangles, width, height = _grid_cells(sources)
    return tuple(rectangles), width, height


class Collage:
    """Synchronized camera videos composed into one collage video."""

    def __init__(self, paths: Sequence[str | Path]):
        if not paths:
            raise ValueError("at least one camera video is required")
        self.videos = [probe_video(path) for path in paths]
        self.fps, self.num_frames = check_synchronized(self.videos)
        sources = tuple((v.width, v.height) for v in self.videos)
        self.rectangles, self.width, self.height = plan_collage(sources)
        self._cache: dict[int, np.ndarray] = {}

    @property
    def duration_s(self) -> float:
        return self.num_frames / self.fps

    def release(self) -> None:
        """Drop decoded frames."""
        self._cache.clear()

    def frames(self, indices: Sequence[int]) -> np.ndarray:
        """``[T, H, W, 3]`` uint8 collage frames for source frame ``indices``."""
        missing = sorted({int(i) for i in indices} - set(self._cache))
        if missing:
            per_view = [
                read_frames(video.path, missing, size=(w, h))
                for video, (_, _, w, h) in zip(self.videos, self.rectangles)
            ]
            for index in missing:
                canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                for frames, (x, y, w, h) in zip(per_view, self.rectangles):
                    canvas[y : y + h, x : x + w] = frames[index]
                self._cache[index] = canvas
        return np.stack([self._cache[int(i)] for i in indices])


def video_metadata(indices: Sequence[int], fps: float, num_frames: int) -> dict[str, Any]:
    """Frame-clock metadata so the model's timestamps are absolute episode seconds."""
    return {
        "total_num_frames": int(num_frames),
        "fps": float(fps),
        "frames_indices": [int(i) for i in indices],
        "duration": float(num_frames) / float(fps),
        "do_sample_frames": False,
    }


def _messages(system: str, user: str) -> list[dict[str, Any]]:
    before, marker, after = user.partition("<video>")
    if not marker:
        raise AssertionError("user prompt lost its video marker")
    content: list[dict[str, Any]] = []
    if before.strip():
        content.append({"type": "text", "text": before.strip()})
    content.append({"type": "video", "video": VIDEO_PLACEHOLDER})
    if after.strip():
        content.append({"type": "text", "text": after.strip()})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def _compact(text: str) -> str:
    compact = " ".join(str(text or "").split())
    if not compact:
        raise ValueError("a non-empty task instruction is required")
    return compact


# ---------------------------------------------------------------------------
# Terminal reward: one 1-5 score per episode
# ---------------------------------------------------------------------------

TERMINAL_MAX_FRAMES = 64
TERMINAL_MAX_SAMPLE_FPS = 10.0
TERMINAL_TOP_LOGPROBS = 20

_TERMINAL_SYSTEM = """You are a terminal-success reward model for robot trajectories.
Given the task instruction and the complete trajectory video, predict a score from 1 to 5.
A complete failure is 1. A complete success is 5.
Judge only what is visible in the trajectory and follow the requested output format exactly."""


def terminal_frame_indices(num_frames: int, fps: float) -> list[int]:
    """Uniform whole-episode sampling: ``min(64, floor(duration * 10))`` frames, even, >= 2.

    The frame count follows the episode duration; keep it, the model relies on it.
    """
    if num_frames <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid timing: frames={num_frames}, fps={fps}")
    count = min(TERMINAL_MAX_FRAMES, max(2, math.floor(num_frames / fps * TERMINAL_MAX_SAMPLE_FPS)))
    count = max(2, count - count % 2)
    if num_frames == 1:
        return [0, 0]
    return [round(i * (num_frames - 1) / (count - 1)) for i in range(count)]


def terminal_messages(task: str, with_rationale: bool = False) -> list[dict[str, Any]]:
    if with_rationale:
        suffix = (
            "Provide one short, compact rationale followed by the score. Return exactly:\n"
            "Rationale: <rationale>\n\nScore: <1-5>"
        )
    else:
        suffix = "Do not provide a rationale. Return exactly:\nScore: <1-5>"
    user = (
        f"Task: {_compact(task)}\n\n"
        "Complete trajectory video:\n<video>\n\n"
        "Evaluate the terminal success of the task."
    )
    return _messages(f"{_TERMINAL_SYSTEM}\n{suffix}", user)


def terminal_regex(with_rationale: bool = False) -> str:
    return r"Rationale: [^\n]{1,512}\n\nScore: [1-5]" if with_rationale else r"Score: [1-5]"


def terminal_max_tokens(with_rationale: bool = False) -> int:
    return 160 if with_rationale else 12


def parse_terminal(text: str, with_rationale: bool = False) -> tuple[int, str | None]:
    if with_rationale:
        match = re.fullmatch(r"Rationale:\s*(.*?)\n\s*\nScore:\s*([1-5])\s*$", text.strip(), re.DOTALL)
        if match is None or not " ".join(match.group(1).split()):
            raise ValueError(f"unexpected terminal answer: {text!r}")
        return int(match.group(2)), " ".join(match.group(1).split())
    match = re.fullmatch(r"Score:\s*([1-5])\s*$", text.strip())
    if match is None:
        raise ValueError(f"unexpected terminal answer: {text!r}")
    return int(match.group(1)), None


def score_probabilities(
    positions: Sequence[tuple[str, float, Sequence[tuple[str, float]]]], score: int
) -> tuple[dict[str, float], float]:
    """``P(1..5)`` at the generated score token, and the expected score.

    ``positions`` has one ``(generated token text, its logprob, top candidates)`` entry per
    generated token, where top candidates are ``(token text, logprob)`` pairs.
    """
    for token, logprob, candidates in reversed(list(positions)):
        if token.strip() != str(score):
            continue
        mass = dict.fromkeys(range(1, 6), 0.0)
        seen = set()
        for text, value in [(token, logprob), *candidates]:
            if text.strip() in {"1", "2", "3", "4", "5"} and (text, value) not in seen:
                seen.add((text, value))
                mass[int(text.strip())] += math.exp(value)
        total = sum(mass.values())
        probabilities = {str(k): mass[k] / total for k in range(1, 6)}
        return probabilities, sum(k * probabilities[str(k)] for k in range(1, 6))
    raise ValueError("the generated score token is missing from the logprobs")


# ---------------------------------------------------------------------------
# Dense reward: one 1-3 score per 0.5 s
# ---------------------------------------------------------------------------
# An episode is covered by 16 s clips of 64 collage frames at 4 FPS. Frame position 9 is the
# clip anchor; consecutive anchors are 10 s apart, and the last clip ends at the episode end.
# The vision encoder merges each pair of frames, so a clip has 32 groups of 0.5 s, and the
# model scores every group. Clips overlap; stitching drops the last 4 s of every clip except
# the final one and gives every source frame the score of the nearest kept group.

DENSE_FRAMES = 64
DENSE_FPS = 4.0
DENSE_GROUPS = 32
DENSE_HISTORY = 8
DENSE_SCORED_SPAN = 40
DENSE_STRIDE_S = 10.0
DENSE_TAIL_TRIM_S = 4.0
DENSE_MAX_TOKENS = 640
MAX_RATIONALE_CHARS = 240
LABELS = {1: "failure", 2: "suboptimal", 3: "normal"}

_DENSE_SYSTEM = f"""You are a dense action-quality reward model for robot manipulation.
You watch a synchronized multiview clip and predict action quality at the finest temporal resolution visible to the model.

The clip contains exactly {DENSE_FRAMES} sampled frames at {DENSE_FPS:g} FPS. The vision encoder combines each consecutive pair of frames into one temporal group, so you must predict exactly {DENSE_GROUPS} ordered groups. Group 1 contains frames 1-2, group 2 contains frames 3-4, and group {DENSE_GROUPS} contains frames {DENSE_FRAMES - 1}-{DENSE_FRAMES}.
The clip is retrospective: a later recovery must not improve an earlier group and a later unrelated mistake must not degrade it.

Use exactly this score scale:
1 = failure: the action causes or clearly constitutes a major task failure, unsafe event, destructive mistake, or loss of task viability.
2 = suboptimal: the action is noticeably hesitant, inefficient, imprecise, or locally incorrect, but the task remains recoverable.
3 = normal: the action is appropriate, safe, and reasonably efficient for the visible situation.

Return the worst score before the temporal sequence. Worst must equal the minimum of all {DENSE_GROUPS} quality digits. Quality must contain exactly {DENSE_GROUPS} characters from 1, 2, or 3 with no spaces; character i is the score for group i.
For every contiguous group range scored 1 or 2, add one short rationale grounded only in visible evidence. Groups scored 3 get no rationale. Rationale ranges must exactly match the contiguous non-normal runs in Quality.

Return exactly:
Worst: <1, 2, or 3>
Quality:
<{DENSE_GROUPS} digits with no spaces>
Rationales:
<first group>-<last group> <short visible-evidence rationale>
<one line per contiguous group range scored 1 or 2, in time order>"""


@dataclass(frozen=True)
class DenseWindow:
    index: int
    anchor: int
    frame_indices: tuple[int, ...]
    real: tuple[bool, ...]
    is_last: bool
    fps: float

    @property
    def start_s(self) -> float:
        return self.frame_indices[0] / self.fps

    @property
    def end_s(self) -> float:
        return self.frame_indices[-1] / self.fps

    def group_frames(self, group: int) -> tuple[int, int]:
        """Source frames of 1-based ``group``."""
        return self.frame_indices[2 * group - 2], self.frame_indices[2 * group - 1]

    def group_valid(self, group: int) -> bool:
        """A group is valid when at least one of its frames is inside the episode."""
        return self.real[2 * group - 2] or self.real[2 * group - 1]

    def group_time_s(self, group: int) -> float:
        left, right = self.group_frames(group)
        return (left + right) / (2.0 * self.fps)


def dense_windows(num_frames: int, fps: float, stride_s: float = DENSE_STRIDE_S) -> list[DenseWindow]:
    if num_frames < 1:
        raise ValueError("episode length must be positive")
    stride = round(stride_s * fps)
    if stride < 1:
        raise ValueError(f"window stride collapses at {fps} FPS")
    last_scored = round((DENSE_SCORED_SPAN - 1) * fps / DENSE_FPS)
    maximum = max(0, num_frames - 1 - last_scored)
    anchors = list(range(0, maximum, stride))
    if not anchors or anchors[-1] != maximum:
        anchors.append(maximum)
    steps = range(-DENSE_HISTORY, DENSE_FRAMES - DENSE_HISTORY)
    windows = []
    for index, anchor in enumerate(anchors):
        positions = [round(anchor + step * fps / DENSE_FPS) for step in steps]
        raw = [anchor + round(step * fps / DENSE_FPS) for step in steps]
        windows.append(
            DenseWindow(
                index=index,
                anchor=anchor,
                frame_indices=tuple(min(num_frames - 1, max(0, value)) for value in positions),
                real=tuple(0 <= value < num_frames for value in raw),
                is_last=index == len(anchors) - 1,
                fps=float(fps),
            )
        )
    return windows


def dense_messages(task: str, window: DenseWindow) -> list[dict[str, Any]]:
    user = "\n".join(
        [
            f"Task: {_compact(task)}",
            "",
            "Synchronized multiview clip:",
            "<video>",
            "",
            (
                f"The clip contains {DENSE_FRAMES} sampled frame positions at {DENSE_FPS:g} FPS "
                f"({DENSE_FRAMES / DENSE_FPS:g} seconds nominal span) and {DENSE_GROUPS} visible "
                f"temporal groups of 2 frames each. The sampled positions run from "
                f"t={window.start_s:.3f} to t={window.end_s:.3f} seconds of the episode. "
                "Episode boundaries repeat the nearest available frame."
            ),
            f"Predict Worst, all {DENSE_GROUPS} group scores, and rationales for non-normal runs.",
        ]
    )
    return _messages(_DENSE_SYSTEM, user)


def dense_regex() -> str:
    numbers = sorted(range(1, DENSE_GROUPS + 1), key=lambda v: -len(str(v)))
    group = "(?:" + "|".join(str(v) for v in numbers) + ")"
    rationale = f"{group}-{group} [^\\n]{{1,{MAX_RATIONALE_CHARS}}}"
    return f"Worst: [123]\\nQuality:\\n[123]{{{DENSE_GROUPS}}}\\nRationales:(?:\\n{rationale})*"


@dataclass(frozen=True)
class DenseAnswer:
    scores: tuple[int, ...]  # one 1-3 score per group
    regions: tuple[tuple[int, int, int, str | None], ...]  # (first group, last group, score, rationale)
    valid: bool
    error: str | None = None
    unmatched_rationales: int = 0  # rationale lines whose span is not a non-normal run
    missing_rationales: int = 0  # non-normal runs without a rationale line


def _runs(values: Sequence[int]) -> list[tuple[int, int, int]]:
    """(start, end_exclusive, value) runs of equal values."""
    runs, start = [], 0
    for end in range(1, len(values) + 1):
        if end == len(values) or values[end] != values[start]:
            runs.append((start, end, int(values[start])))
            start = end
    return runs


def parse_dense(text: str) -> DenseAnswer:
    """Strict parse. An unparseable answer becomes all-normal with ``valid=False``."""
    try:
        lines = text.strip("\n").split("\n")
        if len(lines) < 4 or lines[1] != "Quality:" or lines[3] != "Rationales:":
            raise ValueError("missing Worst / Quality / Rationales headers")
        worst = re.fullmatch(r"Worst: ([123])", lines[0])
        if worst is None or re.fullmatch(rf"[123]{{{DENSE_GROUPS}}}", lines[2]) is None:
            raise ValueError("malformed Worst or Quality field")
        scores = tuple(int(c) for c in lines[2])
        if int(worst.group(1)) != min(scores):
            raise ValueError("Worst disagrees with Quality")
        given = {}
        for line in lines[4:]:
            match = re.fullmatch(r"(\d{1,2})-(\d{1,2}) (.+)", line)
            if match is None:
                raise ValueError(f"malformed rationale line: {line!r}")
            given[(int(match.group(1)), int(match.group(2)))] = " ".join(match.group(3).split())
    except ValueError as error:
        return DenseAnswer((3,) * DENSE_GROUPS, ((1, DENSE_GROUPS, 3, None),), False, str(error))
    runs = [(start + 1, end, score) for start, end, score in _runs(scores)]
    expected = {(first, last) for first, last, score in runs if score < 3}
    regions = tuple((first, last, score, given.get((first, last))) for first, last, score in runs)
    return DenseAnswer(scores, regions, True, None, len(set(given) - expected), len(expected - set(given)))


def window_regions(window: DenseWindow, answer: DenseAnswer) -> list[dict[str, Any]]:
    """Same-score group runs of one clip, with absolute times and rationales."""
    return [
        {
            "first_group": first,
            "last_group": last,
            "score": score,
            "label": LABELS[score],
            "start_s": round(window.group_time_s(first), 3),
            "end_s": round(window.group_time_s(last), 3),
            "rationale": rationale,
        }
        for first, last, score, rationale in answer.regions
    ]


def stitch_dense(
    windows: Sequence[DenseWindow],
    answers: Sequence[DenseAnswer],
    num_frames: int,
    tail_trim_s: float = DENSE_TAIL_TRIM_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-source-frame scores from overlapping clips.

    Every clip except the last drops its final ``tail_trim_s`` seconds. Each source frame then
    takes the score of the nearest kept group; ties prefer the group farther from its clip's
    edges, then the earlier clip. Returns ``(scores, clip index, group)`` per source frame.
    """
    if tail_trim_s < 0 or (tail_trim_s * DENSE_FPS) % 2:
        raise ValueError("tail trim must be a non-negative multiple of 0.5 s")
    trim_groups = round(tail_trim_s * DENSE_FPS / 2)
    if trim_groups >= DENSE_GROUPS:
        raise ValueError("tail trim removes every group")
    rows = []  # (frame, -interiority, clip, group, score)
    for window, answer in zip(windows, answers, strict=True):
        kept = DENSE_GROUPS if window.is_last else DENSE_GROUPS - trim_groups
        for group in range(1, kept + 1):
            if not window.group_valid(group):
                continue
            interiority = min(group - 1, DENSE_GROUPS - group)
            for frame in sorted(set(window.group_frames(group))):
                rows.append((frame, -interiority, window.index, group, answer.scores[group - 1]))
    if not rows:
        raise ValueError("no kept groups cover the episode")
    rows.sort(key=lambda row: (row[1], row[2], row[3]))
    frames = np.asarray([row[0] for row in rows], dtype=np.int64)
    best = np.empty(num_frames, dtype=np.int64)
    for start in range(0, num_frames, 4096):
        query = np.arange(start, min(num_frames, start + 4096), dtype=np.int64)
        best[start : start + len(query)] = np.argmin(np.abs(frames[None, :] - query[:, None]), axis=1)
    pick = [rows[i] for i in best]
    return (
        np.asarray([row[4] for row in pick], dtype=np.int8),
        np.asarray([row[2] for row in pick], dtype=np.int32),
        np.asarray([row[3] for row in pick], dtype=np.int32),
    )


def dense_segments(
    scores: np.ndarray,
    clip_of_frame: np.ndarray,
    group_of_frame: np.ndarray,
    answers: Sequence[DenseAnswer],
    fps: float,
) -> list[dict[str, Any]]:
    """Episode-level runs (end-exclusive frames) with the rationale covering most frames."""
    segments = []
    for start, end, score in _runs(scores.tolist()):
        rationale = None
        if score < 3:
            votes: Counter[str] = Counter()
            for frame in range(start, end):
                answer, group = answers[int(clip_of_frame[frame])], int(group_of_frame[frame])
                for first, last, _, text in answer.regions:
                    if first <= group <= last and text:
                        votes[text] += 1
                        break
            if votes:
                rationale = max(sorted(votes), key=lambda text: (votes[text], text))
        segments.append(
            {
                "start_frame": start,
                "end_frame": end,
                "start_s": round(start / fps, 3),
                "end_s": round(end / fps, 3),
                "score": score,
                "label": LABELS[score],
                "rationale": rationale,
            }
        )
    return segments


# ---------------------------------------------------------------------------
# Rubric reward: a score for every step of a user-defined rubric
# ---------------------------------------------------------------------------
# Each camera is a separate video ("Video 1", "Video 2", ...), not a collage. The model
# thinks first, then answers ``ANSWER: [s_1, ..., s_n]`` with one allowed value per step.

RUBRIC_SYSTEM = "You are a helpful assistant."
RUBRIC_FPS = 2.0
RUBRIC_MAX_FRAMES = 166
RUBRIC_MIN_PIXELS = 64 * 32 * 32
RUBRIC_MAX_PIXELS = 128 * 32 * 32
RUBRIC_TOTAL_PIXELS = 224 * 1024 * 32 * 32
_COUNT = {1: "one video", 2: "two time-synchronized videos", 3: "three time-synchronized videos",
          4: "four time-synchronized videos"}

_RUBRIC_INSTRUCTIONS = (
    "Score how well the robot performed, using the rubric below. The rubric lists the task's "
    "steps; each step has its own maximum number of points and an explicit list of the point "
    "values you may award for it. For every step, award exactly one of the values listed for "
    "that step -- not a value in between.\n"
)
_RUBRIC_FOOTER = (
    "\nRespond with your scores as a JSON array of numbers -- one score per step, in the "
    "exact order listed above (a length-{n} array). Each number must be one of the values "
    "allowed for that step. Output the array on a single line prefixed with 'ANSWER:' and "
    "nothing after it, e.g.\nANSWER: {example}\n"
)


def rubric_header(robot: str, cameras: Sequence[str]) -> str:
    """Describe the robot and every camera, in video order.

    Example: ``rubric_header("bimanual robot", ["a top (scene) camera", "a left-wrist camera",
    "a right-wrist camera"])``.
    """
    if not 1 <= len(cameras) <= 4:
        raise ValueError("1 to 4 camera videos are supported")
    views = [f"Video {i} is {_compact(text)}" for i, text in enumerate(cameras, 1)]
    if len(views) > 1:
        views[-1] = "and " + views[-1]
    return (
        "You are an expert evaluator of robot manipulation. You are shown "
        f"{_COUNT[len(cameras)]} of a {_compact(robot)} performing one task: {', '.join(views)}.\n\n"
    )


def _steps(rubric: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    steps = []
    for item in rubric:
        grading = [{"label": str(g["label"]), "points": float(g["points"])} for g in item["grading"]]
        if not grading:
            raise ValueError("every rubric step needs at least one allowed value")
        maximum = float(item.get("max_score", max(g["points"] for g in grading)))
        steps.append({"action": _compact(item["action"]), "max_score": maximum, "grading": grading})
    if not steps:
        raise ValueError("the rubric has no steps")
    return steps


def rubric_prompt(
    task: str,
    rubric: Sequence[Mapping[str, Any]],
    header: str,
    task_name: str | None = None,
) -> str:
    """The text that follows the camera videos.

    ``rubric`` is a list of steps: ``{"action": str, "grading": [{"label": str, "points": float},
    ...]}`` with optional ``max_score`` (default: the largest allowed value). The step maxima
    should sum to 1.
    """
    steps = _steps(rubric)
    instruction = _compact(task)
    lines = [header + _RUBRIC_INSTRUCTIONS, f"Task: {_compact(task_name or instruction)}"]
    lines.append(f'Instruction to the robot: "{instruction}"')
    lines.append("\nScoring rubric -- award one of the listed values for each step:")
    for i, step in enumerate(steps, 1):
        lines.append(f"  {i}. {step['action']} (max {step['max_score']:g})")
        tiers = " | ".join(f"{g['label']} = {g['points']:.2f}" for g in step["grading"])
        lines.append(f"       award one of: {tiers}")
    example = json.dumps([round(step["max_score"], 3) for step in steps])
    lines.append(_RUBRIC_FOOTER.format(n=len(steps), example=example))
    return "\n".join(lines)


def rubric_messages(prompt: str, video_paths: Sequence[str | Path], fps: float = RUBRIC_FPS,
                    max_frames: int = RUBRIC_MAX_FRAMES) -> list[dict[str, Any]]:
    """Chat messages in the format read by the video loader of ``qwen_vl_utils``."""
    video = {"min_pixels": RUBRIC_MIN_PIXELS, "max_pixels": RUBRIC_MAX_PIXELS,
             "total_pixels": RUBRIC_TOTAL_PIXELS, "max_frames": max_frames, "fps": fps}
    content = [{"type": "video", "video": str(path), **video} for path in video_paths]
    content.append({"type": "text", "text": prompt})
    return [{"role": "system", "content": RUBRIC_SYSTEM}, {"role": "user", "content": content}]


def split_thinking(text: str) -> tuple[str, str | None]:
    """``(answer, thinking)``; the answer is empty when generation stopped inside the trace."""
    before, closed, after = text.partition("</think>")
    if not closed:
        return "", before.replace("<think>", "", 1).strip() or None
    return after.strip(), before.replace("<think>", "", 1).strip() or None


def parse_rubric(answer: str, rubric: Sequence[Mapping[str, Any]]) -> tuple[list[float], bool]:
    """Scores after the last ``ANSWER:``, aligned to the steps and clamped to ``[0, max]``.

    Returns ``(scores, ok)``. A missing or wrong-length answer gives ``ok=False``; missing
    steps are scored 0.
    """
    steps = _steps(rubric)
    parsed = None
    if answer:
        segments = re.split(r"ANSWER\s*:", answer, flags=re.IGNORECASE)
        spaces = [segments[-1]] + ([answer] if len(segments) > 1 else [])
        for space in spaces:
            for match in re.finditer(r"\[[^\[\]]*\]", space, re.DOTALL):
                try:
                    values = json.loads(match.group(0))
                except json.JSONDecodeError:
                    numbers = re.findall(r"-?\d+(?:\.\d+)?", match.group(0))
                    values = [float(x) for x in numbers] if numbers else None
                if isinstance(values, list) and values and all(isinstance(v, (int, float)) for v in values):
                    parsed = [float(v) for v in values]
                    break
            if parsed is not None:
                break
        if parsed is None:
            numbers = re.findall(r"-?\d+(?:\.\d+)?", spaces[0])
            parsed = [float(x) for x in numbers[: len(steps)]] if numbers else None
    ok = parsed is not None and len(parsed) == len(steps)
    values = list(parsed or []) + [0.0] * len(steps)
    return [min(max(0.0, values[i]), steps[i]["max_score"]) for i in range(len(steps))], ok


def medoid(score_vectors: Sequence[Sequence[float]]) -> int:
    """Index of the vector with the smallest summed L1 distance to all others."""
    if not score_vectors:
        raise ValueError("no parsed rubric answers")

    def l1(a, b):
        return sum(abs(x - y) for x, y in zip(a, b))

    return min(
        range(len(score_vectors)),
        key=lambda i: (sum(l1(score_vectors[i], other) for other in score_vectors), i),
    )
