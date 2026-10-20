# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from fractions import Fraction
TARGET_SAMPLE_FPS = 8
TARGET_MIN_FRAMES = 8
TARGET_MAX_FRAMES = 40
TARGET_FRAME_FACTOR = 2
CONTEXT_SAMPLE_FPS = 8
CONTEXT_MIN_FRAMES = 4
CONTEXT_MAX_FRAMES = 12
CONTEXT_FRAME_FACTOR = 4
MAX_TOTAL_FRAMES = 64
SAMPLER_VERSION = 'duration_adaptive_uniform_bin_centers_nested_context_v2'

class U2ContractError(ValueError):
    pass

def uniform_bin_centers(start: int, end_exclusive: int, count: int) -> list[int]:
    length = end_exclusive - start
    if count <= 0:
        return []
    if length < count:
        raise U2ContractError(f'span [{start}, {end_exclusive}) has fewer than {count} frames')
    result = [start + (2 * index + 1) * length // (2 * count) for index in range(count)]
    if len(result) != len(set(result)) or result[0] < start or result[-1] >= end_exclusive:
        raise AssertionError('uniform frame sampler violated its contract')
    return result

def duration_adaptive_sample(start: int, end_exclusive: int, *, source_fps_num: int, source_fps_den: int, sample_fps: int, min_frames: int, max_frames: int, frame_factor: int) -> tuple[list[int], dict[str, int | bool | str]]:
    available = end_exclusive - start
    if source_fps_num <= 0 or source_fps_den <= 0 or available <= 0:
        raise U2ContractError('source FPS and frame span must be positive')
    if min_frames % frame_factor or max_frames % frame_factor:
        raise U2ContractError('adaptive frame bounds must align to frame factor')
    duration = Fraction(available * source_fps_den, source_fps_num)
    requested = min(Fraction(max_frames), max(Fraction(min_frames), duration * sample_fps))
    desired_count = requested.numerator // (requested.denominator * frame_factor) * frame_factor
    actual_count = min(desired_count, available // frame_factor * frame_factor)
    if actual_count < frame_factor:
        raise U2ContractError(f'span [{start}, {end_exclusive}) has fewer than {frame_factor} frames')
    indices = uniform_bin_centers(start, end_exclusive, actual_count)
    return (indices, {'policy': 'duration_fps_bounded_unique_frames_v1', 'sampler_version': SAMPLER_VERSION, 'source_fps_num': source_fps_num, 'source_fps_den': source_fps_den, 'sample_fps_num': sample_fps, 'sample_fps_den': 1, 'min_frames': min_frames, 'max_frames': max_frames, 'frame_factor': frame_factor, 'available_unique_source_frames': available, 'desired_frame_count': desired_count, 'actual_frame_count': actual_count, 'source_limited': actual_count < desired_count, 'selection': 'uniform_bin_centers'})

def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)

def seconds_to_frame_span(start_sec: float | str, end_sec: float | str, *, source_fps_num: int, source_fps_den: int, source_frame_count: int) -> tuple[int, int]:
    start = Fraction(str(start_sec))
    end = Fraction(str(end_sec))
    if start < 0 or end <= start:
        raise U2ContractError(f'invalid seconds span [{start_sec}, {end_sec})')
    fps = Fraction(source_fps_num, source_fps_den)
    start_frame = max(0, min(source_frame_count, _ceil_fraction(start * fps)))
    end_frame = max(0, min(source_frame_count, _ceil_fraction(end * fps)))
    if end_frame <= start_frame:
        raise U2ContractError(f'seconds span [{start_sec}, {end_sec}) maps to no source frames')
    return (start_frame, end_frame)
