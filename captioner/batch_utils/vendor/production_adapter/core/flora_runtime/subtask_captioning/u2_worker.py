# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from flora_runtime.subtask_captioning.u2_contract import CONTEXT_FRAME_FACTOR, CONTEXT_MAX_FRAMES, CONTEXT_MIN_FRAMES, CONTEXT_SAMPLE_FPS, MAX_TOTAL_FRAMES, TARGET_FRAME_FACTOR, TARGET_MAX_FRAMES, TARGET_MIN_FRAMES, TARGET_SAMPLE_FPS, duration_adaptive_sample, seconds_to_frame_span
_ROLES = ('BEFORE', 'TARGET', 'AFTER')

class U2WorkerError(RuntimeError):
    pass

@dataclass(frozen=True)
class _BlockPlan:
    role: str
    start_frame: int
    end_frame_exclusive: int
    frame_indices: tuple[int, ...]
    sampling_receipt: dict[str, Any]

def _plan_blocks(u2_context: dict[str, Any], media: dict[str, Any]) -> tuple[_BlockPlan, ...]:
    plans: list[_BlockPlan] = []
    for role, key in zip(_ROLES, ('before', 'target', 'after'), strict=True):
        span = u2_context.get(key)
        if span is None:
            if role == 'TARGET':
                raise U2WorkerError('U2 TARGET context is absent')
            continue
        if not isinstance(span, dict):
            raise U2WorkerError(f'U2 {key} span must be an object')
        start, end = seconds_to_frame_span(span.get('start_s'), span.get('end_s'), source_fps_num=media['fps_num'], source_fps_den=media['fps_den'], source_frame_count=media['frame_count'])
        target = role == 'TARGET'
        frame_factor = TARGET_FRAME_FACTOR if target else CONTEXT_FRAME_FACTOR
        if end - start < frame_factor:
            if not target:
                continue
            raise U2WorkerError(f'U2 {role} span has fewer than the required {frame_factor}-frame group')
        indices, receipt = duration_adaptive_sample(start, end, source_fps_num=media['fps_num'], source_fps_den=media['fps_den'], sample_fps=TARGET_SAMPLE_FPS if target else CONTEXT_SAMPLE_FPS, min_frames=TARGET_MIN_FRAMES if target else CONTEXT_MIN_FRAMES, max_frames=TARGET_MAX_FRAMES if target else CONTEXT_MAX_FRAMES, frame_factor=frame_factor)
        plans.append(_BlockPlan(role=role, start_frame=start, end_frame_exclusive=end, frame_indices=tuple(indices), sampling_receipt=dict(receipt)))
    total = sum((len(plan.frame_indices) for plan in plans))
    if total > MAX_TOTAL_FRAMES:
        raise U2WorkerError(f'U2 frame budget {total} exceeds {MAX_TOTAL_FRAMES}')
    return tuple(plans)
if __name__ == '__main__':
    raise SystemExit(main())
