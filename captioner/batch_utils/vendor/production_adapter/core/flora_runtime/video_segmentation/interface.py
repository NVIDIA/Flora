# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from collections.abc import Awaitable, Sequence
from enum import Enum
from typing import Protocol, runtime_checkable
from flora_runtime.schema import AnnotationSource, TemporalSegment, TimeSpan

class SegmentTopology(str, Enum):
    EXCLUSIVE = 'exclusive_intervals'
    CONCURRENT = 'concurrent_intervals'

def segment_topology(segmenter: object) -> SegmentTopology:
    value = getattr(segmenter, 'segment_topology', SegmentTopology.EXCLUSIVE)
    try:
        return SegmentTopology(value)
    except ValueError as exc:
        raise ValueError(f'segmenter declares unsupported segment_topology: {value!r}') from exc

@runtime_checkable
class VideoSegmenter(Protocol):

    def segment(self, source: AnnotationSource) -> Sequence[TemporalSegment | TimeSpan] | Awaitable[Sequence[TemporalSegment | TimeSpan]]:
        ...
