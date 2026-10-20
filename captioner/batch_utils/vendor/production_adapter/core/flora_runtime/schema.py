# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import os
from typing import Any

def _require_nonempty(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f'{name} must not be empty')

def _require_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite, got {value}')

@dataclass(frozen=True)
class VideoStream:
    camera: str
    location: str | os.PathLike[str]
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty(self.camera, 'camera')
        location = os.fspath(self.location)
        _require_nonempty(location, 'location')
        object.__setattr__(self, 'location', location)
        object.__setattr__(self, 'metadata', dict(self.metadata))
        if self.fps is not None:
            _require_finite(self.fps, 'fps')
            if self.fps <= 0:
                raise ValueError(f'fps must be positive, got {self.fps}')
        for value, name in ((self.width, 'width'), (self.height, 'height')):
            if value is not None and value <= 0:
                raise ValueError(f'{name} must be positive, got {value}')

@dataclass(frozen=True)
class AnnotationSource:
    source_id: str
    location: str | os.PathLike[str]
    video_streams: tuple[VideoStream, ...] = ()
    duration_s: float | None = None
    fps: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty(self.source_id, 'source_id')
        location = os.fspath(self.location)
        _require_nonempty(location, 'location')
        object.__setattr__(self, 'location', location)
        object.__setattr__(self, 'video_streams', tuple(self.video_streams))
        object.__setattr__(self, 'metadata', dict(self.metadata))
        cameras = [stream.camera for stream in self.video_streams]
        if len(cameras) != len(set(cameras)):
            raise ValueError(f'video stream camera names must be unique, got {cameras}')
        if self.duration_s is not None:
            _require_finite(self.duration_s, 'duration_s')
            if self.duration_s <= 0:
                raise ValueError(f'duration_s must be positive, got {self.duration_s}')
        if self.fps is not None:
            _require_finite(self.fps, 'fps')
            if self.fps <= 0:
                raise ValueError(f'fps must be positive, got {self.fps}')

@dataclass(frozen=True, order=True)
class TimeSpan:
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        _require_finite(self.start_s, 'start_s')
        _require_finite(self.end_s, 'end_s')
        if self.start_s < 0:
            raise ValueError(f'start_s must be non-negative, got {self.start_s}')
        if self.end_s <= self.start_s:
            raise ValueError(f'end_s must be greater than start_s, got [{self.start_s}, {self.end_s})')

@dataclass(frozen=True)
class TemporalSegment:
    span: TimeSpan
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'metadata', dict(self.metadata))
