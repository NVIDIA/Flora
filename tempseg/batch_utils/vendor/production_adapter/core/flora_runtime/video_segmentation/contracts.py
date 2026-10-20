# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from typing import Any, Literal
from flora_runtime.schema import TimeSpan
EventCompleteness = Literal['complete', 'left_censored', 'right_censored', 'both_censored', 'uncertain']

def _require_nonempty(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label} must be non-empty')

def _validate_confidence(value: float | None, label: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f'{label} must be a number in [0, 1]')
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f'{label} must be finite and in [0, 1]')

def span_contains(outer: TimeSpan, inner: TimeSpan, *, tolerance_s: float=1e-09) -> bool:
    if tolerance_s < 0 or not math.isfinite(tolerance_s):
        raise ValueError('tolerance_s must be finite and non-negative')
    return outer.start_s <= inner.start_s + tolerance_s and outer.end_s + tolerance_s >= inner.end_s

@dataclass(frozen=True)
class ProcessingScope:
    scope_id: str
    source_id: str
    core_span: TimeSpan
    context_span: TimeSpan
    proposal_confidence: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty(self.scope_id, 'scope_id')
        _require_nonempty(self.source_id, 'source_id')
        if not span_contains(self.context_span, self.core_span):
            raise ValueError('context_span must contain core_span')
        _validate_confidence(self.proposal_confidence, 'proposal_confidence')
        object.__setattr__(self, 'metadata', dict(self.metadata))

@dataclass(frozen=True)
class EventInterval:
    event_id: str
    source_id: str
    span: TimeSpan
    completeness: EventCompleteness = 'complete'
    start_confidence: float | None = None
    end_confidence: float | None = None
    claim: str | None = None
    source_scope_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty(self.event_id, 'event_id')
        _require_nonempty(self.source_id, 'source_id')
        if self.completeness not in {'complete', 'left_censored', 'right_censored', 'both_censored', 'uncertain'}:
            raise ValueError(f'unsupported completeness: {self.completeness!r}')
        _validate_confidence(self.start_confidence, 'start_confidence')
        _validate_confidence(self.end_confidence, 'end_confidence')
        if self.claim is not None:
            _require_nonempty(self.claim, 'claim')
            object.__setattr__(self, 'claim', self.claim.strip())
        scope_ids = tuple(self.source_scope_ids)
        if len(scope_ids) != len(set(scope_ids)):
            raise ValueError('source_scope_ids must be unique')
        for scope_id in scope_ids:
            _require_nonempty(scope_id, 'source_scope_id')
        object.__setattr__(self, 'source_scope_ids', scope_ids)
        object.__setattr__(self, 'metadata', dict(self.metadata))
