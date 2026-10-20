# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
import json
import math
from flora_runtime.schema import TimeSpan
from flora_runtime.video_segmentation.contracts import EventInterval, ProcessingScope
TICK_US = 50000
CORE_US = 6000000
LEFT_GUARD_US = 1000000
RIGHT_GUARD_US = 1000000
SLOT_COUNT = 64
SLOT_FPS = 8
LEGACY_FINE_WINDOW_PROFILE_ID = 'atomic_1plus6plus1_64slots_8hz'
HUMAN_FINE_WINDOW_PROFILE_ID = 'human_16s_4hz'
YAM_ATOMIC_LONG_FINE_WINDOW_PROFILE_ID = 'yam_atomic_2plus12plus2_64slots_4hz'
YAM_INTENT_32S_FINE_WINDOW_PROFILE_ID = 'yam_intent_4plus24plus4_128slots_4hz'
DEFAULT_FINE_WINDOW_PROFILE_ID = LEGACY_FINE_WINDOW_PROFILE_ID
MAX_SEGMENTS = 32
STITCH_TOLERANCE_TICKS = 2
MAX_BOUNDED_ANCHOR_TICKS = 4
STRICT_TRACE_STITCHING_CONTRACT = 'strict_shared_trace_endpoint_v1'
SHARED_TRACE_IOU_STITCHING_CONTRACT = 'shared_trace_iou_endpoint_v1'
COMPONENT_LOCAL_IDENTITY_STITCHING_CONTRACT = 'component_local_identity_anchor_v1'
COMPONENT_LOCAL_BOUNDED_ANCHOR_STITCHING_CONTRACT = 'component_local_bounded_anchor_certificate_v1'
COMPONENT_LOCAL_STITCHING_CONTRACTS = frozenset({COMPONENT_LOCAL_IDENTITY_STITCHING_CONTRACT, COMPONENT_LOCAL_BOUNDED_ANCHOR_STITCHING_CONTRACT})
STITCHING_CONTRACTS = {STRICT_TRACE_STITCHING_CONTRACT, SHARED_TRACE_IOU_STITCHING_CONTRACT, *COMPONENT_LOCAL_STITCHING_CONTRACTS}
STITCHING_SEAM_DIAGNOSTICS_SCHEMA = 'data_engine_fine_stitching_seam_diagnostics_v1'
WINDOW_SCHEMA = 'data_engine_atomic_fine_window_v1'
MODEL_IO_CONTRACT = 'flora_inference_atomic_temporal_segmentation_v1'
HUMAN_MODEL_IO_CONTRACT = 'flora_human_temporal_segmentation_v1'
YAM_ATOMIC_LONG_MODEL_IO_CONTRACT = 'data_engine_yam_atomic_temporal_segmentation_16s4hz_v1'
YAM_INTENT_32S_MODEL_IO_CONTRACT = 'data_engine_yam_intent_temporal_segmentation_32s4hz_v1'
ComponentPairKey = tuple[int, int, int]

class FineTemporalError(ValueError):
    pass

class FinePredictionParseError(FineTemporalError):
    pass

@dataclass(frozen=True)
class FineWindowProfile:
    profile_id: str
    left_guard_us: int
    core_us: int
    right_guard_us: int
    slot_count: int
    slot_fps: int
    model_io_contract: str
    target_granularity: str

    def __post_init__(self) -> None:
        if not self.profile_id or not self.model_io_contract:
            raise FineTemporalError('fine window profile identity and model contract are required')
        if self.target_granularity not in {'atomic_event', 'coarse_subtask_run'}:
            raise FineTemporalError('fine window target granularity is unsupported')
        values = (self.left_guard_us, self.core_us, self.right_guard_us, self.slot_count, self.slot_fps)
        if any((isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values)):
            raise FineTemporalError('fine window geometry and sampling values must be positive integers')
        if self.view_us % TICK_US or self.core_us % TICK_US:
            raise FineTemporalError('fine window geometry must align to the 20 Hz model clock')
        if self.left_guard_us != self.right_guard_us:
            raise FineTemporalError('fine window profiles currently require symmetric guards')
        if 1000000 % self.slot_fps:
            raise FineTemporalError('fine window sample rate must divide one second exactly')
        if self.view_us * self.slot_fps != self.slot_count * 1000000:
            raise FineTemporalError('fine window slots, duration, and sample rate disagree')

    @property
    def view_us(self) -> int:
        return self.left_guard_us + self.core_us + self.right_guard_us

    @property
    def view_ticks(self) -> int:
        return self.view_us // TICK_US

    @property
    def core_ticks(self) -> int:
        return self.core_us // TICK_US

    @property
    def slot_spacing_us(self) -> int:
        return 1000000 // self.slot_fps

    @property
    def scheduler_contract(self) -> str:
        core = self._duration_token(self.core_us)
        guard = self._duration_token(self.left_guard_us)
        return f'fixed_{core}_core_{guard}_guards_global_grid_v1'

    def retry_scheduler_contract(self, *, directional: bool) -> str:
        prefix = 'adaptive_directional' if directional else 'adaptive_recentered'
        core = self._duration_token(self.core_us)
        guard = self._duration_token(self.left_guard_us)
        geometry = f'{core}_core_{guard}_guards'
        return f'{prefix}_{geometry}_v1'

    @staticmethod
    def _duration_token(value_us: int) -> str:
        return f'{value_us // 1000000}s' if value_us % 1000000 == 0 else f'{value_us}us'

    def to_dict(self) -> dict[str, int | str]:
        return {'profile_id': self.profile_id, 'left_guard_us': self.left_guard_us, 'core_us': self.core_us, 'right_guard_us': self.right_guard_us, 'view_us': self.view_us, 'slot_count': self.slot_count, 'slot_fps': self.slot_fps, 'model_io_contract': self.model_io_contract, 'target_granularity': self.target_granularity}
FINE_WINDOW_PROFILES: Mapping[str, FineWindowProfile] = {LEGACY_FINE_WINDOW_PROFILE_ID: FineWindowProfile(profile_id=LEGACY_FINE_WINDOW_PROFILE_ID, left_guard_us=LEFT_GUARD_US, core_us=CORE_US, right_guard_us=RIGHT_GUARD_US, slot_count=SLOT_COUNT, slot_fps=SLOT_FPS, model_io_contract=MODEL_IO_CONTRACT, target_granularity='atomic_event'), HUMAN_FINE_WINDOW_PROFILE_ID: FineWindowProfile(profile_id=HUMAN_FINE_WINDOW_PROFILE_ID, left_guard_us=2000000, core_us=12000000, right_guard_us=2000000, slot_count=64, slot_fps=4, model_io_contract=HUMAN_MODEL_IO_CONTRACT, target_granularity='coarse_subtask_run'), YAM_ATOMIC_LONG_FINE_WINDOW_PROFILE_ID: FineWindowProfile(profile_id=YAM_ATOMIC_LONG_FINE_WINDOW_PROFILE_ID, left_guard_us=2000000, core_us=12000000, right_guard_us=2000000, slot_count=64, slot_fps=4, model_io_contract=YAM_ATOMIC_LONG_MODEL_IO_CONTRACT, target_granularity='atomic_event'), YAM_INTENT_32S_FINE_WINDOW_PROFILE_ID: FineWindowProfile(profile_id=YAM_INTENT_32S_FINE_WINDOW_PROFILE_ID, left_guard_us=4000000, core_us=24000000, right_guard_us=4000000, slot_count=128, slot_fps=4, model_io_contract=YAM_INTENT_32S_MODEL_IO_CONTRACT, target_granularity='atomic_event')}

def fine_window_profile(profile_id: str=DEFAULT_FINE_WINDOW_PROFILE_ID) -> FineWindowProfile:
    try:
        return FINE_WINDOW_PROFILES[profile_id]
    except (KeyError, TypeError) as exc:
        raise FineTemporalError(f'unknown fine window profile: {profile_id!r}') from exc

@dataclass(frozen=True)
class StitchingPolicy:
    contract: str = STRICT_TRACE_STITCHING_CONTRACT
    endpoint_tolerance_ticks: int = STITCH_TOLERANCE_TICKS
    min_shared_trace_iou_numerator: int = 1
    min_shared_trace_iou_denominator: int = 2

    def __post_init__(self) -> None:
        if self.contract not in STITCHING_CONTRACTS:
            raise FineTemporalError(f'unknown stitching contract: {self.contract}')
        integer_fields = ('endpoint_tolerance_ticks', 'min_shared_trace_iou_numerator', 'min_shared_trace_iou_denominator')
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise FineTemporalError(f'{field_name} must be an integer')
        if self.endpoint_tolerance_ticks < 0:
            raise FineTemporalError('endpoint_tolerance_ticks must be non-negative')
        if self.contract == COMPONENT_LOCAL_BOUNDED_ANCHOR_STITCHING_CONTRACT and (not STITCH_TOLERANCE_TICKS < self.endpoint_tolerance_ticks <= MAX_BOUNDED_ANCHOR_TICKS):
            raise FineTemporalError('bounded-anchor tolerance must be above strict and at most four ticks')
        if not 0 < self.min_shared_trace_iou_numerator <= self.min_shared_trace_iou_denominator:
            raise FineTemporalError('shared-trace IoU threshold must lie in (0,1]')

    @classmethod
    def strict(cls, tolerance_ticks: int=STITCH_TOLERANCE_TICKS) -> StitchingPolicy:
        return cls(contract=STRICT_TRACE_STITCHING_CONTRACT, endpoint_tolerance_ticks=tolerance_ticks)

    @classmethod
    def shared_trace_iou(cls, *, endpoint_tolerance_ticks: int, numerator: int=1, denominator: int=2) -> StitchingPolicy:
        return cls(contract=SHARED_TRACE_IOU_STITCHING_CONTRACT, endpoint_tolerance_ticks=endpoint_tolerance_ticks, min_shared_trace_iou_numerator=numerator, min_shared_trace_iou_denominator=denominator)

    @classmethod
    def component_local_identity(cls, *, endpoint_tolerance_ticks: int=4, numerator: int=3, denominator: int=4) -> StitchingPolicy:
        if endpoint_tolerance_ticks <= STITCH_TOLERANCE_TICKS:
            raise FineTemporalError('component-local candidate tolerance must exceed the strict tolerance')
        return cls(contract=COMPONENT_LOCAL_IDENTITY_STITCHING_CONTRACT, endpoint_tolerance_ticks=endpoint_tolerance_ticks, min_shared_trace_iou_numerator=numerator, min_shared_trace_iou_denominator=denominator)

    @classmethod
    def component_local_bounded_anchor(cls, *, endpoint_tolerance_ticks: int=4, numerator: int=3, denominator: int=4) -> StitchingPolicy:
        if not STITCH_TOLERANCE_TICKS < endpoint_tolerance_ticks <= MAX_BOUNDED_ANCHOR_TICKS:
            raise FineTemporalError('bounded-anchor candidate tolerance must be above strict and at most four ticks')
        return cls(contract=COMPONENT_LOCAL_BOUNDED_ANCHOR_STITCHING_CONTRACT, endpoint_tolerance_ticks=endpoint_tolerance_ticks, min_shared_trace_iou_numerator=numerator, min_shared_trace_iou_denominator=denominator)

    def to_dict(self) -> dict[str, int | str]:
        return {'contract': self.contract, 'endpoint_tolerance_ticks': self.endpoint_tolerance_ticks, 'min_shared_trace_iou_numerator': self.min_shared_trace_iou_numerator, 'min_shared_trace_iou_denominator': self.min_shared_trace_iou_denominator}

def floor_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise FineTemporalError('divisor must be positive')
    return value // divisor

def ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise FineTemporalError('divisor must be positive')
    return -(-value // divisor)

def round_half_up_div(value: int, divisor: int) -> int:
    if divisor <= 0 or value < 0:
        raise FineTemporalError('round_half_up_div requires non-negative input and positive divisor')
    quotient, remainder = divmod(value, divisor)
    return quotient + int(remainder * 2 >= divisor)

@dataclass(frozen=True)
class FineWindow:
    source_id: str
    window_index: int
    video_end_us: int
    requested_view_start_us: int
    requested_view_end_us: int
    valid_start_us: int
    valid_end_us: int
    core_start_us: int
    core_end_us: int
    valid_start_tick: int
    valid_end_tick: int
    core_start_tick: int
    core_end_tick: int
    scope_id: str
    window_profile_id: str = DEFAULT_FINE_WINDOW_PROFILE_ID
    coarse_spans: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not self.source_id or not self.scope_id or self.window_index < 0 or (self.video_end_us <= 0):
            raise FineTemporalError('window identity, index, and duration are required')
        profile = fine_window_profile(self.window_profile_id)
        if self.requested_view_end_us - self.requested_view_start_us != profile.view_us:
            raise FineTemporalError(f'requested view must match profile {profile.profile_id!r}')
        if not self.requested_view_start_us <= self.valid_start_us < self.valid_end_us <= self.requested_view_end_us:
            raise FineTemporalError('invalid visible-video envelope')
        if not 0 <= self.valid_start_us < self.valid_end_us <= self.video_end_us:
            raise FineTemporalError('valid envelope must lie inside the video')
        if not self.valid_start_us <= self.core_start_us < self.core_end_us <= self.valid_end_us:
            raise FineTemporalError('emit core must lie inside the valid envelope')
        if not 0 <= self.valid_start_tick < self.valid_end_tick <= profile.view_ticks:
            raise FineTemporalError('invalid visible tick envelope')
        if not self.valid_start_tick <= self.core_start_tick < self.core_end_tick <= self.valid_end_tick:
            raise FineTemporalError('invalid core tick envelope')
        coarse_spans = tuple((dict(item) for item in self.coarse_spans))
        if len(coarse_spans) > 64:
            raise FineTemporalError('one view may contain at most 64 coarse spans')
        previous: tuple[int, int, bool, bool] | None = None
        for index, row in enumerate(coarse_spans):
            if set(row) != {'s', 'e', 'lo', 'ro'}:
                raise FineTemporalError(f'coarse span {index} must contain exactly s,e,lo,ro')
            fragment = FineFragment(row['s'], row['e'], row['lo'], row['ro'])
            if isinstance(fragment.s, bool) or isinstance(fragment.e, bool):
                raise FineTemporalError(f'coarse span {index} ticks must be integers')
            if not isinstance(fragment.s, int) or not isinstance(fragment.e, int):
                raise FineTemporalError(f'coarse span {index} ticks must be integers')
            if type(fragment.lo) is not bool or type(fragment.ro) is not bool:
                raise FineTemporalError(f'coarse span {index} open flags must be booleans')
            if not self.valid_start_tick <= fragment.s < fragment.e <= self.valid_end_tick:
                raise FineTemporalError(f'coarse span {index} lies outside the valid envelope')
            if fragment.lo and fragment.s != self.valid_start_tick:
                raise FineTemporalError(f'coarse span {index} lo=true is not clipped to valid start')
            if fragment.ro and fragment.e != self.valid_end_tick:
                raise FineTemporalError(f'coarse span {index} ro=true is not clipped to valid end')
            if previous is not None and fragment.sort_key <= previous:
                raise FineTemporalError('coarse spans must be unique and in canonical order')
            previous = fragment.sort_key
        object.__setattr__(self, 'coarse_spans', coarse_spans)

    @property
    def global_tick_offset(self) -> int:
        if self.requested_view_start_us % TICK_US:
            raise FineTemporalError('requested view is not aligned to the 20 Hz model clock')
        return self.requested_view_start_us // TICK_US

    @property
    def window_profile(self) -> FineWindowProfile:
        return fine_window_profile(self.window_profile_id)

    def worker_payload(self) -> dict[str, object]:
        return {'schema_version': WINDOW_SCHEMA, 'model_io_contract': self.window_profile.model_io_contract, 'window_profile_id': self.window_profile_id, 'scope_id': self.scope_id, 'window_index': self.window_index, 'requested_view_start_us': self.requested_view_start_us, 'requested_view_end_us': self.requested_view_end_us, 'valid_start_tick': self.valid_start_tick, 'valid_end_tick': self.valid_end_tick, 'core_start_tick': self.core_start_tick, 'core_end_tick': self.core_end_tick, 'coarse_spans': [dict(item) for item in self.coarse_spans]}

@dataclass(frozen=True)
class FineFragment:
    s: int
    e: int
    lo: bool
    ro: bool

    @property
    def sort_key(self) -> tuple[int, int, bool, bool]:
        return (self.s, self.e, self.lo, self.ro)

def _reject_constant(value: str) -> None:
    raise FinePredictionParseError(f'non-finite JSON constant is forbidden: {value}')

def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FinePredictionParseError(f'duplicate JSON key is forbidden: {key}')
        result[key] = value
    return result

def _validate_fragment_envelope(fragment: FineFragment, window: FineWindow) -> None:
    if isinstance(fragment.s, bool) or isinstance(fragment.e, bool):
        raise FinePredictionParseError('segment ticks must be integers, not booleans')
    if not isinstance(fragment.s, int) or not isinstance(fragment.e, int):
        raise FinePredictionParseError('segment ticks must be integers')
    if type(fragment.lo) is not bool or type(fragment.ro) is not bool:
        raise FinePredictionParseError('open flags must be JSON booleans')
    if not window.valid_start_tick <= fragment.s < fragment.e <= window.valid_end_tick <= window.window_profile.view_ticks:
        raise FinePredictionParseError('segment lies outside the valid tick envelope')
    if fragment.lo and fragment.s != window.valid_start_tick:
        raise FinePredictionParseError('lo=true requires clipping to valid_start_tick')
    if fragment.ro and fragment.e != window.valid_end_tick:
        raise FinePredictionParseError('ro=true requires clipping to valid_end_tick')

def _has_emit_core_support(fragment: FineFragment, window: FineWindow) -> bool:
    core_intersects = fragment.s < window.core_end_tick and fragment.e > window.core_start_tick
    core_touches = fragment.e == window.core_start_tick or fragment.s == window.core_end_tick
    return core_intersects or core_touches

def _validate_fragment(fragment: FineFragment, window: FineWindow) -> None:
    _validate_fragment_envelope(fragment, window)
    if not _has_emit_core_support(fragment, window):
        raise FinePredictionParseError('segment has no emit-core support')

@dataclass(frozen=True)
class ParsedFinePrediction:
    fragments: tuple[FineFragment, ...]
    guard_only_fragments: tuple[FineFragment, ...]

def parse_prediction_with_diagnostics(payload: str, window: FineWindow, *, drop_guard_only: bool) -> ParsedFinePrediction:
    if not isinstance(payload, str) or not payload:
        raise FinePredictionParseError('prediction_json must be a non-empty string')
    try:
        value = json.loads(payload, object_pairs_hook=_object_without_duplicate_keys, parse_constant=_reject_constant)
    except FinePredictionParseError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise FinePredictionParseError('prediction is not one complete JSON object') from exc
    if not isinstance(value, dict) or list(value) != ['segments']:
        raise FinePredictionParseError('prediction must have exactly one top-level key: segments')
    rows = value['segments']
    if not isinstance(rows, list):
        raise FinePredictionParseError('segments must be a list')
    if len(rows) > MAX_SEGMENTS:
        raise FinePredictionParseError(f'prediction contains more than {MAX_SEGMENTS} segments')
    result: list[FineFragment] = []
    guard_only: list[FineFragment] = []
    previous: tuple[int, int, bool, bool] | None = None
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or list(row) != ['s', 'e', 'lo', 'ro']:
            raise FinePredictionParseError(f'segment {index}: keys must appear exactly as s,e,lo,ro')
        fragment = FineFragment(row['s'], row['e'], row['lo'], row['ro'])
        try:
            _validate_fragment_envelope(fragment, window)
        except FinePredictionParseError as exc:
            raise FinePredictionParseError(f'segment {index}: {exc}') from exc
        if previous is not None and fragment.sort_key < previous:
            raise FinePredictionParseError('segments are not in canonical order')
        previous = fragment.sort_key
        if not _has_emit_core_support(fragment, window):
            if not drop_guard_only:
                raise FinePredictionParseError(f'segment {index}: segment has no emit-core support')
            guard_only.append(fragment)
            continue
        result.append(fragment)
    return ParsedFinePrediction(tuple(result), tuple(guard_only))

@dataclass(frozen=True)
class _Observation:
    observation_id: int
    window_index: int
    row_ordinal: int
    scope_id: str
    start_tick: int
    end_tick: int
    lo: bool
    ro: bool
    valid_start_tick: int
    valid_end_tick: int

    @property
    def has_closed_start(self) -> bool:
        return not self.lo

    @property
    def has_closed_end(self) -> bool:
        return not self.ro

@dataclass(frozen=True)
class UnresolvedFineEvent:
    observation_ids: tuple[int, ...]
    source_scope_ids: tuple[str, ...]
    observation_spans_ticks: tuple[tuple[int, int], ...]
    closed_start_ticks: tuple[int, ...]
    closed_end_ticks: tuple[int, ...]
    support_start_tick: int
    support_end_tick: int
    matching_ambiguous: bool

@dataclass(frozen=True)
class FineSegmentationResult:
    events: tuple[EventInterval, ...]
    fragments: tuple[EventInterval, ...]
    unresolved: tuple[UnresolvedFineEvent, ...]
    adjacent_matching_ambiguities: int
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'diagnostics', dict(self.diagnostics))

def _observations(window: FineWindow, fragments: Sequence[FineFragment], *, first_id: int) -> tuple[_Observation, ...]:
    result: list[_Observation] = []
    for ordinal, fragment in enumerate(fragments):
        _validate_fragment(fragment, window)
        offset = window.global_tick_offset
        result.append(_Observation(observation_id=first_id + ordinal, window_index=window.window_index, row_ordinal=ordinal, scope_id=window.scope_id, start_tick=offset + fragment.s, end_tick=offset + fragment.e, lo=fragment.lo, ro=fragment.ro, valid_start_tick=offset + window.valid_start_tick, valid_end_tick=offset + window.valid_end_tick))
    return tuple(result)

def _shared_trace(item: _Observation, start: int, end: int) -> tuple[int, int] | None:
    trace = (max(item.start_tick, start), min(item.end_tick, end))
    return trace if trace[0] < trace[1] else None

@dataclass(frozen=True)
class _CandidateEvaluation:
    cost: tuple[int, int] | None
    rejection: str | None

def _candidate_evaluation(left: _Observation, right: _Observation, overlap_start: int, overlap_end: int, policy: StitchingPolicy) -> _CandidateEvaluation:
    left_trace = _shared_trace(left, overlap_start, overlap_end)
    right_trace = _shared_trace(right, overlap_start, overlap_end)
    if left_trace is None or right_trace is None:
        return _CandidateEvaluation(None, 'no_shared_trace')
    deltas = (abs(left_trace[0] - right_trace[0]), abs(left_trace[1] - right_trace[1]))
    if policy.contract in {STRICT_TRACE_STITCHING_CONTRACT, *COMPONENT_LOCAL_STITCHING_CONTRACTS}:
        if max(deltas) > policy.endpoint_tolerance_ticks:
            return _CandidateEvaluation(None, 'shared_trace_endpoint_delta')
        trace_cost = sum(deltas)
    else:
        intersection = max(0, min(left_trace[1], right_trace[1]) - max(left_trace[0], right_trace[0]))
        union = max(left_trace[1], right_trace[1]) - min(left_trace[0], right_trace[0])
        if union <= 0 or intersection * policy.min_shared_trace_iou_denominator < union * policy.min_shared_trace_iou_numerator:
            return _CandidateEvaluation(None, 'shared_trace_iou')
        trace_cost = (union - intersection) * 1000000 // union
    closed_cost = 0
    if left.has_closed_start and right.has_closed_start:
        delta = abs(left.start_tick - right.start_tick)
        if delta > policy.endpoint_tolerance_ticks:
            return _CandidateEvaluation(None, 'closed_start_delta')
        closed_cost += delta
    elif left.has_closed_start and left.start_tick > right.valid_start_tick + policy.endpoint_tolerance_ticks or (right.has_closed_start and right.start_tick > left.valid_start_tick + policy.endpoint_tolerance_ticks):
        return _CandidateEvaluation(None, 'closed_start_visible_to_censored_peer')
    if left.has_closed_end and right.has_closed_end:
        delta = abs(left.end_tick - right.end_tick)
        if delta > policy.endpoint_tolerance_ticks:
            return _CandidateEvaluation(None, 'closed_end_delta')
        closed_cost += delta
    elif left.has_closed_end and left.end_tick < right.valid_end_tick - policy.endpoint_tolerance_ticks or (right.has_closed_end and right.end_tick < left.valid_end_tick - policy.endpoint_tolerance_ticks):
        return _CandidateEvaluation(None, 'closed_end_visible_to_censored_peer')
    return _CandidateEvaluation((trace_cost, closed_cost), None)

@dataclass(frozen=True)
class _MatchState:
    count: int = 0
    trace_cost: int = 0
    closed_cost: int = 0
    pairings: tuple[tuple[tuple[int, int], ...], ...] = ((),)

    @property
    def key(self) -> tuple[int, int, int]:
        return (-self.count, self.trace_cost, self.closed_cost)

def _choose(states: Sequence[_MatchState]) -> _MatchState:
    best_key = min((state.key for state in states))
    optimal = [state for state in states if state.key == best_key]
    pairings = tuple(sorted({pairing for state in optimal for pairing in state.pairings})[:2])
    return _MatchState(-best_key[0], best_key[1], best_key[2], pairings)

def _match_adjacent(left: Sequence[_Observation], right: Sequence[_Observation], left_window: FineWindow, right_window: FineWindow, policy: StitchingPolicy) -> tuple[tuple[tuple[int, int], ...], bool, tuple[tuple[tuple[int, int], ...], ...]]:
    if right_window.window_index != left_window.window_index + 1:
        raise FineTemporalError('stitching requires the complete ordered global window grid')
    overlap_start = max(left_window.global_tick_offset + left_window.valid_start_tick, right_window.global_tick_offset + right_window.valid_start_tick)
    overlap_end = min(left_window.global_tick_offset + left_window.valid_end_tick, right_window.global_tick_offset + right_window.valid_end_tick)
    if overlap_start >= overlap_end:
        raise FineTemporalError('adjacent windows have no shared valid-visible guard')
    n, m = (len(left), len(right))
    dp = [[_MatchState() for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0]
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = [dp[i - 1][j], dp[i][j - 1]]
            evaluation = _candidate_evaluation(left[i - 1], right[j - 1], overlap_start, overlap_end, policy)
            if evaluation.cost is not None:
                prior = dp[i - 1][j - 1]
                pair = (left[i - 1].observation_id, right[j - 1].observation_id)
                candidates.append(_MatchState(prior.count + 1, prior.trace_cost + evaluation.cost[0], prior.closed_cost + evaluation.cost[1], tuple(((*pairing, pair) for pairing in prior.pairings))))
            dp[i][j] = _choose(candidates)
    result = dp[n][m]
    return (result.pairings[0], len(result.pairings) > 1, result.pairings)

class _UnionFind:

    def __init__(self, values: Iterable[int]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root, right_root = (self.find(left), self.find(right))
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)

def _median(values: Sequence[int]) -> int:
    ordered = sorted(values)
    if not ordered:
        raise FineTemporalError('cannot form endpoint consensus from no values')
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    value = Fraction(ordered[middle - 1] + ordered[middle], 2)
    return math.floor(value + Fraction(1, 2))

@dataclass(frozen=True)
class _ComponentState:
    root: int
    observation_ids: tuple[int, ...]
    starts: tuple[int, ...]
    ends: tuple[int, ...]

    @property
    def resolved(self) -> bool:
        return bool(self.starts) and bool(self.ends) and (self.starts[-1] - self.starts[0] <= STITCH_TOLERANCE_TICKS) and (self.ends[-1] - self.ends[0] <= STITCH_TOLERANCE_TICKS) and (_median(self.ends) > _median(self.starts))

    @property
    def partial_endpoint(self) -> tuple[str, tuple[int, ...]] | None:
        if bool(self.starts) == bool(self.ends):
            return None
        name, values = ('start', self.starts) if self.starts else ('end', self.ends)
        if values[-1] - values[0] > STITCH_TOLERANCE_TICKS:
            return None
        return (name, values)

@dataclass(frozen=True)
class _LocalCandidate:
    seam_index: int
    left_observation_id: int
    right_observation_id: int
    resolved_root: int
    partial_root: int
    anchor_endpoint: str
    anchor_tick: int
    anchor_delta_ticks: int
    trace_intersection_ticks: int
    trace_union_ticks: int

    @property
    def pair(self) -> tuple[int, int]:
        return (self.left_observation_id, self.right_observation_id)

    def to_dict(self) -> dict[str, object]:
        result = {'left_observation_id': self.left_observation_id, 'right_observation_id': self.right_observation_id, 'resolved_component_root': self.resolved_root, 'partial_component_root': self.partial_root, 'anchor_endpoint': self.anchor_endpoint, 'anchor_tick': self.anchor_tick, 'trace_intersection_ticks': self.trace_intersection_ticks, 'trace_union_ticks': self.trace_union_ticks}
        if self.anchor_delta_ticks:
            result['anchor_delta_ticks'] = self.anchor_delta_ticks
        return result

@dataclass(frozen=True)
class _AssociationPlan:
    base_pairs_by_seam: tuple[tuple[tuple[int, int], ...], ...]
    pairs_by_seam: tuple[tuple[tuple[int, int], ...], ...]
    ambiguous_by_seam: tuple[bool, ...]
    competing_by_seam: tuple[tuple[tuple[tuple[int, int], ...], ...], ...]
    component_consistency_tolerance_ticks: int
    component_local_by_seam: tuple[Mapping[str, object], ...]
    component_local_summary: Mapping[str, object] | None
    component_local_geometry_by_seam: tuple[Mapping[tuple[int, int], tuple[int, int]], ...]

def _component_states(observations: Sequence[_Observation], union_find: _UnionFind) -> dict[int, _ComponentState]:
    grouped: dict[int, list[_Observation]] = {}
    for observation in observations:
        grouped.setdefault(union_find.find(observation.observation_id), []).append(observation)
    return {root: _ComponentState(root=root, observation_ids=tuple(sorted((observation.observation_id for observation in component))), starts=tuple(sorted((observation.start_tick for observation in component if not observation.lo))), ends=tuple(sorted((observation.end_tick for observation in component if not observation.ro)))) for root, component in grouped.items()}

def _resolved_component_geometry(states: Mapping[int, _ComponentState]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(((_median(state.starts), _median(state.ends)) for state in states.values() if state.resolved)))

def _trace_iou_counts(left: _Observation, right: _Observation, *, overlap_start: int, overlap_end: int) -> tuple[int, int]:
    left_trace = _shared_trace(left, overlap_start, overlap_end)
    right_trace = _shared_trace(right, overlap_start, overlap_end)
    if left_trace is None or right_trace is None:
        return (0, 0)
    intersection = max(0, min(left_trace[1], right_trace[1]) - max(left_trace[0], right_trace[0]))
    union = max(left_trace[1], right_trace[1]) - min(left_trace[0], right_trace[0])
    return (intersection, union)

def _identity_anchor(left: _ComponentState, right: _ComponentState, *, maximum_delta_ticks: int) -> tuple[tuple[int, int, str, int, int] | None, str | None]:
    if left.resolved == right.resolved:
        return (None, 'requires_one_resolved_one_partial')
    resolved, partial = (left, right) if left.resolved else (right, left)
    partial_endpoint = partial.partial_endpoint
    if partial_endpoint is None:
        return (None, 'partial_component_is_not_one_sided')
    endpoint_name, values = partial_endpoint
    anchor_values = resolved.starts if endpoint_name == 'start' else resolved.ends
    anchor_tick = _median(anchor_values)
    maximum_delta = max((abs(value - anchor_tick) for value in values))
    if maximum_delta > maximum_delta_ticks:
        return (None, 'partial_endpoint_is_not_exactly_anchored' if maximum_delta_ticks == 0 else 'partial_endpoint_exceeds_anchor_tolerance')
    if maximum_delta == 0 and _median((*anchor_values, *values)) != anchor_tick:
        return (None, 'partial_endpoint_would_move_resolved_geometry')
    return ((resolved.root, partial.root, endpoint_name, anchor_tick, maximum_delta), None)

def _monotone_with_pairs(pair: tuple[int, int], *, base_pairs: Sequence[tuple[int, int]], left: Sequence[_Observation], right: Sequence[_Observation]) -> bool:
    left_position = {item.observation_id: index for index, item in enumerate(left)}
    right_position = {item.observation_id: index for index, item in enumerate(right)}
    left_index, right_index = (left_position[pair[0]], right_position[pair[1]])
    return all((not (left_index < left_position[base_left] and right_index > right_position[base_right] or (left_index > left_position[base_left] and right_index < right_position[base_right])) for base_left, base_right in base_pairs))

def _component_local_absorptions(ordered: Sequence[FineWindow], by_window: Mapping[int, Sequence[_Observation]], observations: Sequence[_Observation], base_pairs_by_seam: Sequence[Sequence[tuple[int, int]]], base_union: _UnionFind, policy: StitchingPolicy) -> tuple[tuple[tuple[tuple[int, int], ...], ...], tuple[Mapping[str, object], ...], Mapping[str, object], tuple[Mapping[tuple[int, int], tuple[int, int]], ...]]:
    states = _component_states(observations, base_union)
    strict_policy = StitchingPolicy.strict()
    candidates: list[_LocalCandidate] = []
    rejection_counts: dict[str, int] = {}
    seam_rejections: list[dict[str, int]] = [{} for _ in range(max(0, len(ordered) - 1))]
    funnel_counts: dict[str, int] = {}
    seam_funnel_counts: list[dict[str, int]] = [{} for _ in range(max(0, len(ordered) - 1))]
    relaxed_rejection_counts: dict[str, int] = {}
    identity_trace_rejection_buckets: dict[str, int] = {}
    anchor_delta_ticks_histogram: dict[str, int] = {}

    def reject(seam_index: int, reason: str) -> None:
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        current = seam_rejections[seam_index]
        current[reason] = current.get(reason, 0) + 1

    def advance(seam_index: int, stage: str) -> None:
        funnel_counts[stage] = funnel_counts.get(stage, 0) + 1
        current = seam_funnel_counts[seam_index]
        current[stage] = current.get(stage, 0) + 1

    def increment(values: dict[str, int], name: str) -> None:
        values[name] = values.get(name, 0) + 1
    for seam_index in range(len(ordered) - 1):
        left_window, right_window = (ordered[seam_index], ordered[seam_index + 1])
        left, right = (by_window[seam_index], by_window[seam_index + 1])
        overlap_start = max(left_window.global_tick_offset + left_window.valid_start_tick, right_window.global_tick_offset + right_window.valid_start_tick)
        overlap_end = min(left_window.global_tick_offset + left_window.valid_end_tick, right_window.global_tick_offset + right_window.valid_end_tick)
        base_pairs = tuple(base_pairs_by_seam[seam_index])
        matched_left = {pair[0] for pair in base_pairs}
        matched_right = {pair[1] for pair in base_pairs}
        for left_item in left:
            if left_item.observation_id in matched_left:
                continue
            for right_item in right:
                if right_item.observation_id in matched_right:
                    continue
                advance(seam_index, 'unmatched_observation_pairs')
                left_root = base_union.find(left_item.observation_id)
                right_root = base_union.find(right_item.observation_id)
                if left_root == right_root:
                    advance(seam_index, 'same_component_pairs')
                    continue
                advance(seam_index, 'cross_component_pairs')
                relaxed = _candidate_evaluation(left_item, right_item, overlap_start, overlap_end, policy)
                if relaxed.cost is None:
                    advance(seam_index, 'relaxed_infeasible_pairs')
                    increment(relaxed_rejection_counts, str(relaxed.rejection or 'unknown'))
                    continue
                advance(seam_index, 'relaxed_feasible_pairs')
                strict = _candidate_evaluation(left_item, right_item, overlap_start, overlap_end, strict_policy)
                if strict.cost is not None:
                    advance(seam_index, 'strict_feasible_pairs')
                    reject(seam_index, 'strict_feasible_not_selected')
                    continue
                advance(seam_index, 'strict_infeasible_pairs')
                intersection, union = _trace_iou_counts(left_item, right_item, overlap_start=overlap_start, overlap_end=overlap_end)
                if union <= 0 or intersection * policy.min_shared_trace_iou_denominator < union * policy.min_shared_trace_iou_numerator:
                    if union <= 0 or intersection <= 0:
                        trace_bucket = 'zero'
                    elif intersection * 4 < union:
                        trace_bucket = 'gt_0_lt_0.25'
                    elif intersection * 2 < union:
                        trace_bucket = '0.25_to_lt_0.5'
                    else:
                        trace_bucket = '0.5_to_lt_0.75'
                    increment(identity_trace_rejection_buckets, trace_bucket)
                    reject(seam_index, 'identity_trace_iou')
                    continue
                advance(seam_index, 'identity_trace_pass_pairs')
                maximum_anchor_delta_ticks = policy.endpoint_tolerance_ticks if policy.contract == COMPONENT_LOCAL_BOUNDED_ANCHOR_STITCHING_CONTRACT else 0
                anchor, reason = _identity_anchor(states[left_root], states[right_root], maximum_delta_ticks=maximum_anchor_delta_ticks)
                if anchor is None:
                    if reason in {'partial_endpoint_is_not_exactly_anchored', 'partial_endpoint_exceeds_anchor_tolerance'}:
                        left_state, right_state = (states[left_root], states[right_root])
                        resolved, partial = (left_state, right_state) if left_state.resolved else (right_state, left_state)
                        partial_endpoint = partial.partial_endpoint
                        if partial_endpoint is None:
                            raise FineTemporalError('exact-anchor rejection lacks a partial endpoint')
                        endpoint_name, values = partial_endpoint
                        anchor_values = resolved.starts if endpoint_name == 'start' else resolved.ends
                        anchor_tick = _median(anchor_values)
                        maximum_delta = max((abs(value - anchor_tick) for value in values))
                        increment(anchor_delta_ticks_histogram, str(maximum_delta))
                    reject(seam_index, str(reason))
                    continue
                advance(seam_index, 'identity_anchor_pass_pairs')
                pair = (left_item.observation_id, right_item.observation_id)
                if not _monotone_with_pairs(pair, base_pairs=base_pairs, left=left, right=right):
                    reject(seam_index, 'crosses_strict_pairing')
                    continue
                advance(seam_index, 'monotone_pair_candidates')
                resolved_root, partial_root, endpoint_name, anchor_tick, anchor_delta_ticks = anchor
                candidates.append(_LocalCandidate(seam_index=seam_index, left_observation_id=pair[0], right_observation_id=pair[1], resolved_root=resolved_root, partial_root=partial_root, anchor_endpoint=endpoint_name, anchor_tick=anchor_tick, anchor_delta_ticks=anchor_delta_ticks, trace_intersection_ticks=intersection, trace_union_ticks=union))
    by_component_pair: dict[tuple[int, int], list[_LocalCandidate]] = {}
    for candidate in candidates:
        key = (candidate.resolved_root, candidate.partial_root)
        by_component_pair.setdefault(key, []).append(candidate)
    unique_edges: list[_LocalCandidate] = []
    for values in by_component_pair.values():
        if len(values) != 1:
            for candidate in values:
                reject(candidate.seam_index, 'multiple_observation_edges')
            continue
        unique_edges.append(values[0])
        advance(values[0].seam_index, 'unique_component_edge_pairs')
    degrees: dict[int, int] = {}
    for candidate in unique_edges:
        degrees[candidate.resolved_root] = degrees.get(candidate.resolved_root, 0) + 1
        degrees[candidate.partial_root] = degrees.get(candidate.partial_root, 0) + 1
    reciprocal_unique: list[_LocalCandidate] = []
    for candidate in unique_edges:
        if degrees[candidate.resolved_root] != 1 or degrees[candidate.partial_root] != 1:
            reject(candidate.seam_index, 'non_unique_component_identity')
            continue
        reciprocal_unique.append(candidate)
        advance(candidate.seam_index, 'reciprocal_unique_pairs')
    crossing: set[_LocalCandidate] = set()
    by_seam_candidates: dict[int, list[_LocalCandidate]] = {}
    for candidate in reciprocal_unique:
        by_seam_candidates.setdefault(candidate.seam_index, []).append(candidate)
    for seam_index, values in by_seam_candidates.items():
        left, right = (by_window[seam_index], by_window[seam_index + 1])
        for candidate in values:
            peers = [peer.pair for peer in values if peer is not candidate]
            if not _monotone_with_pairs(candidate.pair, base_pairs=peers, left=left, right=right):
                crossing.add(candidate)
    accepted = []
    for candidate in reciprocal_unique:
        if candidate in crossing:
            reject(candidate.seam_index, 'crossing_local_candidates')
            continue
        accepted.append(candidate)
        advance(candidate.seam_index, 'geometry_accepted_pairs')
    accepted_by_seam: list[tuple[tuple[int, int], ...]] = []
    geometry_by_seam: list[Mapping[tuple[int, int], tuple[int, int]]] = []
    diagnostics_by_seam: list[Mapping[str, object]] = []
    for seam_index in range(len(ordered) - 1):
        seam_candidates = tuple((candidate for candidate in candidates if candidate.seam_index == seam_index))
        seam_accepted = tuple((candidate for candidate in accepted if candidate.seam_index == seam_index))
        accepted_by_seam.append(tuple(sorted((candidate.pair for candidate in seam_accepted))))
        geometry_by_seam.append({candidate.pair: (_median(states[candidate.resolved_root].starts), _median(states[candidate.resolved_root].ends)) for candidate in seam_accepted})
        diagnostics_by_seam.append({'candidate_pairs': [candidate.to_dict() for candidate in seam_candidates], 'accepted_pairs': [candidate.to_dict() for candidate in seam_accepted], 'rejection_counts': dict(sorted(seam_rejections[seam_index].items())), 'candidate_funnel': dict(sorted(seam_funnel_counts[seam_index].items()))})
    summary = {'contract': policy.contract, 'strict_base_tolerance_ticks': STITCH_TOLERANCE_TICKS, 'candidate_tolerance_ticks': policy.endpoint_tolerance_ticks, 'minimum_identity_trace_iou': {'numerator': policy.min_shared_trace_iou_numerator, 'denominator': policy.min_shared_trace_iou_denominator}, 'candidate_pair_count': len(candidates), 'accepted_pair_count': len(accepted), 'absorbed_partial_component_count': len(accepted), 'resolved_event_count_change_by_contract': 0, 'resolved_event_geometry_change_by_contract': False, 'new_model_calls': 0, 'rejection_counts': dict(sorted(rejection_counts.items())), 'candidate_funnel': dict(sorted(funnel_counts.items())), 'candidate_funnel_details': {'relaxed_infeasible_rejection_counts': dict(sorted(relaxed_rejection_counts.items())), 'identity_trace_iou_rejection_buckets': dict(sorted(identity_trace_rejection_buckets.items())), 'anchor_maximum_delta_ticks_histogram': dict(sorted(anchor_delta_ticks_histogram.items(), key=lambda item: int(item[0])))}}
    if policy.contract == COMPONENT_LOCAL_BOUNDED_ANCHOR_STITCHING_CONTRACT:
        summary['anchor_contract'] = {'mode': 'bounded_certificate_only', 'maximum_delta_ticks': policy.endpoint_tolerance_ticks, 'resolved_geometry_authority': 'strict_resolved_component', 'explicit_allowlist_required_for_runtime': True}
    return (tuple(accepted_by_seam), tuple(diagnostics_by_seam), summary, tuple(geometry_by_seam))

def _association_plan(ordered: Sequence[FineWindow], by_window: Mapping[int, Sequence[_Observation]], observations: Sequence[_Observation], policy: StitchingPolicy, component_pair_allowlist: Collection[ComponentPairKey] | None=None) -> _AssociationPlan:
    match_policy = StitchingPolicy.strict() if policy.contract in COMPONENT_LOCAL_STITCHING_CONTRACTS else policy
    base_pairs: list[tuple[tuple[int, int], ...]] = []
    ambiguous: list[bool] = []
    competing: list[tuple[tuple[tuple[int, int], ...], ...]] = []
    for index in range(len(ordered) - 1):
        pairs, is_ambiguous, alternatives = _match_adjacent(by_window[index], by_window[index + 1], ordered[index], ordered[index + 1], match_policy)
        base_pairs.append(pairs)
        ambiguous.append(is_ambiguous)
        competing.append(alternatives)
    if policy.contract not in COMPONENT_LOCAL_STITCHING_CONTRACTS:
        return _AssociationPlan(base_pairs_by_seam=tuple(base_pairs), pairs_by_seam=tuple(base_pairs), ambiguous_by_seam=tuple(ambiguous), competing_by_seam=tuple(competing), component_consistency_tolerance_ticks=policy.endpoint_tolerance_ticks, component_local_by_seam=tuple(({} for _ in base_pairs)), component_local_summary=None, component_local_geometry_by_seam=tuple(({} for _ in base_pairs)))
    base_union = _UnionFind((item.observation_id for item in observations))
    for pairs in base_pairs:
        for left_id, right_id in pairs:
            base_union.union(left_id, right_id)
    accepted, diagnostics, summary, geometry_by_seam = _component_local_absorptions(ordered, by_window, observations, base_pairs, base_union, policy)
    if component_pair_allowlist is not None:
        available = {(seam_index, *pair) for seam_index, seam_pairs in enumerate(accepted) for pair in seam_pairs}
        requested = set(component_pair_allowlist)
        unknown = requested - available
        if unknown:
            raise FineTemporalError(f'component pair allowlist contains non-candidates: {sorted(unknown)}')
        filtered = tuple((tuple((pair for pair in seam_pairs if (seam_index, *pair) in requested)) for seam_index, seam_pairs in enumerate(accepted)))
        rejected_count = sum(map(len, accepted)) - sum(map(len, filtered))
        filtered_diagnostics = []
        for seam_index, row in enumerate(diagnostics):
            current = dict(row)
            current['accepted_pairs'] = [pair for pair in row['accepted_pairs'] if (seam_index, int(pair['left_observation_id']), int(pair['right_observation_id'])) in requested]
            rejection_counts = dict(row['rejection_counts'])
            seam_rejected = len(accepted[seam_index]) - len(filtered[seam_index])
            if seam_rejected:
                rejection_counts['caption_certificate'] = seam_rejected
            current['rejection_counts'] = dict(sorted(rejection_counts.items()))
            filtered_diagnostics.append(current)
        accepted = filtered
        geometry_by_seam = tuple(({pair: geometry for pair, geometry in seam_geometry.items() if (seam_index, *pair) in requested} for seam_index, seam_geometry in enumerate(geometry_by_seam)))
        diagnostics = tuple(filtered_diagnostics)
        summary = dict(summary)
        summary['accepted_pair_count'] = sum(map(len, accepted))
        summary['absorbed_partial_component_count'] = sum(map(len, accepted))
        rejection_counts = dict(summary['rejection_counts'])
        if rejected_count:
            rejection_counts['caption_certificate'] = rejected_count
        summary['rejection_counts'] = dict(sorted(rejection_counts.items()))
    combined = tuple((tuple(sorted((*strict_pairs, *local_pairs))) for strict_pairs, local_pairs in zip(base_pairs, accepted, strict=True)))
    return _AssociationPlan(base_pairs_by_seam=tuple(base_pairs), pairs_by_seam=combined, ambiguous_by_seam=tuple(ambiguous), competing_by_seam=tuple(competing), component_consistency_tolerance_ticks=STITCH_TOLERANCE_TICKS, component_local_by_seam=diagnostics, component_local_summary=summary, component_local_geometry_by_seam=geometry_by_seam)

def _fragment_event(source_id: str, window: FineWindow, fragment: FineFragment, observation_id: int, row_ordinal: int) -> EventInterval:
    start_us = max(0, (window.global_tick_offset + fragment.s) * TICK_US)
    end_us = min(window.video_end_us, (window.global_tick_offset + fragment.e) * TICK_US)
    if end_us <= start_us:
        raise FineTemporalError('quantized local fragment is degenerate in source time')
    completeness = {(False, False): 'complete', (True, False): 'left_censored', (False, True): 'right_censored', (True, True): 'both_censored'}[fragment.lo, fragment.ro]
    return EventInterval(event_id=f'{source_id}:fragment:{observation_id:06d}', source_id=source_id, span=TimeSpan(start_us / 1000000, end_us / 1000000), completeness=completeness, source_scope_ids=(window.scope_id,), metadata={'window_index': window.window_index, 'row_ordinal': row_ordinal, 'observation_id': observation_id})

def _observation_payload(item: _Observation) -> dict[str, int | bool | str]:
    return {'observation_id': item.observation_id, 'window_index': item.window_index, 'row_ordinal': item.row_ordinal, 'scope_id': item.scope_id, 'start_tick': item.start_tick, 'end_tick': item.end_tick, 'lo': item.lo, 'ro': item.ro, 'valid_start_tick': item.valid_start_tick, 'valid_end_tick': item.valid_end_tick}

def analyze_stitching_seams(windows: Sequence[FineWindow], fragments_by_window: Mapping[int, Sequence[FineFragment]], *, policy: StitchingPolicy | None=None, tolerance_ticks: int=STITCH_TOLERANCE_TICKS, component_pair_allowlist: Collection[ComponentPairKey] | None=None) -> dict[str, object]:
    selected = policy or StitchingPolicy.strict(tolerance_ticks)
    if policy is not None and tolerance_ticks != STITCH_TOLERANCE_TICKS:
        raise FineTemporalError('pass either policy or a non-default tolerance_ticks, not both')
    if component_pair_allowlist is not None and (policy is None or policy.contract not in COMPONENT_LOCAL_STITCHING_CONTRACTS):
        raise FineTemporalError('component pair allowlist requires a component-local policy')
    ordered = tuple(sorted(windows, key=lambda item: item.window_index))
    if not ordered or [item.window_index for item in ordered] != list(range(len(ordered))):
        raise FineTemporalError('windows must be one complete zero-based sequence')
    if len({item.source_id for item in ordered}) != 1 or len({item.video_end_us for item in ordered}) != 1 or len({item.window_profile_id for item in ordered}) != 1:
        raise FineTemporalError('all windows must belong to one source, duration, and profile')
    unknown = set(fragments_by_window) - {item.window_index for item in ordered}
    if unknown:
        raise FineTemporalError(f'fragments supplied for unknown windows: {sorted(unknown)}')
    by_window: dict[int, tuple[_Observation, ...]] = {}
    observations: list[_Observation] = []
    next_id = 0
    for window in ordered:
        current = _observations(window, tuple(fragments_by_window.get(window.window_index, ())), first_id=next_id)
        next_id += len(current)
        observations.extend(current)
        by_window[window.window_index] = current
    plan = _association_plan(ordered, by_window, observations, selected, component_pair_allowlist)
    seams: list[dict[str, object]] = []
    rejection_totals: dict[str, int] = {}
    total_feasible = total_matched = total_ambiguous = 0
    for index in range(len(ordered) - 1):
        left_window, right_window = (ordered[index], ordered[index + 1])
        left, right = (by_window[index], by_window[index + 1])
        overlap_start = max(left_window.global_tick_offset + left_window.valid_start_tick, right_window.global_tick_offset + right_window.valid_start_tick)
        overlap_end = min(left_window.global_tick_offset + left_window.valid_end_tick, right_window.global_tick_offset + right_window.valid_end_tick)
        rejection_counts: dict[str, int] = {}
        feasible_pairs: list[dict[str, object]] = []
        for left_item in left:
            for right_item in right:
                evaluation = _candidate_evaluation(left_item, right_item, overlap_start, overlap_end, selected)
                if evaluation.cost is None:
                    reason = str(evaluation.rejection)
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                    rejection_totals[reason] = rejection_totals.get(reason, 0) + 1
                    continue
                feasible_pairs.append({'left_observation_id': left_item.observation_id, 'right_observation_id': right_item.observation_id, 'trace_cost': evaluation.cost[0], 'closed_endpoint_cost': evaluation.cost[1]})
        pairs = plan.pairs_by_seam[index]
        ambiguous = plan.ambiguous_by_seam[index]
        competing = plan.competing_by_seam[index]
        matched_left = {item[0] for item in pairs}
        matched_right = {item[1] for item in pairs}
        total_feasible += len(feasible_pairs)
        total_matched += len(pairs)
        total_ambiguous += int(ambiguous)
        seams.append({'left_window_index': left_window.window_index, 'right_window_index': right_window.window_index, 'overlap_start_tick': overlap_start, 'overlap_end_tick': overlap_end, 'left_observations': [_observation_payload(item) for item in left], 'right_observations': [_observation_payload(item) for item in right], 'feasible_pairs': feasible_pairs, 'matched_pairs': [list(item) for item in pairs], 'competing_optimal_pairings': [[list(item) for item in pairing] for pairing in competing], 'matching_ambiguous': ambiguous, 'unmatched_left_observation_ids': [item.observation_id for item in left if item.observation_id not in matched_left], 'unmatched_right_observation_ids': [item.observation_id for item in right if item.observation_id not in matched_right], 'rejection_counts': dict(sorted(rejection_counts.items())), 'component_local_identity': dict(plan.component_local_by_seam[index])})
    return {'schema_version': STITCHING_SEAM_DIAGNOSTICS_SCHEMA, 'policy': selected.to_dict(), 'window_count': len(ordered), 'seam_count': len(seams), 'observation_count': next_id, 'feasible_pair_count': total_feasible, 'matched_pair_count': total_matched, 'ambiguous_seam_count': total_ambiguous, 'rejection_counts': dict(sorted(rejection_totals.items())), 'component_local_identity': dict(plan.component_local_summary) if plan.component_local_summary is not None else None, 'seams': seams}

def stitch_predictions(windows: Sequence[FineWindow], fragments_by_window: Mapping[int, Sequence[FineFragment]], *, tolerance_ticks: int=STITCH_TOLERANCE_TICKS, policy: StitchingPolicy | None=None, component_pair_allowlist: Collection[ComponentPairKey] | None=None) -> FineSegmentationResult:
    if policy is not None and tolerance_ticks != STITCH_TOLERANCE_TICKS:
        raise FineTemporalError('pass either policy or a non-default tolerance_ticks, not both')
    if component_pair_allowlist is not None and (policy is None or policy.contract not in COMPONENT_LOCAL_STITCHING_CONTRACTS):
        raise FineTemporalError('component pair allowlist requires a component-local policy')
    selected = policy or StitchingPolicy.strict(tolerance_ticks)
    if selected.contract == COMPONENT_LOCAL_BOUNDED_ANCHOR_STITCHING_CONTRACT and component_pair_allowlist is None:
        raise FineTemporalError('bounded-anchor runtime requires an explicit component pair allowlist')
    ordered = tuple(sorted(windows, key=lambda item: item.window_index))
    if not ordered or [item.window_index for item in ordered] != list(range(len(ordered))):
        raise FineTemporalError('windows must be one complete zero-based sequence')
    if len({item.source_id for item in ordered}) != 1 or len({item.video_end_us for item in ordered}) != 1 or len({item.window_profile_id for item in ordered}) != 1:
        raise FineTemporalError('all windows must belong to one source, duration, and profile')
    unknown = set(fragments_by_window) - {item.window_index for item in ordered}
    if unknown:
        raise FineTemporalError(f'fragments supplied for unknown windows: {sorted(unknown)}')
    observations: list[_Observation] = []
    by_window: dict[int, tuple[_Observation, ...]] = {}
    fragments: list[EventInterval] = []
    next_id = 0
    for window in ordered:
        rows = tuple(fragments_by_window.get(window.window_index, ()))
        current = _observations(window, rows, first_id=next_id)
        for offset, fragment in enumerate(rows):
            fragments.append(_fragment_event(window.source_id, window, fragment, next_id + offset, offset))
        next_id += len(current)
        observations.extend(current)
        by_window[window.window_index] = current
    plan = _association_plan(ordered, by_window, observations, selected, component_pair_allowlist)
    baseline_geometry: tuple[tuple[int, int], ...] | None = None
    baseline_unresolved_count: int | None = None
    if selected.contract in COMPONENT_LOCAL_STITCHING_CONTRACTS:
        baseline_union = _UnionFind((item.observation_id for item in observations))
        for pairs in plan.base_pairs_by_seam:
            for left_id, right_id in pairs:
                baseline_union.union(left_id, right_id)
        baseline_states = _component_states(observations, baseline_union)
        baseline_geometry = _resolved_component_geometry(baseline_states)
        baseline_unresolved_count = sum((not state.resolved for state in baseline_states.values()))
    union_find = _UnionFind((item.observation_id for item in observations))
    ambiguity_ids: set[int] = set()
    ambiguity_count = 0
    for pairs, ambiguous, competing in zip(plan.pairs_by_seam, plan.ambiguous_by_seam, plan.competing_by_seam, strict=True):
        if ambiguous:
            ambiguity_count += 1
            ambiguity_ids.update((value for pairing in competing for pair in pairing for value in pair))
        for left_id, right_id in pairs:
            union_find.union(left_id, right_id)
    authoritative_geometry_by_root: dict[int, tuple[int, int]] = {}
    for geometry_by_pair in plan.component_local_geometry_by_seam:
        for (left_id, right_id), geometry in geometry_by_pair.items():
            root = union_find.find(left_id)
            if root != union_find.find(right_id):
                raise FineTemporalError('component-local geometry pair was not associated')
            prior = authoritative_geometry_by_root.get(root)
            if prior is not None and prior != geometry:
                raise FineTemporalError('component-local candidates disagree on authoritative geometry')
            authoritative_geometry_by_root[root] = geometry
    if baseline_geometry is not None and baseline_unresolved_count is not None:
        final_states = _component_states(observations, union_find)
        accepted_count = sum((len(pairs) - len(base_pairs) for pairs, base_pairs in zip(plan.pairs_by_seam, plan.base_pairs_by_seam, strict=True)))
        effective_geometry = tuple(sorted((authoritative_geometry_by_root.get(root, (_median(state.starts), _median(state.ends))) for root, state in final_states.items() if root in authoritative_geometry_by_root or state.resolved)))
        if effective_geometry != baseline_geometry:
            raise FineTemporalError('component-local identity relaxation changed resolved event geometry')
        if sum((root not in authoritative_geometry_by_root and (not state.resolved) for root, state in final_states.items())) != baseline_unresolved_count - accepted_count:
            raise FineTemporalError('component-local identity relaxation changed unsupported component structure')
    grouped: dict[int, list[_Observation]] = {}
    for observation in observations:
        grouped.setdefault(union_find.find(observation.observation_id), []).append(observation)
    ambiguous_roots = {union_find.find(value) for value in ambiguity_ids}
    resolved: list[tuple[TimeSpan, tuple[str, ...], dict[str, object]]] = []
    unresolved: list[UnresolvedFineEvent] = []
    video_end_us = ordered[0].video_end_us
    windows_by_index = {item.window_index: item for item in ordered}
    for root, component in sorted(grouped.items()):
        starts = tuple(sorted((item.start_tick for item in component if item.has_closed_start)))
        ends = tuple(sorted((item.end_tick for item in component if item.has_closed_end)))
        observation_spans = tuple(((item.start_tick, item.end_tick) for item in sorted(component, key=lambda item: item.observation_id)))
        scope_ids = tuple(dict.fromkeys((item.scope_id for item in sorted(component, key=lambda item: item.window_index))))
        raw_consistent = bool(starts) and bool(ends) and (starts[-1] - starts[0] <= plan.component_consistency_tolerance_ticks) and (ends[-1] - ends[0] <= plan.component_consistency_tolerance_ticks)
        authoritative_geometry = authoritative_geometry_by_root.get(root)
        consistent = authoritative_geometry is not None or raw_consistent
        if authoritative_geometry is not None:
            start_tick, end_tick = authoritative_geometry
            start_us = max(0, start_tick * TICK_US)
            end_us = min(video_end_us, end_tick * TICK_US)
        else:
            start_us = max(0, _median(starts) * TICK_US) if starts else None
            end_us = min(video_end_us, _median(ends) * TICK_US) if ends else None
        if not consistent or start_us is None or end_us is None or (end_us <= start_us):
            support_window_indices = {item.window_index for item in component}
            if not starts:
                support_window_indices.update((item.window_index - 1 for item in component if item.window_index > 0))
            if not ends:
                support_window_indices.update((item.window_index + 1 for item in component if item.window_index + 1 < len(ordered)))
            if starts and ends and (not consistent):
                support_window_indices.update((item.window_index - 1 for item in component if item.window_index > 0))
                support_window_indices.update((item.window_index + 1 for item in component if item.window_index + 1 < len(ordered)))
            support_windows = tuple((windows_by_index[index] for index in sorted(support_window_indices)))
            support_start_tick = min((item.global_tick_offset + item.valid_start_tick for item in support_windows))
            support_end_tick = max((item.global_tick_offset + item.valid_end_tick for item in support_windows))
            unresolved.append(UnresolvedFineEvent(tuple(sorted((item.observation_id for item in component))), scope_ids, observation_spans, starts, ends, support_start_tick, support_end_tick, root in ambiguous_roots))
            continue
        resolved.append((TimeSpan(start_us / 1000000, end_us / 1000000), scope_ids, {'observation_ids': tuple(sorted((item.observation_id for item in component))), 'matching_ambiguous': root in ambiguous_roots, **({'component_local_geometry_authority': 'strict_resolved_component', 'raw_closed_endpoints_consistent': raw_consistent} if authoritative_geometry is not None and selected.contract == COMPONENT_LOCAL_BOUNDED_ANCHOR_STITCHING_CONTRACT else {})}))
    resolved.sort(key=lambda item: (item[0].start_s, item[0].end_s, item[1]))
    identified = tuple((EventInterval(event_id=f'{ordered[0].source_id}:fine:{index:06d}', source_id=ordered[0].source_id, span=span, source_scope_ids=scope_ids, metadata=metadata) for index, (span, scope_ids, metadata) in enumerate(resolved)))
    return FineSegmentationResult(events=identified, fragments=tuple(fragments), unresolved=tuple(unresolved), adjacent_matching_ambiguities=ambiguity_count, diagnostics={'component_local_identity': dict(plan.component_local_summary)} if plan.component_local_summary is not None else {})

def window_from_scope(scope: ProcessingScope) -> FineWindow:
    metadata = dict(scope.metadata)
    if metadata.get('schema_version') != WINDOW_SCHEMA:
        raise FineTemporalError('processing scope is not a fine-window scope')
    profile_id = metadata.get('window_profile_id', DEFAULT_FINE_WINDOW_PROFILE_ID)
    profile = fine_window_profile(profile_id)
    if metadata.get('model_io_contract') != profile.model_io_contract:
        raise FineTemporalError('processing scope model I/O contract mismatch')
    declared_profile = metadata.get('window_profile')
    if declared_profile is not None and declared_profile != profile.to_dict():
        raise FineTemporalError('processing scope fine window profile binding mismatch')
    required = ('window_index', 'video_end_us', 'requested_view_start_us', 'requested_view_end_us', 'valid_start_us', 'valid_end_us', 'core_start_us', 'core_end_us', 'valid_start_tick', 'valid_end_tick', 'core_start_tick', 'core_end_tick')
    values: dict[str, int] = {}
    for key in required:
        value = metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise FineTemporalError(f'scope metadata {key} must be an integer')
        values[key] = value
    window = FineWindow(source_id=scope.source_id, scope_id=scope.scope_id, window_profile_id=profile.profile_id, coarse_spans=tuple(metadata.get('coarse_spans') or ()), **values)
    expected_core = TimeSpan(window.core_start_us / 1000000, window.core_end_us / 1000000)
    expected_context = TimeSpan(window.valid_start_us / 1000000, window.valid_end_us / 1000000)
    if scope.core_span != expected_core or scope.context_span != expected_context:
        raise FineTemporalError('float scope spans disagree with bound exact-us geometry')
    return window
