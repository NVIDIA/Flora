# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import asyncio
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from typing import Any
from flora_runtime.schema import AnnotationSource, TemporalSegment, TimeSpan
from flora_runtime.video_segmentation.contracts import EventInterval, ProcessingScope
from flora_runtime.video_segmentation.fine_temporal import DEFAULT_FINE_WINDOW_PROFILE_ID, MAX_BOUNDED_ANCHOR_TICKS, STITCH_TOLERANCE_TICKS, TICK_US, FineFragment, FineSegmentationResult, FineTemporalError, FineWindow, StitchingPolicy, UnresolvedFineEvent, analyze_stitching_seams, ceil_div, fine_window_profile, parse_prediction_with_diagnostics, stitch_predictions, window_from_scope
from flora_runtime.video_segmentation.interface import SegmentTopology, VideoSegmenter
from flora_runtime.video_segmentation.scope_scheduler import FINE_DIRECTIONAL_RETRY_SCOPE_NAMESPACE, schedule_fine_scopes, schedule_recentered_fine_retry_scopes, seconds_to_us
from flora_runtime.worker_transport import JsonPredictor, SubtaskAnnotationAdapterError, source_payload
FINE_REQUEST_SCHEMA = 'subtask_annotation_atomic_fine_batch_request_v1'
FINE_RESPONSE_SCHEMA = 'subtask_annotation_atomic_fine_batch_response_v1'
FINE_QUALITY_POLICY_SCHEMA = 'subtask_annotation_atomic_fine_quality_policy_v1'
PROVIDER_FREE_FINE_COMPLETION_POLICIES = {'adaptive_fine_then_conservative_completion'}
ADAPTIVE_FINE_REPAIR_SCHEMA = 'subtask_annotation_adaptive_fine_repair_v1'
ADAPTIVE_FINE_RETRY_MAX_WINDOWS = 4
ADAPTIVE_FINE_RETRY_HARD_MAX_WINDOWS = 64
CONSERVATIVE_FINE_COMPLETION_SCHEMA = 'subtask_annotation_conservative_fine_completion_v2'
UNRESOLVED_LOCAL_STITCH_SCHEMA = 'subtask_annotation_unresolved_local_stitch_v1'
SEGMENT_LOCAL_REPAIR_ENDPOINT_TOLERANCE_S = 0.5

class FineSegmentationOutputError(SubtaskAnnotationAdapterError):

    def __init__(self, message: str, *, diagnostics: Mapping[str, object], partial_result: FineSegmentationResult | None=None) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)
        self.partial_result = partial_result

@dataclass(frozen=True)
class FineSegmentationQualityPolicy:
    soft_max_segments_per_window: int = 32
    hard_max_segments_per_window: int = 32
    hard_max_overlap_depth: int = 32
    max_adjacent_matching_ambiguities: int = 0
    drop_guard_only_segments: bool = False
    require_event_when_coarse_present: bool = False

    def __post_init__(self) -> None:
        integer_fields = ('soft_max_segments_per_window', 'hard_max_segments_per_window', 'hard_max_overlap_depth', 'max_adjacent_matching_ambiguities')
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a non-negative integer')
        if self.soft_max_segments_per_window > self.hard_max_segments_per_window:
            raise ValueError('soft_max_segments_per_window must not exceed the hard maximum')
        if type(self.drop_guard_only_segments) is not bool:
            raise TypeError('drop_guard_only_segments must be a boolean')
        if type(self.require_event_when_coarse_present) is not bool:
            raise TypeError('require_event_when_coarse_present must be a boolean')

    @classmethod
    def from_value(cls, value: object) -> FineSegmentationQualityPolicy:
        if not isinstance(value, dict):
            raise TypeError('fine quality policy must be an object')
        expected = {'schema_version', 'soft_max_segments_per_window', 'hard_max_segments_per_window', 'hard_max_overlap_depth', 'max_adjacent_matching_ambiguities', 'drop_guard_only_segments', 'require_event_when_coarse_present'}
        missing, unknown = (expected - set(value), set(value) - expected)
        if missing or unknown:
            raise ValueError(f'fine quality policy key mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}')
        if value['schema_version'] != FINE_QUALITY_POLICY_SCHEMA:
            raise ValueError(f'fine quality policy schema must be {FINE_QUALITY_POLICY_SCHEMA!r}')
        return cls(**{key: value[key] for key in expected - {'schema_version'}})

    def to_dict(self) -> dict[str, object]:
        return {'schema_version': FINE_QUALITY_POLICY_SCHEMA, 'soft_max_segments_per_window': self.soft_max_segments_per_window, 'hard_max_segments_per_window': self.hard_max_segments_per_window, 'hard_max_overlap_depth': self.hard_max_overlap_depth, 'max_adjacent_matching_ambiguities': self.max_adjacent_matching_ambiguities, 'drop_guard_only_segments': self.drop_guard_only_segments, 'require_event_when_coarse_present': self.require_event_when_coarse_present}

def _span_payload(span: TimeSpan | None) -> dict[str, float] | None:
    if span is None:
        return None
    return {'start_s': span.start_s, 'end_s': span.end_s}

def _nearest_nonoverlapping_context(ordered: Sequence[EventInterval | TemporalSegment], index: int) -> tuple[TimeSpan | None, TimeSpan | None]:
    target = ordered[index].span
    before_candidates = [item.span for item in ordered if item.span.end_s <= target.start_s]
    after_candidates = [item.span for item in ordered if item.span.start_s >= target.end_s]
    before = max(before_candidates, key=lambda span: (span.end_s, span.start_s), default=None)
    after = min(after_candidates, key=lambda span: (span.start_s, span.end_s), default=None)
    return (before, after)

def events_to_temporal_segments(events: Sequence[EventInterval], *, diagnostics: Mapping[str, object] | None=None, resolution: str='fine') -> tuple[TemporalSegment, ...]:
    ordered = tuple(sorted(events, key=lambda item: (item.span, item.event_id)))
    result: list[TemporalSegment] = []
    for index, event in enumerate(ordered):
        before, after = _nearest_nonoverlapping_context(ordered, index)
        metadata = dict(event.metadata)
        metadata.setdefault('segmentation_provenance', 'fine')
        metadata.setdefault('segmentation_resolution', resolution)
        event_identity = {'fine_event_id': event.event_id} if metadata['segmentation_resolution'] == 'fine' else {'segment_event_id': event.event_id}
        metadata.update({**event_identity, 'source_scope_ids': list(event.source_scope_ids), 'segment_topology': SegmentTopology.CONCURRENT.value, 'u2_context': {'before': _span_payload(before), 'target': _span_payload(event.span), 'after': _span_payload(after)}})
        if diagnostics is not None:
            metadata['fine_segmentation_diagnostics'] = dict(diagnostics)
        result.append(TemporalSegment(span=event.span, metadata=metadata))
    return tuple(result)

def _overlap_statistics(events: Sequence[EventInterval]) -> tuple[int, int]:
    overlap_pairs = 0
    for left_index, left in enumerate(events):
        overlap_pairs += sum((min(left.span.end_s, right.span.end_s) > max(left.span.start_s, right.span.start_s) for right in events[left_index + 1:]))
    points = [(item.span.start_s, 1) for item in events] + [(item.span.end_s, -1) for item in events]
    depth = current = 0
    for _, delta in sorted(points, key=lambda item: (item[0], item[1])):
        current += delta
        depth = max(depth, current)
    return (overlap_pairs, depth)

def _exact_duplicate_intervals(events: Sequence[EventInterval]) -> list[dict[str, object]]:
    counts = Counter(((item.span.start_s, item.span.end_s) for item in events))
    return [{'start_s': start_s, 'end_s': end_s, 'count': count} for (start_s, end_s), count in sorted(counts.items()) if count > 1]

def _diagnostic_sha256(value: Mapping[str, object]) -> str:
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode()
    return hashlib.sha256(payload).hexdigest()

def _inference_context(source: AnnotationSource, coarse: Sequence[TemporalSegment], *, fine_prior_mode: str) -> dict[str, object]:
    timestamp_prior = {'schema_version': 'subtask_annotation_coarse_timestamp_prior_v1', 'source_id': source.source_id, 'spans': [{'start_s': segment.span.start_s, 'end_s': segment.span.end_s} for segment in coarse]}
    return {'fine_prior_mode': fine_prior_mode, 'coarse_prior_count': len(coarse), 'coarse_prior_sha256': _diagnostic_sha256(timestamp_prior)}

def _with_inference_context(diagnostics: Mapping[str, object], context: Mapping[str, object]) -> dict[str, object]:
    result = dict(diagnostics)
    result.update(context)
    result['diagnostics_sha256'] = _diagnostic_sha256({key: value for key, value in result.items() if key != 'diagnostics_sha256'})
    return result

@dataclass(frozen=True)
class _FineWindowBatchPrediction:
    fragments_by_window: Mapping[int, tuple[FineFragment, ...]]
    raw_segment_counts: tuple[int, ...]
    emitted_segment_counts: tuple[int, ...]
    guard_only_fragments: tuple[tuple[FineFragment, ...], ...]

@dataclass(frozen=True)
class _FineRetryPrediction:
    candidates_by_window: Mapping[int, tuple[TemporalSegment, ...]]
    diagnostics: Mapping[str, object]

@dataclass(frozen=True)
class FineSubtaskSegmenter:
    predictor: JsonPredictor
    model_binding: Mapping[str, Any]
    profile_id: str = 'human_ego_atomic_v0_2'
    window_profile_id: str = DEFAULT_FINE_WINDOW_PROFILE_ID
    fail_on_unresolved: bool = True
    batch_concurrency: int = 1
    unresolved_stitch_tolerance_ticks: int = STITCH_TOLERANCE_TICKS
    quality_policy: FineSegmentationQualityPolicy = field(default_factory=FineSegmentationQualityPolicy)
    segment_topology = SegmentTopology.CONCURRENT

    def __post_init__(self) -> None:
        binding = dict(self.model_binding)
        if not binding:
            raise ValueError('model_binding must not be empty')
        window_profile = fine_window_profile(self.window_profile_id)
        declared_model_io_contract = binding.get('model_io_contract')
        if declared_model_io_contract is not None and declared_model_io_contract != window_profile.model_io_contract:
            raise ValueError('model_binding.model_io_contract must match the selected fine window profile')
        if isinstance(self.batch_concurrency, bool) or not isinstance(self.batch_concurrency, int) or self.batch_concurrency <= 0:
            raise ValueError('batch_concurrency must be a positive integer')
        if isinstance(self.unresolved_stitch_tolerance_ticks, bool) or not isinstance(self.unresolved_stitch_tolerance_ticks, int) or (not STITCH_TOLERANCE_TICKS <= self.unresolved_stitch_tolerance_ticks <= MAX_BOUNDED_ANCHOR_TICKS):
            raise ValueError(f'unresolved_stitch_tolerance_ticks must be an integer in [{STITCH_TOLERANCE_TICKS}, {MAX_BOUNDED_ANCHOR_TICKS}]')
        object.__setattr__(self, 'model_binding', binding)

    def _stitch_with_unresolved_local_candidates(self, windows: Sequence[FineWindow], fragments_by_window: Mapping[int, Sequence[FineFragment]]) -> FineSegmentationResult:
        strict = stitch_predictions(windows, fragments_by_window)
        tolerance_ticks = self.unresolved_stitch_tolerance_ticks
        if not strict.unresolved or tolerance_ticks == STITCH_TOLERANCE_TICKS:
            return strict
        policy = StitchingPolicy.component_local_bounded_anchor(endpoint_tolerance_ticks=tolerance_ticks)
        seam_diagnostics = analyze_stitching_seams(windows, fragments_by_window, policy=policy)
        candidate_rows: list[dict[str, object]] = []
        allowlist: set[tuple[int, int, int]] = set()
        seams = seam_diagnostics.get('seams')
        if not isinstance(seams, list):
            raise FineTemporalError('component-local seam diagnostics are missing seam rows')
        for seam_index, seam in enumerate(seams):
            if not isinstance(seam, dict):
                raise FineTemporalError('component-local seam diagnostic row must be an object')
            local = seam.get('component_local_identity')
            if not isinstance(local, dict):
                raise FineTemporalError('component-local seam evidence is missing')
            accepted_pairs = local.get('accepted_pairs')
            if not isinstance(accepted_pairs, list):
                raise FineTemporalError('component-local accepted-pair evidence is invalid')
            for pair in accepted_pairs:
                if not isinstance(pair, dict):
                    raise FineTemporalError('component-local accepted pair must be an object')
                left_id = pair.get('left_observation_id')
                right_id = pair.get('right_observation_id')
                if isinstance(left_id, bool) or not isinstance(left_id, int) or isinstance(right_id, bool) or (not isinstance(right_id, int)):
                    raise FineTemporalError('component-local accepted pair identity is invalid')
                allowlist.add((seam_index, left_id, right_id))
                candidate_rows.append({'seam_index': seam_index, **pair})
        if not allowlist:
            return strict
        relaxed = stitch_predictions(windows, fragments_by_window, policy=policy, component_pair_allowlist=allowlist)
        if len(relaxed.unresolved) >= len(strict.unresolved):
            raise FineTemporalError('component-local unresolved stitching did not reduce unresolved components')
        unresolved_by_observation = {observation_id: component_index for component_index, component in enumerate(strict.unresolved) for observation_id in component.observation_ids}
        resolved_by_observation = {observation_id: event for event in strict.events for observation_id in event.metadata.get('observation_ids', ()) if isinstance(observation_id, int)}
        candidates: list[dict[str, object]] = []
        seen_components: set[int] = set()
        for row in candidate_rows:
            partial_root = row.get('partial_component_root')
            resolved_root = row.get('resolved_component_root')
            if isinstance(partial_root, bool) or not isinstance(partial_root, int) or isinstance(resolved_root, bool) or (not isinstance(resolved_root, int)):
                raise FineTemporalError('component-local candidate roots are invalid')
            component_index = unresolved_by_observation.get(partial_root)
            resolved_event = resolved_by_observation.get(resolved_root)
            if component_index is None or resolved_event is None:
                raise FineTemporalError('component-local candidate does not map to one strict unresolved and one strict resolved component')
            if component_index in seen_components:
                raise FineTemporalError('component-local candidate discovery produced multiple targets for one unresolved component')
            seen_components.add(component_index)
            candidates.append({'partial_component_index': component_index, 'resolved_event_id': resolved_event.event_id, 'resolved_event_span': _span_payload(resolved_event.span), 'pair': row})
        local_stitch_diagnostics = {'schema_version': UNRESOLVED_LOCAL_STITCH_SCHEMA, 'mode': 'candidate_only_requires_adaptive_fine_confirmation_v1', 'strict_tolerance_ticks': STITCH_TOLERANCE_TICKS, 'candidate_tolerance_ticks': tolerance_ticks, 'tick_us': TICK_US, 'strict_unresolved_component_count': len(strict.unresolved), 'candidate_component_count': len(candidates), 'potential_remaining_unresolved_component_count': len(relaxed.unresolved), 'potential_absorbed_unresolved_component_count': len(strict.unresolved) - len(relaxed.unresolved), 'accepted_pair_count': len(allowlist), 'resolved_geometry_changed': False, 'semantic_quality_established': False, 'clean_acceptance_requires_adaptive_fine_confirmation': True, 'candidates': candidates}
        return replace(strict, diagnostics={**dict(strict.diagnostics), 'unresolved_local_stitch': local_stitch_diagnostics})

    def _assess_result(self, result: FineSegmentationResult, *, inference_arm: str, raw_segment_counts: Sequence[int], emitted_segment_counts: Sequence[int], guard_only_fragments: Sequence[Sequence[object]]) -> FineSegmentationResult:
        overlap_pairs, overlap_depth = _overlap_statistics(result.events)
        exact_duplicates = _exact_duplicate_intervals(result.events)
        policy = self.quality_policy
        window_profile = fine_window_profile(self.window_profile_id)
        violations: list[dict[str, object]] = []
        for index, count in enumerate(raw_segment_counts):
            if count > policy.hard_max_segments_per_window:
                violations.append({'code': 'hard_segment_count', 'window_index': index, 'observed': count, 'maximum': policy.hard_max_segments_per_window, 'failure_class': 'delivery_budget_exceeded'})
        if overlap_depth > policy.hard_max_overlap_depth:
            violations.append({'code': 'overlap_depth', 'observed': overlap_depth, 'maximum': policy.hard_max_overlap_depth, 'failure_class': 'delivery_budget_exceeded'})
        if exact_duplicates:
            violations.append({'code': 'exact_duplicate_intervals', 'observed': exact_duplicates, 'failure_class': 'structural_invalid'})
        if result.adjacent_matching_ambiguities > policy.max_adjacent_matching_ambiguities:
            violations.append({'code': 'stitch_ambiguity', 'observed': result.adjacent_matching_ambiguities, 'maximum': policy.max_adjacent_matching_ambiguities, 'failure_class': 'structural_invalid'})
        if self.fail_on_unresolved and result.unresolved:
            violations.append({'code': 'unresolved_components', 'observed': len(result.unresolved), 'maximum': 0, 'failure_class': 'structural_invalid'})
        diagnostics: dict[str, object] = {'schema_version': 'subtask_annotation_atomic_fine_diagnostics_v1', 'inference_arm': inference_arm, 'window_profile_id': window_profile.profile_id, 'window_profile': window_profile.to_dict(), 'model_io_contract': window_profile.model_io_contract, 'target_granularity': window_profile.target_granularity, 'policy': policy.to_dict(), 'raw_segment_counts': list(raw_segment_counts), 'emitted_segment_counts': list(emitted_segment_counts), 'guard_only_dropped': [[asdict(fragment) for fragment in fragments] for fragments in guard_only_fragments], 'soft_limit_exceeded_windows': [index for index, count in enumerate(raw_segment_counts) if count > policy.soft_max_segments_per_window], 'resolved_event_count': len(result.events), 'unresolved_components': [asdict(item) for item in result.unresolved], 'adjacent_matching_ambiguities': result.adjacent_matching_ambiguities, 'overlap_pair_count': overlap_pairs, 'max_overlap_depth': overlap_depth, 'exact_duplicate_intervals': exact_duplicates, 'acceptance_failure_classes': sorted({str(item['failure_class']) for item in violations}), 'violations': violations}
        if result.diagnostics:
            diagnostics['stitch_diagnostics'] = dict(result.diagnostics)
        diagnostics['diagnostics_sha256'] = _diagnostic_sha256(diagnostics)
        assessed = replace(result, diagnostics=diagnostics)
        if violations:
            codes = ','.join((str(item['code']) for item in violations))
            raise FineSegmentationOutputError(f'fine segmentation failed output acceptance gate: {codes}', diagnostics=diagnostics, partial_result=assessed)
        return assessed

    def _predict_scheduled_windows(self, source: AnnotationSource, windows: Sequence[FineWindow]) -> _FineWindowBatchPrediction:
        return asyncio.run(self._predict_scheduled_windows_async(source, windows))

    async def _predict_scheduled_windows_async(self, source: AnnotationSource, windows: Sequence[FineWindow]) -> _FineWindowBatchPrediction:
        request = {'schema_version': FINE_REQUEST_SCHEMA, 'profile_id': self.profile_id, 'model_binding': dict(self.model_binding), 'source': source_payload(source), 'windows': [window.worker_payload() for window in windows], 'max_concurrency': min(self.batch_concurrency, len(windows))}
        apredict = getattr(self.predictor, 'apredict', None)
        if callable(apredict):
            response = await apredict(request)
        else:
            response = await asyncio.to_thread(self.predictor.predict, request)
        required_keys = {'schema_version', 'source_id', 'model_io_contract', 'model_binding', 'predictions'}
        if set(response) != required_keys:
            raise SubtaskAnnotationAdapterError('fine-segmentation response must contain exactly schema_version, source_id, model_io_contract, model_binding, predictions')
        if response['schema_version'] != FINE_RESPONSE_SCHEMA:
            raise SubtaskAnnotationAdapterError('fine-segmentation response schema mismatch')
        if response['source_id'] != source.source_id:
            raise SubtaskAnnotationAdapterError('fine-segmentation response identity mismatch')
        if response['model_io_contract'] != fine_window_profile(self.window_profile_id).model_io_contract:
            raise SubtaskAnnotationAdapterError('fine-segmentation model I/O contract mismatch')
        if response['model_binding'] != self.model_binding:
            raise SubtaskAnnotationAdapterError('fine-segmentation worker model_binding mismatch')
        prediction_rows = response['predictions']
        if not isinstance(prediction_rows, list) or len(prediction_rows) != len(windows):
            raise SubtaskAnnotationAdapterError('fine-segmentation predictions must exactly cover every requested window')
        expected_scope_ids = [window.scope_id for window in windows]
        observed_scope_ids: list[str] = []
        fragments_by_window = {}
        raw_segment_counts: list[int] = []
        emitted_segment_counts: list[int] = []
        guard_only_fragments: list[Sequence[object]] = []
        for window, prediction in zip(windows, prediction_rows, strict=True):
            if not isinstance(prediction, dict) or set(prediction) != {'scope_id', 'prediction_json'}:
                raise SubtaskAnnotationAdapterError('each fine-segmentation prediction must contain exactly scope_id and prediction_json')
            observed_scope_ids.append(prediction['scope_id'])
            if prediction['scope_id'] != window.scope_id:
                raise SubtaskAnnotationAdapterError('fine-segmentation prediction order or scope differs')
            prediction_json = prediction['prediction_json']
            if not isinstance(prediction_json, str):
                raise SubtaskAnnotationAdapterError('fine-segmentation prediction_json must be a string')
            try:
                parsed = parse_prediction_with_diagnostics(prediction_json, window, drop_guard_only=self.quality_policy.drop_guard_only_segments)
            except FineTemporalError as exc:
                raise FineSegmentationOutputError(f'fine-segmentation window {window.window_index} returned an invalid prediction: {exc}', diagnostics={'schema_version': 'subtask_annotation_atomic_fine_diagnostics_v1', 'stage': 'parse_prediction', 'window_index': window.window_index, 'error': str(exc)}) from exc
            fragments_by_window[window.window_index] = parsed.fragments
            raw_segment_counts.append(len(parsed.fragments) + len(parsed.guard_only_fragments))
            emitted_segment_counts.append(len(parsed.fragments))
            guard_only_fragments.append(parsed.guard_only_fragments)
        if observed_scope_ids != expected_scope_ids or len(set(observed_scope_ids)) != len(observed_scope_ids):
            raise SubtaskAnnotationAdapterError('fine-segmentation response scope coverage differs')
        return _FineWindowBatchPrediction(fragments_by_window={index: tuple(fragments) for index, fragments in fragments_by_window.items()}, raw_segment_counts=tuple(raw_segment_counts), emitted_segment_counts=tuple(emitted_segment_counts), guard_only_fragments=tuple((tuple(items) for items in guard_only_fragments)))

    def predict_repair_candidates(self, source: AnnotationSource, windows: Sequence[FineWindow], *, arm: str='adaptive_fine_repair') -> _FineRetryPrediction:
        return asyncio.run(self.predict_repair_candidates_async(source, windows, arm=arm))

    async def predict_repair_candidates_async(self, source: AnnotationSource, windows: Sequence[FineWindow], *, arm: str='adaptive_fine_repair') -> _FineRetryPrediction:
        batch = await self._predict_scheduled_windows_async(source, windows)
        hard_limit_violations = [{'code': 'hard_segment_count', 'window_index': window.window_index, 'observed': count, 'maximum': self.quality_policy.hard_max_segments_per_window, 'failure_class': 'delivery_budget_exceeded'} for window, count in zip(windows, batch.raw_segment_counts, strict=True) if count > self.quality_policy.hard_max_segments_per_window]
        diagnostics: dict[str, object] = {'schema_version': ADAPTIVE_FINE_REPAIR_SCHEMA, 'stage': 'fine_retry_prediction', 'inference_arm': arm, 'worker_batch_call_count': 1, 'retry_window_count': len(windows), 'raw_segment_counts': list(batch.raw_segment_counts), 'emitted_segment_counts': list(batch.emitted_segment_counts), 'guard_only_dropped': [[asdict(fragment) for fragment in fragments] for fragments in batch.guard_only_fragments], 'violations': hard_limit_violations}
        diagnostics['diagnostics_sha256'] = _diagnostic_sha256(diagnostics)
        if hard_limit_violations:
            raise FineSegmentationOutputError('adaptive fine retry exceeded the per-window segment budget', diagnostics=diagnostics)
        candidates_by_window: dict[int, tuple[TemporalSegment, ...]] = {}
        for window in windows:
            candidates: list[TemporalSegment] = []
            for fragment in batch.fragments_by_window[window.window_index]:
                start_us = (window.global_tick_offset + fragment.s) * TICK_US
                end_us = (window.global_tick_offset + fragment.e) * TICK_US
                physical_start_us = max(window.valid_start_us, start_us)
                physical_end_us = min(window.valid_end_us, end_us)
                if physical_end_us <= physical_start_us:
                    raise FineSegmentationOutputError('adaptive fine retry produced an empty physical source interval', diagnostics={**diagnostics, 'window_index': window.window_index, 'fragment': asdict(fragment), 'model_tick_span_us': {'start': start_us, 'end': end_us}, 'physical_valid_span_us': {'start': window.valid_start_us, 'end': window.valid_end_us}})
                physical_clip = None
                if physical_start_us != start_us or physical_end_us != end_us:
                    physical_clip = {'contract': 'model_tick_span_intersect_physical_valid_span_v1', 'model_tick_span_us': {'start': start_us, 'end': end_us}, 'physical_valid_span_us': {'start': window.valid_start_us, 'end': window.valid_end_us}, 'clipped_start': physical_start_us != start_us, 'clipped_end': physical_end_us != end_us}
                candidates.append(TemporalSegment(TimeSpan(physical_start_us / 1000000, physical_end_us / 1000000), {'scope_clipped_start': fragment.lo, 'scope_clipped_end': fragment.ro, 'fine_retry_scope_id': window.scope_id, 'fine_retry_window_index': window.window_index, **({'fine_retry_physical_clip': physical_clip} if physical_clip is not None else {})}))
            candidates_by_window[window.window_index] = tuple(candidates)
        return _FineRetryPrediction(candidates_by_window=candidates_by_window, diagnostics=diagnostics)

    def segment_with_priors(self, source: AnnotationSource, coarse_proposals: Sequence[ProcessingScope], *, arm: str='no_prior_global_scan') -> FineSegmentationResult:
        return asyncio.run(self.segment_with_priors_async(source, coarse_proposals, arm=arm))

    async def segment_with_priors_async(self, source: AnnotationSource, coarse_proposals: Sequence[ProcessingScope], *, arm: str='no_prior_global_scan') -> FineSegmentationResult:
        if source.duration_s is None:
            raise SubtaskAnnotationAdapterError('fine segmentation requires exact source duration_s')
        scopes = schedule_fine_scopes(source_id=source.source_id, duration_s=source.duration_s, coarse_proposals=coarse_proposals, arm=arm, window_profile_id=self.window_profile_id)
        windows = tuple((window_from_scope(scope) for scope in scopes))
        batch = await self._predict_scheduled_windows_async(source, windows)
        result = self._stitch_with_unresolved_local_candidates(windows, batch.fragments_by_window)
        return self._assess_result(result, inference_arm=arm, raw_segment_counts=batch.raw_segment_counts, emitted_segment_counts=batch.emitted_segment_counts, guard_only_fragments=batch.guard_only_fragments)

    def segment(self, source: AnnotationSource) -> tuple[TemporalSegment, ...]:
        return asyncio.run(self.segment_async(source))

    async def segment_async(self, source: AnnotationSource) -> tuple[TemporalSegment, ...]:
        result = await self.segment_with_priors_async(source, (), arm='no_prior_global_scan')
        return events_to_temporal_segments(result.events, diagnostics=result.diagnostics)

@dataclass(frozen=True)
class _AcceptedFineRetry:
    component_index: int
    window: FineWindow
    candidate: TemporalSegment
    repaired_span: TimeSpan
    repair_mode: str = 'boundary_completion'
    confirmation_target_event_id: str | None = None
    local_stitch_candidate: Mapping[str, object] | None = None

@dataclass(frozen=True)
class _MatchedFineRetryCandidate:
    candidate: TemporalSegment
    repaired_span: TimeSpan
    repair_mode: str
    local_stitch_candidate: Mapping[str, object] | None = None

def _tick_seconds(value: int) -> float:
    return value * TICK_US / 1000000

def _same_event_geometry(left: TimeSpan, right: TimeSpan, *, tolerance_s: float=SEGMENT_LOCAL_REPAIR_ENDPOINT_TOLERANCE_S) -> bool:
    return abs(left.start_s - right.start_s) <= tolerance_s and abs(left.end_s - right.end_s) <= tolerance_s

def _unresolved_local_stitch_candidates(diagnostics: Mapping[str, object]) -> dict[int, dict[str, object]]:
    stitch_diagnostics = diagnostics.get('stitch_diagnostics')
    if not isinstance(stitch_diagnostics, dict):
        return {}
    local = stitch_diagnostics.get('unresolved_local_stitch')
    if not isinstance(local, dict):
        return {}
    candidates = local.get('candidates')
    if not isinstance(candidates, list):
        raise SubtaskAnnotationAdapterError('unresolved-local stitch diagnostics contain an invalid candidate ledger')
    by_component: dict[int, dict[str, object]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise SubtaskAnnotationAdapterError('unresolved-local stitch candidate must be an object')
        component_index = candidate.get('partial_component_index')
        event_id = candidate.get('resolved_event_id')
        span = candidate.get('resolved_event_span')
        if isinstance(component_index, bool) or not isinstance(component_index, int) or component_index < 0 or (not isinstance(event_id, str)) or (not event_id) or (not isinstance(span, dict)) or (set(span) != {'start_s', 'end_s'}):
            raise SubtaskAnnotationAdapterError('unresolved-local stitch candidate binding is invalid')
        if component_index in by_component:
            raise SubtaskAnnotationAdapterError('unresolved-local stitch candidate ledger repeats a component')
        by_component[component_index] = dict(candidate)
    return by_component

def _local_stitch_target_span(candidate: Mapping[str, object]) -> TimeSpan:
    payload = candidate.get('resolved_event_span')
    if not isinstance(payload, dict):
        raise SubtaskAnnotationAdapterError('unresolved-local stitch candidate target span is missing')
    start_s, end_s = (payload.get('start_s'), payload.get('end_s'))
    if isinstance(start_s, bool) or not isinstance(start_s, int | float) or isinstance(end_s, bool) or (not isinstance(end_s, int | float)):
        raise SubtaskAnnotationAdapterError('unresolved-local stitch candidate target span is invalid')
    return TimeSpan(float(start_s), float(end_s))

def _retry_confirms_local_stitch(retry_candidate: TemporalSegment, local_candidate: Mapping[str, object]) -> bool:
    return _same_event_geometry(retry_candidate.span, _local_stitch_target_span(local_candidate), tolerance_s=STITCH_TOLERANCE_TICKS * TICK_US / 1000000)

def _component_observation_span(component: UnresolvedFineEvent) -> TimeSpan:
    return TimeSpan(_tick_seconds(min((item[0] for item in component.observation_spans_ticks))), _tick_seconds(max((item[1] for item in component.observation_spans_ticks))))

def _endpoint_consensus_ticks(values: Sequence[int]) -> int:
    ordered = sorted(values)
    if not ordered:
        raise SubtaskAnnotationAdapterError('cannot form endpoint consensus without evidence')
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle] + 1) // 2

def _conservative_component_completion_span(source: AnnotationSource, component: UnresolvedFineEvent) -> tuple[TimeSpan, str]:
    if source.duration_s is None:
        raise SubtaskAnnotationAdapterError('conservative fine completion requires exact source duration_s')
    if not component.observation_spans_ticks:
        raise SubtaskAnnotationAdapterError('conservative fine completion requires at least one observed fragment')
    observed_start_tick = min((item[0] for item in component.observation_spans_ticks))
    observed_end_tick = max((item[1] for item in component.observation_spans_ticks))
    has_start = bool(component.closed_start_ticks)
    has_end = bool(component.closed_end_ticks)
    start_tick = _endpoint_consensus_ticks(component.closed_start_ticks) if has_start else observed_start_tick
    end_tick = _endpoint_consensus_ticks(component.closed_end_ticks) if has_end else observed_end_tick
    strategy = {(True, False): 'closed_start_plus_observed_censored_end_v1', (False, True): 'observed_censored_start_plus_closed_end_v1', (True, True): 'closed_endpoint_consensus_v1', (False, False): 'observed_fragment_envelope_v1'}[has_start, has_end]
    start_s = max(0.0, _tick_seconds(start_tick))
    end_s = min(source.duration_s, _tick_seconds(end_tick))
    if end_s <= start_s:
        start_s = max(0.0, _tick_seconds(observed_start_tick))
        end_s = min(source.duration_s, _tick_seconds(observed_end_tick))
        strategy = 'observed_fragment_envelope_after_endpoint_disagreement_v1'
    if end_s <= start_s:
        raise SubtaskAnnotationAdapterError('conservative fine completion produced an empty physical interval')
    return (TimeSpan(start_s, end_s), strategy)

def _component_boundary_completion_span(component: UnresolvedFineEvent, segment: TemporalSegment) -> TimeSpan | None:
    has_start = bool(component.closed_start_ticks)
    has_end = bool(component.closed_end_ticks)
    if has_start == has_end:
        return None
    observation = _component_observation_span(component)
    if has_start:
        if segment.metadata.get('scope_clipped_end') is True:
            return None
        if not segment.span.start_s < observation.end_s <= segment.span.end_s + 1e-09:
            return None
        start_s = _tick_seconds(_endpoint_consensus_ticks(component.closed_start_ticks))
        end_s = segment.span.end_s
    else:
        if segment.metadata.get('scope_clipped_start') is True:
            return None
        if not segment.span.start_s <= observation.start_s + 1e-09 < segment.span.end_s:
            return None
        start_s = segment.span.start_s
        end_s = _tick_seconds(_endpoint_consensus_ticks(component.closed_end_ticks))
    if end_s <= start_s:
        return None
    return TimeSpan(start_s, end_s)

def _match_fine_retry_candidates(component_index: int, component: UnresolvedFineEvent, candidates: Sequence[TemporalSegment], local_stitch_candidates: Mapping[int, Mapping[str, object]]) -> tuple[tuple[_MatchedFineRetryCandidate, ...], tuple[dict[str, object], ...]]:
    local_candidate = local_stitch_candidates.get(component_index)
    matching: list[_MatchedFineRetryCandidate] = []
    ignored: list[dict[str, object]] = []
    for candidate in candidates:
        if local_candidate is not None and _retry_confirms_local_stitch(candidate, local_candidate):
            matching.append(_MatchedFineRetryCandidate(candidate=candidate, repaired_span=_local_stitch_target_span(local_candidate), repair_mode='unresolved_local_stitch_confirmed', local_stitch_candidate=local_candidate))
            continue
        repaired_span = _component_boundary_completion_span(component, candidate)
        if repaired_span is not None:
            matching.append(_MatchedFineRetryCandidate(candidate=candidate, repaired_span=repaired_span, repair_mode='boundary_completion'))
            continue
        ignored.append({'span': _span_payload(candidate.span), 'scope_clipped_start': candidate.metadata.get('scope_clipped_start'), 'scope_clipped_end': candidate.metadata.get('scope_clipped_end')})
    return (tuple(matching), tuple(ignored))

def _covering_span(left: TimeSpan, right: TimeSpan) -> TimeSpan:
    return TimeSpan(min(left.start_s, right.start_s), max(left.end_s, right.end_s))

def _unsupported_unresolved_component_indices(source: AnnotationSource, unresolved: Sequence[UnresolvedFineEvent]) -> list[int]:
    return [index for index, component in enumerate(unresolved) if component.matching_ambiguous or (component.closed_start_ticks and component.closed_end_ticks) or (not component.closed_start_ticks and (not component.closed_end_ticks)) or (not component.closed_start_ticks and min((item[0] for item in component.observation_spans_ticks)) <= 0) or (not component.closed_end_ticks and source.duration_s is not None and (_tick_seconds(max((item[1] for item in component.observation_spans_ticks))) >= source.duration_s - 1e-09))]

def _adaptive_fine_failure(exc: FineSegmentationOutputError, *, code: str, detail: Mapping[str, object]) -> FineSegmentationOutputError:
    diagnostics = dict(exc.diagnostics)
    failures = list(diagnostics.get('adaptive_fine_repair_failures', []))
    failures.append({'code': code, **dict(detail)})
    diagnostics['adaptive_fine_repair_failures'] = failures
    diagnostics['diagnostics_sha256'] = _diagnostic_sha256({key: value for key, value in diagnostics.items() if key != 'diagnostics_sha256'})
    return FineSegmentationOutputError(f'fine segmentation could not be repaired by bounded fine retry: {code}', diagnostics=diagnostics, partial_result=exc.partial_result)

def _remaining_adaptive_fine_error(exc: FineSegmentationOutputError, result: FineSegmentationResult, *, repaired_events: tuple[EventInterval, ...], accepted_component_indices: Sequence[int], local_stitch_candidates: Mapping[int, Mapping[str, object]], deferred_failures: Sequence[Mapping[str, object]], repair_payload: Mapping[str, object], inference_context: Mapping[str, object]) -> FineSegmentationOutputError:
    accepted = set(accepted_component_indices)
    remaining_indices = [index for index in range(len(result.unresolved)) if index not in accepted]
    if not remaining_indices:
        raise SubtaskAnnotationAdapterError('remaining adaptive Fine error requires at least one unresolved component')
    original_rows = exc.diagnostics.get('unresolved_components')
    rows_are_aligned = isinstance(original_rows, list) and len(original_rows) == len(result.unresolved)
    remaining_rows = [original_rows[index] for index in remaining_indices] if rows_are_aligned else []
    repaired_rows = [original_rows[index] for index in sorted(accepted)] if rows_are_aligned else []
    reduced_index = {old_index: new_index for new_index, old_index in enumerate(remaining_indices)}
    remaining_local_candidates = []
    for old_index in remaining_indices:
        candidate = local_stitch_candidates.get(old_index)
        if candidate is None:
            continue
        rebound = dict(candidate)
        rebound['partial_component_index'] = reduced_index[old_index]
        rebound['original_partial_component_index'] = old_index
        remaining_local_candidates.append(rebound)
    diagnostics = dict(exc.diagnostics)
    failures = list(diagnostics.get('adaptive_fine_repair_failures', []))
    failures.extend((dict(item) for item in deferred_failures))
    diagnostics['adaptive_fine_repair_failures'] = failures
    diagnostics['pre_repair_violations'] = list(diagnostics.get('violations', []))
    diagnostics['repaired_unresolved_components'] = repaired_rows
    diagnostics['unresolved_components'] = remaining_rows
    violations = [dict(item) for item in diagnostics.get('violations', []) if isinstance(item, dict)]
    for violation in violations:
        if violation.get('code') == 'unresolved_components':
            violation['observed'] = len(remaining_indices)
    diagnostics['violations'] = violations
    diagnostics['resolved_event_count'] = len(repaired_events)
    diagnostics['overlap_pair_count'] = repair_payload['post_repair_overlap_pair_count']
    diagnostics['max_overlap_depth'] = repair_payload['post_repair_max_overlap_depth']
    diagnostics['exact_duplicate_intervals'] = repair_payload['post_repair_exact_duplicate_intervals']
    diagnostics['adaptive_fine_repair'] = dict(repair_payload)
    diagnostics['adaptive_fine_repair_sha256'] = _diagnostic_sha256(repair_payload)
    stitch_diagnostics = dict(diagnostics.get('stitch_diagnostics', {}))
    local_diagnostics = dict(stitch_diagnostics.get('unresolved_local_stitch', {}))
    if local_diagnostics:
        local_diagnostics.update({'accepted_pair_count': len(remaining_local_candidates), 'candidate_component_count': len(remaining_local_candidates), 'candidates': remaining_local_candidates, 'potential_absorbed_unresolved_component_count': len(remaining_local_candidates), 'potential_remaining_unresolved_component_count': len(remaining_indices) - len(remaining_local_candidates), 'strict_unresolved_component_count': len(remaining_indices)})
        stitch_diagnostics['unresolved_local_stitch'] = local_diagnostics
        diagnostics['stitch_diagnostics'] = stitch_diagnostics
    diagnostics = _with_inference_context(diagnostics, inference_context)
    partial_result = FineSegmentationResult(events=repaired_events, fragments=result.fragments, unresolved=tuple((result.unresolved[index] for index in remaining_indices)), adjacent_matching_ambiguities=result.adjacent_matching_ambiguities, diagnostics=diagnostics)
    return FineSegmentationOutputError('bounded fine retry partially repaired the source; remaining components require conservative completion', diagnostics=diagnostics, partial_result=partial_result)

@dataclass(frozen=True)
class HierarchicalSubtaskSegmenter:
    coarse_segmenter: VideoSegmenter
    fine_segmenter: FineSubtaskSegmenter
    fine_prior_mode: str = 'none'
    fine_failure_policy: str = 'adaptive_fine_then_conservative_completion'
    fine_output_coverage_policy: str = 'allow_gaps'
    adaptive_fine_retry_max_windows: int = ADAPTIVE_FINE_RETRY_MAX_WINDOWS
    segment_topology = SegmentTopology.CONCURRENT

    def __post_init__(self):
        if self.fine_prior_mode != 'none' or self.fine_failure_policy != 'adaptive_fine_then_conservative_completion' or self.fine_output_coverage_policy != 'allow_gaps':
            raise ValueError('Use the released inference workflow configuration.')
        if isinstance(self.adaptive_fine_retry_max_windows, bool) or not isinstance(self.adaptive_fine_retry_max_windows, int) or (not 1 <= self.adaptive_fine_retry_max_windows <= ADAPTIVE_FINE_RETRY_HARD_MAX_WINDOWS):
            raise ValueError('Invalid adaptive inference retry budget')

    def adaptive_fine_repair(self, source: AnnotationSource, exc: FineSegmentationOutputError, *, inference_context: Mapping[str, object] | None=None) -> tuple[TemporalSegment, ...]:
        return asyncio.run(self.adaptive_fine_repair_async(source, exc, inference_context=inference_context))

    async def adaptive_fine_repair_async(self, source: AnnotationSource, exc: FineSegmentationOutputError, *, inference_context: Mapping[str, object] | None=None) -> tuple[TemporalSegment, ...]:
        if inference_context is None:
            inference_context = _inference_context(source, (), fine_prior_mode='none')
        result = exc.partial_result
        violation_codes = [str(item.get('code')) for item in exc.diagnostics.get('violations', []) if isinstance(item, dict)]
        if result is None or not result.unresolved or violation_codes != ['unresolved_components']:
            raise _adaptive_fine_failure(exc, code='ineligible_fine_failure', detail={'violation_codes': violation_codes}) from exc
        if source.duration_s is None:
            raise _adaptive_fine_failure(exc, code='duration_unavailable', detail={}) from exc
        progressive_provider_free = self.fine_failure_policy in PROVIDER_FREE_FINE_COMPLETION_POLICIES
        window_profile = fine_window_profile(self.fine_segmenter.window_profile_id)
        local_stitch_candidates = _unresolved_local_stitch_candidates(exc.diagnostics)
        unsupported_components = _unsupported_unresolved_component_indices(source, result.unresolved)
        if unsupported_components and (not progressive_provider_free):
            raise _adaptive_fine_failure(exc, code='unsupported_unresolved_geometry', detail={'component_indices': unsupported_components}) from exc
        if len(result.unresolved) > self.adaptive_fine_retry_max_windows and (not progressive_provider_free):
            raise _adaptive_fine_failure(exc, code='retry_budget_exceeded', detail={'unresolved_component_count': len(result.unresolved), 'max_retry_windows': self.adaptive_fine_retry_max_windows}) from exc
        deferred_failures: list[dict[str, object]] = []
        unsupported_set = set(unsupported_components)
        if unsupported_components:
            deferred_failures.append({'code': 'unsupported_unresolved_geometry', 'component_indices': unsupported_components})
        eligible_component_indices = [index for index in range(len(result.unresolved)) if index not in unsupported_set]
        eligible_component_indices.sort(key=lambda index: (index not in local_stitch_candidates, index))
        selected_component_indices = eligible_component_indices[:self.adaptive_fine_retry_max_windows]
        budget_deferred_indices = eligible_component_indices[self.adaptive_fine_retry_max_windows:]
        if budget_deferred_indices:
            deferred_failures.append({'code': 'retry_budget_exceeded', 'unresolved_component_count': len(result.unresolved), 'max_retry_windows': self.adaptive_fine_retry_max_windows, 'deferred_component_indices': budget_deferred_indices, 'candidate_priority': 'unresolved_local_stitch_first_then_source_order_v1'})
        if not selected_component_indices:
            diagnostics = dict(exc.diagnostics)
            failures = list(diagnostics.get('adaptive_fine_repair_failures', []))
            failures.extend(deferred_failures)
            diagnostics['adaptive_fine_repair_failures'] = failures
            diagnostics['diagnostics_sha256'] = _diagnostic_sha256({key: value for key, value in diagnostics.items() if key != 'diagnostics_sha256'})
            raise FineSegmentationOutputError('fine segmentation had no eligible bounded fine retry components', diagnostics=diagnostics, partial_result=result) from exc
        component_edge_ticks = tuple(((component_index, max((item[1] for item in component.observation_spans_ticks)) if component.closed_start_ticks else min((item[0] for item in component.observation_spans_ticks))) for component_index in selected_component_indices for component in (result.unresolved[component_index],)))
        try:
            scopes = schedule_recentered_fine_retry_scopes(source_id=source.source_id, duration_s=source.duration_s, component_edge_ticks=component_edge_ticks, window_profile_id=window_profile.profile_id)
            windows = tuple((window_from_scope(scope) for scope in scopes))
        except Exception as geometry_error:
            raise _adaptive_fine_failure(exc, code='invalid_retry_geometry', detail={'error_type': type(geometry_error).__name__, 'error': str(geometry_error)}) from geometry_error
        try:
            prediction = await self.fine_segmenter.predict_repair_candidates_async(source, windows)
        except FineSegmentationOutputError as retry_error:
            raise _adaptive_fine_failure(exc, code='retry_output_rejected', detail={'error': str(retry_error), 'retry_diagnostics': retry_error.diagnostics}) from retry_error
        accepted_by_component: dict[int, _AcceptedFineRetry] = {}
        retry_diagnostics_by_component: dict[int, dict[str, object]] = {}
        incomplete_primary: list[tuple[int, int, dict[str, object]]] = []
        for (component_index, edge_tick), window in zip(component_edge_ticks, windows, strict=True):
            component = result.unresolved[component_index]
            candidates = prediction.candidates_by_window[window.window_index]
            matching_candidates, ignored_candidates = _match_fine_retry_candidates(component_index, component, candidates, local_stitch_candidates)
            failure_detail = {'component_index': component_index, 'retry_scope_id': window.scope_id, 'retry_edge_tick': edge_tick, 'candidate_count': len(candidates), 'matching_candidate_count': len(matching_candidates), 'ignored_candidates': ignored_candidates, 'retry_prediction_diagnostics': dict(prediction.diagnostics)}
            if len(matching_candidates) > 1:
                if not progressive_provider_free:
                    raise _adaptive_fine_failure(exc, code='ambiguous_component_coverage', detail=failure_detail)
                deferred_failures.append({'code': 'ambiguous_component_coverage', **failure_detail})
                continue
            if not matching_candidates:
                if not candidates:
                    if not progressive_provider_free:
                        raise _adaptive_fine_failure(exc, code='incomplete_component_coverage', detail=failure_detail)
                    deferred_failures.append({'code': 'incomplete_component_coverage', **failure_detail})
                    continue
                incomplete_primary.append((component_index, edge_tick, failure_detail))
                continue
            match = matching_candidates[0]
            accepted_by_component[component_index] = _AcceptedFineRetry(component_index=component_index, window=window, candidate=match.candidate, repaired_span=match.repaired_span, repair_mode=match.repair_mode, confirmation_target_event_id=str(match.local_stitch_candidate['resolved_event_id']) if match.local_stitch_candidate is not None else None, local_stitch_candidate=match.local_stitch_candidate)
            retry_diagnostics_by_component[component_index] = {'component_index': component_index, 'retry_pass': 'centered', 'repair_mode': match.repair_mode, 'retry_scope_id': window.scope_id, 'retry_edge_tick': edge_tick, 'requested_view_start_s': window.requested_view_start_us / 1000000, 'requested_view_end_s': window.requested_view_end_us / 1000000, 'candidate_count': len(candidates), 'candidate_evidence_span': _span_payload(match.candidate.span), 'repaired_span': _span_payload(match.repaired_span), 'ignored_candidates': list(ignored_candidates), **({'unresolved_local_stitch_confirmation': dict(match.local_stitch_candidate)} if match.local_stitch_candidate is not None else {})}
        secondary_windows: tuple[FineWindow, ...] = ()
        secondary_prediction: _FineRetryPrediction | None = None
        if incomplete_primary:
            secondary_edges = tuple(((component_index, edge_tick) for component_index, edge_tick, _ in incomplete_primary))
            secondary_offsets = {component_index: window_profile.left_guard_us // TICK_US if result.unresolved[component_index].closed_start_ticks else window_profile.core_ticks - window_profile.right_guard_us // TICK_US for component_index, _, _ in incomplete_primary}
            try:
                secondary_scopes = schedule_recentered_fine_retry_scopes(source_id=source.source_id, duration_s=source.duration_s, component_edge_ticks=secondary_edges, arm='adaptive_fine_directional_retry', edge_core_offset_ticks=secondary_offsets, scope_namespace=FINE_DIRECTIONAL_RETRY_SCOPE_NAMESPACE, window_profile_id=window_profile.profile_id)
                secondary_windows = tuple((window_from_scope(scope) for scope in secondary_scopes))
            except Exception as geometry_error:
                raise _adaptive_fine_failure(exc, code='invalid_directional_retry_geometry', detail={'error_type': type(geometry_error).__name__, 'error': str(geometry_error), 'component_indices': [item[0] for item in incomplete_primary]}) from geometry_error
            try:
                secondary_prediction = await self.fine_segmenter.predict_repair_candidates_async(source, secondary_windows, arm='adaptive_fine_directional_retry')
            except FineSegmentationOutputError as retry_error:
                raise _adaptive_fine_failure(exc, code='directional_retry_output_rejected', detail={'error': str(retry_error), 'retry_diagnostics': retry_error.diagnostics}) from retry_error
            for (component_index, edge_tick, primary_failure), window in zip(incomplete_primary, secondary_windows, strict=True):
                component = result.unresolved[component_index]
                candidates = secondary_prediction.candidates_by_window[window.window_index]
                matching_candidates, ignored_candidates = _match_fine_retry_candidates(component_index, component, candidates, local_stitch_candidates)
                if len(matching_candidates) != 1:
                    failure = {'code': 'incomplete_component_coverage' if not matching_candidates else 'ambiguous_component_coverage', 'component_index': component_index, 'retry_pass': 'directional_missing_endpoint_context', 'retry_scope_id': window.scope_id, 'retry_edge_tick': edge_tick, 'candidate_count': len(candidates), 'matching_candidate_count': len(matching_candidates), 'ignored_candidates': list(ignored_candidates), 'primary_retry_failure': primary_failure, 'retry_prediction_diagnostics': dict(secondary_prediction.diagnostics)}
                    if not progressive_provider_free:
                        raise _adaptive_fine_failure(exc, code=str(failure.pop('code')), detail=failure)
                    deferred_failures.append(failure)
                    continue
                match = matching_candidates[0]
                accepted_by_component[component_index] = _AcceptedFineRetry(component_index=component_index, window=window, candidate=match.candidate, repaired_span=match.repaired_span, repair_mode=match.repair_mode, confirmation_target_event_id=str(match.local_stitch_candidate['resolved_event_id']) if match.local_stitch_candidate is not None else None, local_stitch_candidate=match.local_stitch_candidate)
                retry_diagnostics_by_component[component_index] = {'component_index': component_index, 'retry_pass': 'directional_missing_endpoint_context', 'repair_mode': match.repair_mode, 'retry_scope_id': window.scope_id, 'retry_edge_tick': edge_tick, 'retry_edge_core_offset_tick': secondary_offsets[component_index], 'requested_view_start_s': window.requested_view_start_us / 1000000, 'requested_view_end_s': window.requested_view_end_us / 1000000, 'candidate_count': len(candidates), 'candidate_evidence_span': _span_payload(match.candidate.span), 'repaired_span': _span_payload(match.repaired_span), 'ignored_candidates': list(ignored_candidates), 'primary_retry_failure': primary_failure, **({'unresolved_local_stitch_confirmation': dict(match.local_stitch_candidate)} if match.local_stitch_candidate is not None else {})}
        accepted = [accepted_by_component[index] for index in sorted(accepted_by_component)]
        retry_diagnostics = [retry_diagnostics_by_component[index] for index in sorted(retry_diagnostics_by_component)]
        combined_events = [replace(event, metadata={'segmentation_provenance': 'fine', **dict(event.metadata)}) for event in result.events]
        retry_event_count_added = 0
        suppressed_duplicate_count = 0
        local_stitch_confirmation_count = 0
        for repair in accepted:
            if repair.repair_mode == 'unresolved_local_stitch_confirmed':
                geometry_matches = [(0.0, index) for index, event in enumerate(combined_events) if event.event_id == repair.confirmation_target_event_id]
                if len(geometry_matches) != 1 or repair.local_stitch_candidate is None:
                    raise _adaptive_fine_failure(exc, code='invalid_local_stitch_confirmation_target', detail={'component_index': repair.component_index, 'target_event_id': repair.confirmation_target_event_id, 'target_match_count': len(geometry_matches)})
            else:
                geometry_matches = [(abs(event.span.start_s - repair.repaired_span.start_s) + abs(event.span.end_s - repair.repaired_span.end_s), index) for index, event in enumerate(combined_events) if _same_event_geometry(event.span, repair.repaired_span)]
            if geometry_matches:
                _, event_index = min(geometry_matches)
                matched = combined_events[event_index]
                metadata = dict(matched.metadata)
                lineage = metadata.get('adaptive_fine_repair')
                prior_lineage = dict(lineage) if isinstance(lineage, dict) else {}
                component_indices = set(prior_lineage.get('component_indices', []))
                component_indices.add(repair.component_index)
                retry_scope_ids = set(prior_lineage.get('retry_scope_ids', []))
                retry_scope_ids.add(repair.window.scope_id)
                metadata.update({'segmentation_provenance': 'fine_repaired_by_fine_retry', 'adaptive_fine_repair': {'component_indices': sorted(component_indices), 'retry_scope_ids': sorted(retry_scope_ids), 'candidate_evidence_span': _span_payload(repair.candidate.span), 'duplicate_suppressed': True, 'repair_mode': repair.repair_mode}, **({'unresolved_local_stitch_confirmation': {'schema_version': UNRESOLVED_LOCAL_STITCH_SCHEMA, 'status': 'confirmed_by_adaptive_fine_retry', 'strict_confirmation_tolerance_ticks': STITCH_TOLERANCE_TICKS, 'retry_candidate_span': _span_payload(repair.candidate.span), 'candidate': dict(repair.local_stitch_candidate)}} if repair.local_stitch_candidate is not None else {})})
                combined_events[event_index] = replace(matched, source_scope_ids=tuple(sorted({*matched.source_scope_ids, repair.window.scope_id})), metadata=metadata)
                suppressed_duplicate_count += 1
                local_stitch_confirmation_count += int(repair.repair_mode == 'unresolved_local_stitch_confirmed')
                continue
            candidate_metadata = {key: value for key, value in repair.candidate.metadata.items() if key not in {'scope_clipped_start', 'scope_clipped_end'}}
            combined_events.append(EventInterval(event_id=f'{source.source_id}:adaptive-fine-repair:{retry_event_count_added:06d}', source_id=source.source_id, span=repair.repaired_span, source_scope_ids=(*sorted({*result.unresolved[repair.component_index].source_scope_ids, repair.window.scope_id}),), metadata={**candidate_metadata, 'segmentation_provenance': 'fine_repaired_by_fine_retry', 'adaptive_fine_repair': {'component_indices': [repair.component_index], 'retry_scope_ids': [repair.window.scope_id], 'candidate_evidence_span': _span_payload(repair.candidate.span), 'duplicate_suppressed': False, 'repair_mode': repair.repair_mode}}))
            retry_event_count_added += 1
        ordered_events = tuple(sorted(combined_events, key=lambda item: (item.span, item.event_id)))
        overlap_pairs, overlap_depth = _overlap_statistics(ordered_events)
        exact_duplicates = _exact_duplicate_intervals(ordered_events)
        post_repair_violations: list[dict[str, object]] = []
        if overlap_depth > self.fine_segmenter.quality_policy.hard_max_overlap_depth:
            post_repair_violations.append({'code': 'overlap_depth', 'observed': overlap_depth, 'maximum': self.fine_segmenter.quality_policy.hard_max_overlap_depth, 'failure_class': 'delivery_budget_exceeded'})
        if exact_duplicates:
            post_repair_violations.append({'code': 'exact_duplicate_intervals', 'observed': exact_duplicates, 'failure_class': 'structural_invalid'})
        if post_repair_violations:
            raise _adaptive_fine_failure(exc, code='invalid_combined_geometry', detail={'combined_event_count': len(ordered_events), 'violations': post_repair_violations})
        repair_payload: dict[str, object] = {'schema_version': ADAPTIVE_FINE_REPAIR_SCHEMA, 'scheduler_contract': 'adaptive_recentered_then_directional_' + window_profile.retry_scheduler_contract(directional=True).removeprefix('adaptive_directional_') if secondary_windows else window_profile.retry_scheduler_contract(directional=False), 'model_io_contract': window_profile.model_io_contract, 'window_profile_id': window_profile.profile_id, 'max_retry_windows': self.adaptive_fine_retry_max_windows, 'unresolved_component_count': len(result.unresolved), 'selected_component_count': len(selected_component_indices), 'selected_component_indices': selected_component_indices, 'selection_priority': 'unresolved_local_stitch_first_then_source_order_v1', 'budget_deferred_component_count': len(budget_deferred_indices), 'unsupported_deferred_component_count': len(unsupported_components), 'repaired_component_count': len(accepted), 'worker_batch_call_count': 1 + int(bool(secondary_windows)), 'retry_window_count': len(windows) + len(secondary_windows), 'centered_retry_window_count': len(windows), 'directional_retry_window_count': len(secondary_windows), 'fine_event_count_before': len(result.events), 'retry_event_count_added': retry_event_count_added, 'retry_duplicate_suppressed': suppressed_duplicate_count, 'unresolved_local_stitch_candidate_count': len(local_stitch_candidates), 'unresolved_local_stitch_confirmation_count': local_stitch_confirmation_count, 'combined_event_count': len(ordered_events), 'post_repair_overlap_pair_count': overlap_pairs, 'post_repair_max_overlap_depth': overlap_depth, 'post_repair_exact_duplicate_intervals': exact_duplicates, 'retry_prediction_diagnostics': dict(prediction.diagnostics), **({'directional_retry_prediction_diagnostics': dict(secondary_prediction.diagnostics)} if secondary_prediction is not None else {}), 'components': retry_diagnostics, 'component_failures': deferred_failures}
        remaining_component_indices = [index for index in range(len(result.unresolved)) if index not in accepted_by_component]
        repair_payload['remaining_component_count'] = len(remaining_component_indices)
        repair_payload['partial_success'] = bool(accepted) and bool(remaining_component_indices)
        if remaining_component_indices:
            raise _remaining_adaptive_fine_error(exc, result, repaired_events=ordered_events, accepted_component_indices=tuple(accepted_by_component), local_stitch_candidates=local_stitch_candidates, deferred_failures=deferred_failures, repair_payload=repair_payload, inference_context=inference_context) from exc
        diagnostics = dict(exc.diagnostics)
        diagnostics['pre_repair_violations'] = list(diagnostics.get('violations', []))
        diagnostics['repaired_unresolved_components'] = list(diagnostics.get('unresolved_components', []))
        diagnostics['violations'] = []
        diagnostics['unresolved_components'] = []
        diagnostics['acceptance_failure_classes'] = []
        diagnostics['resolved_event_count'] = len(ordered_events)
        diagnostics['overlap_pair_count'] = overlap_pairs
        diagnostics['max_overlap_depth'] = overlap_depth
        diagnostics['exact_duplicate_intervals'] = exact_duplicates
        diagnostics['adaptive_fine_repair'] = repair_payload
        diagnostics['adaptive_fine_repair_sha256'] = _diagnostic_sha256(repair_payload)
        diagnostics = _with_inference_context(diagnostics, inference_context)
        return events_to_temporal_segments(ordered_events, diagnostics=diagnostics)

    def _conservative_fine_completion(self, source: AnnotationSource, exc: FineSegmentationOutputError, *, inference_context: Mapping[str, object]) -> tuple[TemporalSegment, ...]:
        result = exc.partial_result
        violations = [dict(item) for item in exc.diagnostics.get('violations', []) if isinstance(item, dict)]
        violation_codes = {str(item.get('code')) for item in violations}
        if result is None or not result.unresolved or (not violation_codes) or (not violation_codes <= {'unresolved_components', 'stitch_ambiguity'}) or (source.duration_s is None):
            raise exc
        combined_events = [replace(event, metadata={'segmentation_provenance': 'fine', 'segmentation_resolution': 'fine', **dict(event.metadata)}) for event in result.events]
        completion_records: list[dict[str, object]] = []
        added_count = 0
        suppressed_duplicate_count = 0
        local_stitch_degraded_count = 0
        local_stitch_candidates = _unresolved_local_stitch_candidates(exc.diagnostics)
        for component_index, component in enumerate(result.unresolved):
            local_candidate = local_stitch_candidates.get(component_index)
            if local_candidate is not None:
                target_event_id = str(local_candidate['resolved_event_id'])
                target_matches = [index for index, event in enumerate(combined_events) if event.event_id == target_event_id]
                if len(target_matches) != 1:
                    raise SubtaskAnnotationAdapterError('unresolved-local degraded completion target is missing or ambiguous')
                target_index = target_matches[0]
                matched = combined_events[target_index]
                strict_target_span = _local_stitch_target_span(local_candidate)
                component_span, component_strategy = _conservative_component_completion_span(source, component)
                completed_span = _covering_span(matched.span, component_span)
                record = {'component_index': component_index, 'strategy': 'unconfirmed_unresolved_local_stitch_coverage_union_v2', 'observation_span': _span_payload(_component_observation_span(component)), 'component_completion_span': _span_payload(component_span), 'component_completion_strategy': component_strategy, 'strict_target_span': _span_payload(strict_target_span), 'pre_completion_target_span': _span_payload(matched.span), 'completed_span': _span_payload(completed_span), 'closed_start_ticks': list(component.closed_start_ticks), 'closed_end_ticks': list(component.closed_end_ticks), 'matching_ambiguous': component.matching_ambiguous, 'boundary_claim': 'observation_supported_union_of_strict_target_and_unresolved_component_v2', 'strict_target_geometry_expanded': completed_span != strict_target_span, 'candidate': dict(local_candidate), 'adaptive_fine_confirmation_failures': list(exc.diagnostics.get('adaptive_fine_repair_failures', [])), 'duplicate_suppressed': True}
                metadata = dict(matched.metadata)
                prior_evidence = metadata.get('unresolved_local_stitch_evidence', [])
                evidence = list(prior_evidence) if isinstance(prior_evidence, list) else []
                evidence.append(record)
                metadata.update({'segmentation_provenance': 'fine_with_unconfirmed_unresolved_local_stitch', 'segmentation_resolution': 'fine_degraded', 'review_required': True, 'semantic_quality_established': False, 'unresolved_local_stitch_evidence': evidence})
                combined_events[target_index] = replace(matched, span=completed_span, source_scope_ids=tuple(sorted({*matched.source_scope_ids, *component.source_scope_ids})), metadata=metadata)
                suppressed_duplicate_count += 1
                local_stitch_degraded_count += 1
                completion_records.append(record)
                continue
            span, strategy = _conservative_component_completion_span(source, component)
            record: dict[str, object] = {'component_index': component_index, 'strategy': strategy, 'observation_span': _span_payload(_component_observation_span(component)), 'completed_span': _span_payload(span), 'closed_start_ticks': list(component.closed_start_ticks), 'closed_end_ticks': list(component.closed_end_ticks), 'matching_ambiguous': component.matching_ambiguous, 'boundary_claim': 'observation_supported_conservative_extent_v1'}
            exact_match_index = next((index for index, event in enumerate(combined_events) if event.span == span), None)
            if exact_match_index is not None:
                matched = combined_events[exact_match_index]
                metadata = dict(matched.metadata)
                prior_evidence = metadata.get('conservative_fine_completion_evidence', [])
                evidence = list(prior_evidence) if isinstance(prior_evidence, list) else []
                evidence.append(record)
                metadata.update({'segmentation_provenance': 'fine_with_conservative_completion_evidence', 'segmentation_resolution': 'fine_degraded', 'review_required': True, 'semantic_quality_established': False, 'conservative_fine_completion_evidence': evidence})
                combined_events[exact_match_index] = replace(matched, source_scope_ids=tuple(sorted({*matched.source_scope_ids, *component.source_scope_ids})), metadata=metadata)
                record['duplicate_suppressed'] = True
                suppressed_duplicate_count += 1
            else:
                record['duplicate_suppressed'] = False
                combined_events.append(EventInterval(event_id=f'{source.source_id}:conservative-fine-completion:{component_index:06d}', source_id=source.source_id, span=span, source_scope_ids=tuple(sorted(component.source_scope_ids)), metadata={'segmentation_provenance': 'fine_conservative_completion', 'segmentation_resolution': 'fine_degraded', 'review_required': True, 'semantic_quality_established': False, 'boundary_claim': 'observation_supported_conservative_extent_v1', 'conservative_fine_completion': record}))
                added_count += 1
            completion_records.append(record)
        ordered_events = tuple(sorted(combined_events, key=lambda item: (item.span, item.event_id)))
        if not ordered_events:
            raise exc
        overlap_pairs, overlap_depth = _overlap_statistics(ordered_events)
        exact_duplicates = _exact_duplicate_intervals(ordered_events)
        if exact_duplicates:
            raise SubtaskAnnotationAdapterError('conservative fine completion failed to suppress exact duplicate intervals')
        completion_payload: dict[str, object] = {'schema_version': CONSERVATIVE_FINE_COMPLETION_SCHEMA, 'model_io_contract': fine_window_profile(self.fine_segmenter.window_profile_id).model_io_contract, 'window_profile_id': self.fine_segmenter.window_profile_id, 'terminal_contract': 'parsed_fine_evidence_only_no_coarse_provider_v1', 'boundary_claim': 'observation_supported_conservative_extent_v1', 'coarse_provider_called': False, 'additional_model_call_count': 0, 'max_retry_windows': self.adaptive_fine_retry_max_windows, 'fine_event_count_before': len(result.events), 'unresolved_component_count': len(result.unresolved), 'completion_event_count_added': added_count, 'completion_duplicate_suppressed': suppressed_duplicate_count, 'unresolved_local_stitch_candidate_count': len(local_stitch_candidates), 'unresolved_local_stitch_degraded_count': local_stitch_degraded_count, 'combined_event_count': len(ordered_events), 'post_completion_overlap_pair_count': overlap_pairs, 'post_completion_max_overlap_depth': overlap_depth, 'post_completion_exact_duplicate_intervals': exact_duplicates, 'components': completion_records}
        diagnostics = dict(exc.diagnostics)
        diagnostics['pre_completion_violations'] = violations
        diagnostics['pre_completion_unresolved_components'] = list(diagnostics.get('unresolved_components', []))
        diagnostics['violations'] = []
        diagnostics['unresolved_components'] = []
        diagnostics['acceptance_failure_classes'] = []
        diagnostics['resolved_event_count'] = len(ordered_events)
        diagnostics['overlap_pair_count'] = overlap_pairs
        diagnostics['max_overlap_depth'] = overlap_depth
        diagnostics['exact_duplicate_intervals'] = exact_duplicates
        diagnostics['output_degraded'] = True
        diagnostics['semantic_quality_established'] = False
        diagnostics['conservative_fine_completion'] = completion_payload
        diagnostics['conservative_fine_completion_sha256'] = _diagnostic_sha256(completion_payload)
        diagnostics = _with_inference_context(diagnostics, inference_context)
        return events_to_temporal_segments(ordered_events, diagnostics=diagnostics)

    @staticmethod
    def _bind_physical_source_envelope(source: AnnotationSource, segments: Sequence[TemporalSegment]) -> tuple[TemporalSegment, ...]:
        if source.duration_s is None:
            raise SubtaskAnnotationAdapterError('hierarchical fine segmentation requires exact source duration_s')
        duration_us = seconds_to_us(source.duration_s, label='duration_s')
        terminal_tick_us = ceil_div(duration_us, TICK_US) * TICK_US
        bounded: list[TemporalSegment] = []
        for segment in segments:
            if segment.span.end_s <= source.duration_s + 1e-06:
                bounded.append(segment)
                continue
            model_end_us = round(segment.span.end_s * 1000000)
            if not math.isclose(segment.span.end_s, model_end_us / 1000000, abs_tol=1e-09) or model_end_us != terminal_tick_us or segment.span.start_s >= source.duration_s:
                raise SubtaskAnnotationAdapterError(f'hierarchical segment for {source.source_id} ends at {segment.span.end_s}s beyond the physical source duration {source.duration_s}s and is not the terminal model tick')
            metadata = dict(segment.metadata)
            metadata['terminal_source_clip'] = {'contract': 'terminal_20hz_tick_to_physical_source_duration_v1', 'model_end_us': model_end_us, 'physical_end_us': duration_us}
            bounded.append(TemporalSegment(TimeSpan(segment.span.start_s, source.duration_s), metadata))
        ordered = tuple(sorted(bounded, key=lambda item: (item.span, repr(item.metadata))))
        output: list[TemporalSegment] = []
        for index, segment in enumerate(ordered):
            before, after = _nearest_nonoverlapping_context(ordered, index)
            metadata = dict(segment.metadata)
            metadata['u2_context'] = {'before': _span_payload(before), 'target': _span_payload(segment.span), 'after': _span_payload(after)}
            output.append(TemporalSegment(segment.span, metadata))
        return tuple(output)

    @staticmethod
    def _with_refreshed_u2_context(segments: Sequence[TemporalSegment]) -> tuple[TemporalSegment, ...]:
        ordered = tuple(sorted(segments, key=lambda item: (item.span, repr(item.metadata))))
        output: list[TemporalSegment] = []
        for index, segment in enumerate(ordered):
            before, after = _nearest_nonoverlapping_context(ordered, index)
            metadata = dict(segment.metadata)
            metadata['u2_context'] = {'before': _span_payload(before), 'target': _span_payload(segment.span), 'after': _span_payload(after)}
            output.append(TemporalSegment(segment.span, metadata))
        return tuple(output)

    def segment(self, source: AnnotationSource) -> tuple[TemporalSegment, ...]:
        return asyncio.run(self.segment_async(source))

    async def segment_async(self, source):
        return self._bind_physical_source_envelope(source, await self._segment_impl(source))

    async def _segment_impl(self, source):
        inference_context = _inference_context(source, (), fine_prior_mode='none')
        segment_with_priors_async = getattr(self.fine_segmenter, 'segment_with_priors_async', None)
        try:
            if callable(segment_with_priors_async):
                result = await segment_with_priors_async(source, (), arm='no_prior_global_scan')
            else:
                result = await asyncio.to_thread(self.fine_segmenter.segment_with_priors, source, (), arm='no_prior_global_scan')
        except FineSegmentationOutputError as exc:
            contextual_error = FineSegmentationOutputError(str(exc), diagnostics=_with_inference_context(exc.diagnostics, inference_context), partial_result=exc.partial_result)
            try:
                return await self.adaptive_fine_repair_async(source, contextual_error, inference_context=inference_context)
            except FineSegmentationOutputError as adaptive_error:
                return self._conservative_fine_completion(source, adaptive_error, inference_context=inference_context)
        diagnostics = _with_inference_context(result.diagnostics, inference_context)
        return events_to_temporal_segments(result.events, diagnostics=diagnostics)
