# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from itertools import pairwise
import math
from flora_runtime.schema import TimeSpan
from flora_runtime.video_segmentation.contracts import ProcessingScope
from flora_runtime.video_segmentation.fine_temporal import DEFAULT_FINE_WINDOW_PROFILE_ID, TICK_US, WINDOW_SCHEMA, ceil_div, fine_window_profile, floor_div, round_half_up_div

class ScopeSchedulerError(ValueError):
    pass
FINE_WINDOW_SCOPE_NAMESPACE = 'fine-window'
FINE_RETRY_SCOPE_NAMESPACE = 'fine-retry'
FINE_DIRECTIONAL_RETRY_SCOPE_NAMESPACE = 'fine-retry-directional'

def seconds_to_us(value: float, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ScopeSchedulerError(f'{label} must be numeric')
    try:
        parsed = Decimal(str(value)) * 1000000
    except (InvalidOperation, ValueError) as exc:
        raise ScopeSchedulerError(f'{label} is not valid decimal seconds') from exc
    if not parsed.is_finite() or parsed != parsed.to_integral_value():
        raise ScopeSchedulerError(f'{label} must resolve to integral microseconds')
    result = int(parsed)
    if result < 0:
        raise ScopeSchedulerError(f'{label} must be non-negative')
    return result

def _quantized_prior(proposal: ProcessingScope, *, requested_start_us: int, valid_start_us: int, valid_end_us: int, valid_start_tick: int, valid_end_tick: int) -> dict[str, object] | None:
    start_us = seconds_to_us(proposal.core_span.start_s, label=f'{proposal.scope_id}.start_s')
    end_us = seconds_to_us(proposal.core_span.end_s, label=f'{proposal.scope_id}.end_s')
    if start_us >= valid_end_us or end_us <= valid_start_us:
        return None
    visible_start = max(start_us, valid_start_us)
    visible_end = min(end_us, valid_end_us)
    lo, ro = (start_us < valid_start_us, end_us > valid_end_us)
    start_delta, end_delta = (visible_start - requested_start_us, visible_end - requested_start_us)
    s = valid_start_tick if lo else round_half_up_div(start_delta, TICK_US)
    e = valid_end_tick if ro else round_half_up_div(end_delta, TICK_US)
    if s >= e:
        s, e = (floor_div(start_delta, TICK_US), ceil_div(end_delta, TICK_US))
    if not valid_start_tick <= s < e <= valid_end_tick:
        raise ScopeSchedulerError(f'proposal {proposal.scope_id} cannot be represented at 20 Hz')
    return {'s': s, 'e': e, 'lo': lo, 'ro': ro}

def schedule_fine_scopes(*, source_id: str, duration_s: float, coarse_proposals: Sequence[ProcessingScope]=(), arm: str='no_prior_global_scan', window_profile_id: str=DEFAULT_FINE_WINDOW_PROFILE_ID) -> tuple[ProcessingScope, ...]:
    if not isinstance(source_id, str) or not source_id.strip():
        raise ScopeSchedulerError('source_id must be non-empty')
    if not isinstance(arm, str) or not arm.strip():
        raise ScopeSchedulerError('arm must be non-empty')
    try:
        profile = fine_window_profile(window_profile_id)
    except ValueError as exc:
        raise ScopeSchedulerError(str(exc)) from exc
    video_end_us = seconds_to_us(duration_s, label='duration_s')
    if video_end_us <= 0:
        raise ScopeSchedulerError('duration_s must be positive')
    proposals = tuple(coarse_proposals)
    proposal_ids = [proposal.scope_id for proposal in proposals]
    if len(proposal_ids) != len(set(proposal_ids)):
        raise ScopeSchedulerError('coarse proposal scope_ids must be unique')
    if any((proposal.source_id != source_id for proposal in proposals)):
        raise ScopeSchedulerError('every coarse proposal must belong to source_id')
    for proposal in proposals:
        proposal_end_us = seconds_to_us(proposal.core_span.end_s, label=f'{proposal.scope_id}.end_s')
        if proposal_end_us > video_end_us:
            raise ScopeSchedulerError(f'coarse proposal {proposal.scope_id} extends past source')
    proposals = tuple(sorted(proposals, key=lambda item: (item.core_span, item.scope_id)))
    scopes: list[ProcessingScope] = []
    window_index = 0
    while True:
        core_start_us = window_index * profile.core_us
        if core_start_us >= video_end_us:
            break
        core_end_us = min(core_start_us + profile.core_us, video_end_us)
        requested_start_us = core_start_us - profile.left_guard_us
        requested_end_us = requested_start_us + profile.view_us
        valid_start_us = max(0, requested_start_us)
        valid_end_us = min(video_end_us, requested_end_us)
        valid_start_tick = floor_div(valid_start_us - requested_start_us, TICK_US)
        valid_end_tick = ceil_div(valid_end_us - requested_start_us, TICK_US)
        core_start_tick = floor_div(core_start_us - requested_start_us, TICK_US)
        core_end_tick = ceil_div(core_end_us - requested_start_us, TICK_US)
        prior_groups: dict[tuple[int, int, bool, bool], list[str]] = {}
        for proposal in proposals:
            row = _quantized_prior(proposal, requested_start_us=requested_start_us, valid_start_us=valid_start_us, valid_end_us=valid_end_us, valid_start_tick=valid_start_tick, valid_end_tick=valid_end_tick)
            if row is not None:
                key = (int(row['s']), int(row['e']), bool(row['lo']), bool(row['ro']))
                prior_groups.setdefault(key, []).append(proposal.scope_id)
        ordered_priors = sorted(prior_groups.items())
        coarse_rows = [{'s': key[0], 'e': key[1], 'lo': key[2], 'ro': key[3]} for key, _ in ordered_priors]
        visible_ids = [scope_id for _, scope_ids in ordered_priors for scope_id in scope_ids]
        coarse_span_lineage = [{'row_index': index, 'scope_ids': scope_ids} for index, (_, scope_ids) in enumerate(ordered_priors)]
        if len(coarse_rows) > 64:
            raise ScopeSchedulerError('one fine view contains more than 64 coarse priors')
        scope_id = f'{source_id}:{FINE_WINDOW_SCOPE_NAMESPACE}:{window_index:06d}'
        metadata = {'schema_version': WINDOW_SCHEMA, 'model_io_contract': profile.model_io_contract, 'window_profile_id': profile.profile_id, 'window_profile': profile.to_dict(), 'scheduler_contract': profile.scheduler_contract, 'arm': arm, 'guidance': 'coarse_prior' if coarse_rows else 'no_prior_global_scan', 'coarse_spans': coarse_rows, 'coarse_scope_ids': visible_ids, 'coarse_span_lineage': coarse_span_lineage, 'window_index': window_index, 'video_end_us': video_end_us, 'requested_view_start_us': requested_start_us, 'requested_view_end_us': requested_end_us, 'valid_start_us': valid_start_us, 'valid_end_us': valid_end_us, 'core_start_us': core_start_us, 'core_end_us': core_end_us, 'valid_start_tick': valid_start_tick, 'valid_end_tick': valid_end_tick, 'core_start_tick': core_start_tick, 'core_end_tick': core_end_tick}
        scopes.append(ProcessingScope(scope_id=scope_id, source_id=source_id, core_span=TimeSpan(core_start_us / 1000000, core_end_us / 1000000), context_span=TimeSpan(valid_start_us / 1000000, valid_end_us / 1000000), metadata=metadata))
        window_index += 1
    if not scopes:
        raise ScopeSchedulerError('scope scheduling produced no windows')
    if scopes[0].core_span.start_s != 0 or not math.isclose(scopes[-1].core_span.end_s, duration_s):
        raise ScopeSchedulerError('fine emit cores do not cover the complete source')
    for left, right in pairwise(scopes):
        if left.core_span.end_s != right.core_span.start_s:
            raise ScopeSchedulerError('fine emit cores are not an exact partition')
    return tuple(scopes)

def schedule_recentered_fine_retry_scopes(*, source_id: str, duration_s: float, component_edge_ticks: Sequence[tuple[int, int]], arm: str='adaptive_fine_repair', edge_core_offset_ticks: Mapping[int, int] | None=None, scope_namespace: str=FINE_RETRY_SCOPE_NAMESPACE, window_profile_id: str=DEFAULT_FINE_WINDOW_PROFILE_ID) -> tuple[ProcessingScope, ...]:
    if not isinstance(source_id, str) or not source_id.strip():
        raise ScopeSchedulerError('source_id must be non-empty')
    if not isinstance(arm, str) or not arm.strip():
        raise ScopeSchedulerError('arm must be non-empty')
    try:
        profile = fine_window_profile(window_profile_id)
    except ValueError as exc:
        raise ScopeSchedulerError(str(exc)) from exc
    if not isinstance(scope_namespace, str) or not scope_namespace or any((not (character.isalnum() or character in '_-') for character in scope_namespace)):
        raise ScopeSchedulerError('adaptive fine retry scope_namespace is invalid')
    video_end_us = seconds_to_us(duration_s, label='duration_s')
    if video_end_us <= 0:
        raise ScopeSchedulerError('duration_s must be positive')
    targets = tuple(component_edge_ticks)
    if not targets:
        raise ScopeSchedulerError('adaptive fine retry requires at least one component edge')
    component_indices = [item[0] for item in targets]
    if len(component_indices) != len(set(component_indices)):
        raise ScopeSchedulerError('adaptive fine retry component indices must be unique')
    offsets = dict(edge_core_offset_ticks or {})
    if not set(offsets) <= set(component_indices):
        raise ScopeSchedulerError('adaptive fine retry offsets contain an unknown component')
    scopes: list[ProcessingScope] = []
    for window_index, (component_index, edge_tick) in enumerate(targets):
        if isinstance(component_index, bool) or not isinstance(component_index, int) or component_index < 0:
            raise ScopeSchedulerError('adaptive fine retry component index must be non-negative')
        if isinstance(edge_tick, bool) or not isinstance(edge_tick, int):
            raise ScopeSchedulerError('adaptive fine retry edge tick must be an integer')
        edge_core_offset_tick = offsets.get(component_index, floor_div(profile.core_us // 2, TICK_US))
        if isinstance(edge_core_offset_tick, bool) or not isinstance(edge_core_offset_tick, int) or (not 0 <= edge_core_offset_tick <= profile.core_ticks):
            raise ScopeSchedulerError('adaptive fine retry edge core offset must lie inside the emit core')
        edge_us = edge_tick * TICK_US
        if not 0 < edge_us < video_end_us:
            raise ScopeSchedulerError('adaptive fine retry edge must be inside the source')
        latest_full_core_start_us = floor_div(max(0, video_end_us - profile.core_us), TICK_US) * TICK_US
        core_start_us = min(max(0, edge_us - edge_core_offset_tick * TICK_US), latest_full_core_start_us)
        core_end_us = min(core_start_us + profile.core_us, video_end_us)
        requested_start_us = core_start_us - profile.left_guard_us
        requested_end_us = requested_start_us + profile.view_us
        valid_start_us = max(0, requested_start_us)
        valid_end_us = min(video_end_us, requested_end_us)
        valid_start_tick = floor_div(valid_start_us - requested_start_us, TICK_US)
        valid_end_tick = ceil_div(valid_end_us - requested_start_us, TICK_US)
        core_start_tick = floor_div(core_start_us - requested_start_us, TICK_US)
        core_end_tick = ceil_div(core_end_us - requested_start_us, TICK_US)
        if not core_start_us <= edge_us <= core_end_us:
            raise ScopeSchedulerError('adaptive fine retry edge does not lie in the emit core')
        scope_id = f'{source_id}:{scope_namespace}:{window_index:06d}'
        metadata = {'schema_version': WINDOW_SCHEMA, 'model_io_contract': profile.model_io_contract, 'window_profile_id': profile.profile_id, 'window_profile': profile.to_dict(), 'scheduler_contract': profile.retry_scheduler_contract(directional=component_index in offsets), 'arm': arm, 'guidance': 'no_prior_adaptive_retry', 'coarse_spans': [], 'coarse_scope_ids': [], 'coarse_span_lineage': [], 'window_index': window_index, 'retry_component_index': component_index, 'retry_edge_tick': edge_tick, 'retry_edge_core_offset_tick': edge_core_offset_tick, 'video_end_us': video_end_us, 'requested_view_start_us': requested_start_us, 'requested_view_end_us': requested_end_us, 'valid_start_us': valid_start_us, 'valid_end_us': valid_end_us, 'core_start_us': core_start_us, 'core_end_us': core_end_us, 'valid_start_tick': valid_start_tick, 'valid_end_tick': valid_end_tick, 'core_start_tick': core_start_tick, 'core_end_tick': core_end_tick}
        scopes.append(ProcessingScope(scope_id=scope_id, source_id=source_id, core_span=TimeSpan(core_start_us / 1000000, core_end_us / 1000000), context_span=TimeSpan(valid_start_us / 1000000, valid_end_us / 1000000), metadata=metadata))
    return tuple(scopes)
