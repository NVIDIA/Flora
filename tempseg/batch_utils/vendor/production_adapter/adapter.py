from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
from flora_runtime.schema import AnnotationSource, TemporalSegment, TimeSpan
from flora_runtime.video_segmentation.fine_subtask import FineSubtaskSegmenter, HierarchicalSubtaskSegmenter, FineSegmentationQualityPolicy
from flora_runtime.video_segmentation.fine_temporal import FineWindow, fine_window_profile, TICK_US, HUMAN_FINE_WINDOW_PROFILE_ID, YAM_ATOMIC_LONG_FINE_WINDOW_PROFILE_ID, YAM_INTENT_32S_FINE_WINDOW_PROFILE_ID
from flora_runtime.video_segmentation.fine_worker import _nearest_index
from flora_runtime.subtask_captioning.u2_worker import _plan_blocks, U2WorkerError
HERE = Path(__file__).resolve().parent
TEMPLATES = {x['pool']: x for x in json.loads((HERE / 'TEMPSEG_PROMPTS.json').read_text())}
PROFILES = {'human': HUMAN_FINE_WINDOW_PROFILE_ID, 'tianji': YAM_ATOMIC_LONG_FINE_WINDOW_PROFILE_ID, 'dex': YAM_ATOMIC_LONG_FINE_WINDOW_PROFILE_ID, 'yam': YAM_ATOMIC_LONG_FINE_WINDOW_PROFILE_ID, 'yam32': YAM_INTENT_32S_FINE_WINDOW_PROFILE_ID}

def temporal_prompt(window, pool):
    assert window.window_profile_id == PROFILES[pool]
    row = TEMPLATES[pool]
    template = row['prompt']
    assert hashlib.sha256(template.encode()).hexdigest() == row['prompt_sha256']
    pattern = '(Canonical window envelope:\\s*)(\\{[^{}]+\\})'
    matches = list(re.finditer(pattern, template))
    assert len(matches) == 1
    match = matches[0]
    keys = list(json.loads(match[2]))
    assert set(keys) == {'valid_start_tick', 'valid_end_tick', 'core_start_tick', 'core_end_tick'}
    envelope = json.dumps({k: getattr(window, k) for k in keys}, separators=(',', ':'))
    return template[:match.start(2)] + envelope + template[match.end(2):]

def window_from_payload(payload, source):
    start = payload['requested_view_start_us']
    fields = {k: payload[k] for k in ('window_index', 'requested_view_start_us', 'requested_view_end_us', 'valid_start_tick', 'valid_end_tick', 'core_start_tick', 'core_end_tick', 'scope_id', 'window_profile_id', 'coarse_spans')}
    duration = round(source.duration_s * 1000000)
    fields.update(source_id=source.source_id, video_end_us=duration, valid_start_us=max(0, start), valid_end_us=min(duration, payload['requested_view_end_us']), core_start_us=max(0, start + payload['core_start_tick'] * TICK_US), core_end_us=min(duration, start + payload['core_end_tick'] * TICK_US))
    return FineWindow(**fields)

class NoCoarse:

    def segment(self, *_args, **_kwargs):
        raise AssertionError('Unexpected coarse segmentation request')

def production_segmenter(predictor, binding, pool):
    assert binding['model_io_contract'] == fine_window_profile(PROFILES[pool]).model_io_contract
    fine = FineSubtaskSegmenter(predictor=predictor, model_binding=binding, profile_id='human_ego_atomic_v0_2', window_profile_id=PROFILES[pool], fail_on_unresolved=True, batch_concurrency=1, unresolved_stitch_tolerance_ticks=2, quality_policy=FineSegmentationQualityPolicy.from_value({'drop_guard_only_segments': True, 'hard_max_overlap_depth': 3, 'hard_max_segments_per_window': 23, 'max_adjacent_matching_ambiguities': 0, 'require_event_when_coarse_present': True, 'schema_version': 'subtask_annotation_atomic_fine_quality_policy_v1', 'soft_max_segments_per_window': 12}))
    return HierarchicalSubtaskSegmenter(coarse_segmenter=NoCoarse(), fine_segmenter=fine, fine_prior_mode='none', fine_failure_policy='adaptive_fine_then_conservative_completion', fine_output_coverage_policy='allow_gaps', adaptive_fine_retry_max_windows=32)

def temporal_frame_plan(window, pts_us):
    assert pts_us and pts_us[0] == 0 and all((a < b for a, b in zip(pts_us, pts_us[1:])))
    assert pts_us[-1] < window.video_end_us
    profile = window.window_profile
    requested = [window.requested_view_start_us + i * profile.slot_spacing_us for i in range(profile.slot_count)]
    pad = [not 0 <= t < window.video_end_us for t in requested]
    indices = [None if padding else _nearest_index(t, pts_us, max_selection_error_us=profile.slot_spacing_us // 2) for t, padding in zip(requested, pad, strict=True)]
    real = [i for i, v in enumerate(indices) if v is not None]
    assert real
    first, last = (real[0], real[-1])
    indices[:first] = [indices[first]] * first
    indices[last + 1:] = [indices[last]] * (len(indices) - last - 1)
    assert all((v is not None for v in indices))
    observed = [v for v, p in zip(indices, pad, strict=True) if not p]
    assert len(set(observed)) == len(observed), 'Real temporal slots must not duplicate evidence'
    return {'source_frame_indices': indices, 'padding_mask': pad, 'requested_us': requested, 'actual_pts_us': [float(pts_us[i]) for i in indices], 'model_clock_hz': profile.slot_fps, 'tick_us': TICK_US, 'max_real_error_us': float(max((abs(pts_us[i] - t) for i, t, p in zip(indices, requested, pad, strict=True) if not p)))}

def caption_plans(segments, media):
    sources = [TemporalSegment(TimeSpan(s['start_s'], s['end_s']), dict(s.get('metadata', {}))) for s in segments]
    sources = HierarchicalSubtaskSegmenter._with_refreshed_u2_context(tuple(sources))
    plans, dispositions = ([], [])
    for i, s in enumerate(sources):
        eid = s.metadata.get('fine_event_id') or f'event_{i:06d}'
        record = {'event_id': eid, 'start_s': s.span.start_s, 'end_s': s.span.end_s, 'event_index': i}
        context = s.metadata['u2_context']
        try:
            blocks = _plan_blocks(context, media)
        except U2WorkerError as error:
            if 'TARGET span has fewer than' not in str(error):
                raise
            dispositions.append(dict(record, status='unsupported_target_frames', reason=str(error)))
            continue
        serialized = [asdict(b) for b in blocks]
        for b in serialized:
            b['frame_indices'] = list(b['frame_indices'])
        plans.append(dict(record, blocks=serialized, source_fps_num=media['fps_num'], source_fps_den=media['fps_den'], metadata=dict(s.metadata)))
        dispositions.append(dict(record, status='caption_input'))
    assert len({x['event_id'] for x in dispositions}) == len(dispositions)
    return (plans, dispositions)

def caption_sample(plan, frame_receipts, camera_roles):
    blocks = []
    clock = Fraction(plan['source_fps_num'], plan['source_fps_den'])
    for b in plan['blocks']:
        ids = b['frame_indices']
        n = len(ids)
        span = b['end_frame_exclusive'] - b['start_frame']
        rate = Fraction(n) * clock / span
        receipts = [frame_receipts[i] for i in ids]
        blocks.append(dict(role=b['role'], paths=[r['path'] for r in receipts], frames=receipts, source_start_frame=b['start_frame'], source_end_frame_exclusive=b['end_frame_exclusive'], clock_fps_num=clock.numerator, clock_fps_den=clock.denominator, block_local_timestamp_indices=[i - b['start_frame'] for i in ids], effective_fps_num=rate.numerator, effective_fps_den=rate.denominator, original_descriptor={'frame_indices': ids}))
    target = next((b for b in blocks if b['role'] == 'TARGET'))
    text = 'REFERENCE_NOT_ADMITTED'
    request = {'target': {'native_span': {'start_frame': target['source_start_frame'], 'end_frame_exclusive': target['source_end_frame_exclusive']}, 'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest()}}
    return {'request': request, 'camera_count': len(camera_roles), 'camera_roles': camera_roles, 'blocks': blocks}
