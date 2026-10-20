import asyncio
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
import time
from .bootstrap import production as a
from .config import MODELS, PROFILE_POOLS
from .media import frames_for, check_synchronized, InputError
from .models import Session, generate, runtime_event
from .preprocessing import temporal_inputs, human_transport, instruction_suffix, three_view_caption_inputs

class Predictor:

    def __init__(self, videos, source, pool, bound, model, proc, instruction, progress):
        self.videos, self.source, self.pool, self.bound = (videos, source, pool, bound)
        self.model, self.proc, self.instruction, self.progress = (model, proc, instruction, progress)
        self.calls = []

    def predict(self, request):
        from flora_runtime.video_segmentation.fine_subtask import FINE_RESPONSE_SCHEMA
        if request['source']['source_id'] != self.source.source_id or request['model_binding'] != self.bound:
            raise RuntimeError('Production request binding changed')
        predictions = []
        for payload in request['windows']:
            window = a.window_from_payload(payload, self.source)
            plan = a.temporal_frame_plan(window, list(self.videos['main'].pts_us))
            mosaic = 'Dex' if self.pool in ('dex', 'tianji') else 'YAM'
            images = frames_for(self.videos, set(plan['source_frame_indices']), mosaic=mosaic)
            if self.pool == 'human':
                images = human_transport(images)
            frames = [images[i] for i in plan['source_frame_indices']]
            inputs, prompt = temporal_inputs(frames, window, self.pool, self.proc, list(self.videos), self.instruction)
            self.progress(0.25, desc=f'Finding actions · window {len(self.calls) + 1}')
            text = generate(self.model, self.proc, inputs, 1024)
            predictions.append(dict(scope_id=window.scope_id, prediction_json=text))
            self.calls.append(dict(window=window.worker_payload(), frame_plan=plan, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), output=text))
        return dict(schema_version=FINE_RESPONSE_SCHEMA, source_id=self.source.source_id, model_io_contract=self.bound['model_io_contract'], model_binding=self.bound, predictions=predictions)

def segment(videos, pool, session, instruction, progress):
    model, proc = session.load('tempseg')
    main = videos['main']
    source = a.AnnotationSource(source_id='upload_' + main.sha256[:24], location=str(main.path), duration_s=main.duration_s, fps=float(main.rate))
    bound = dict(candidate_id='flora_final', checkpoint_sha256=MODELS['tempseg']['sha256'], model_io_contract=a.fine_window_profile(a.PROFILES[pool]).model_io_contract)
    predictor = Predictor(videos, source, pool, bound, model, proc, instruction, progress)
    if len(videos) == 3:
        fine = a.FineSubtaskSegmenter(predictor=predictor, model_binding=bound, profile_id='robot_atomic_intent_v1', window_profile_id=a.PROFILES[pool], fail_on_unresolved=True, batch_concurrency=1, unresolved_stitch_tolerance_ticks=2, quality_policy=a.FineSegmentationQualityPolicy(drop_guard_only_segments=True))
        engine = a.HierarchicalSubtaskSegmenter(coarse_segmenter=a.NoCoarse(), fine_segmenter=fine, fine_prior_mode='none', fine_failure_policy='adaptive_fine_then_conservative_completion', fine_output_coverage_policy='allow_gaps', adaptive_fine_retry_max_windows=32)
    else:
        engine = a.production_segmenter(predictor, bound, pool)
    segments = asyncio.run(engine.segment_async(source))
    return ([dict(start_s=s.span.start_s, end_s=s.span.end_s, metadata=dict(s.metadata)) for s in segments], predictor.calls)

def single_plans(videos, segments, profile):
    if profile in ('Human', 'Human actions'):
        from pts_input_adapter import native_blocks, microseconds
        from flora_runtime.subtask_captioning import u2_contract
        main = videos['main']
        pts = [float(t / 1000000) for t in main.pts_us]
        plans, dispositions = ([], [])
        for i, s in enumerate(segments):
            record = dict(event_id=f'event_{i:04d}', event_index=i, start_s=s['start_s'], end_s=s['end_s'])
            try:
                blocks = native_blocks(pts, microseconds(s['start_s']), microseconds(s['end_s']), main.duration_us, main.rate, u2_contract)
            except ValueError:
                dispositions.append(dict(record, status='unsupported_target_frames'))
                continue
            plans.append(dict(record, blocks=blocks, source_fps_num=main.rate.numerator, source_fps_den=main.rate.denominator))
            dispositions.append(dict(record, status='caption_input'))
        return (plans, dispositions)
    main = videos['main']
    if not main.cfr:
        raise InputError('Use the Human actions profile for videos with variable frame timestamps.')
    return a.caption_plans(segments, dict(fps_num=main.rate.numerator, fps_den=main.rate.denominator, frame_count=len(main.pts_us)))

def caption_single(videos, segments, profile, session, instruction, progress):
    from flora_unified.unified_input import processor_inputs, prompt as native_prompt
    from transformers.video_utils import VideoMetadata
    plans, dispositions = single_plans(videos, segments, profile)
    if not plans:
        return ([dict(d, caption='', status=d['status']) for d in dispositions], [])
    model, proc = session.load('captioner')
    rows, audit = ({}, [])
    with tempfile.TemporaryDirectory(prefix='flora-frames-') as temporary:
        root = Path(temporary)
        wanted = {i for plan in plans for block in plan['blocks'] for i in block['frame_indices']}
        images = frames_for(videos, wanted)
        receipts = {}
        for i, image in images.items():
            path = root / f'{i:012d}.png'
            image.save(path, format='PNG')
            receipts[i] = dict(path=path.name, png_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size=list(image.size))
        for n, plan in enumerate(plans):
            progress(0.55 + 0.4 * n / len(plans), desc=f'Writing captions · {n + 1}/{len(plans)}')
            sample = a.caption_sample(plan, receipts, ['main'])
            if instruction.strip():
                from flora_unified.unified_input import load_frames
                frames, _ = load_frames(sample, root)
                text = proc.apply_chat_template([dict(role='user', content=native_prompt(sample, proc.video_token) + instruction_suffix(instruction))], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                metadata = [VideoMetadata(total_num_frames=len(f), fps=float(videos['main'].rate), frames_indices=list(b['block_local_timestamp_indices'])) for f, b in zip(frames, sample['blocks'], strict=True)]
                inputs = proc(text=[text], videos=frames, video_metadata=metadata, do_sample_frames=False, return_metadata=True, return_tensors='pt')
            else:
                inputs, _, text = processor_inputs(sample, root, proc)
            caption = generate(model, proc, inputs, 512)
            rows[plan['event_id']] = dict(start_s=plan['start_s'], end_s=plan['end_s'], caption=caption, status='complete')
            audit.append(dict(event_id=plan['event_id'], blocks=plan['blocks'], prompt_sha256=hashlib.sha256(text.encode()).hexdigest()))
    return ([rows.get(d['event_id'], dict(start_s=d['start_s'], end_s=d['end_s'], caption='', status=d['status'])) for d in dispositions], audit)

def caption_three(videos, segments, session, instruction, progress, profile='YAM'):
    from plans import plans_for_episode
    from robot_caption_geometry import normalized, source_frames
    main = videos['main']
    from math import lcm
    scale = lcm(*(t.denominator for t in main.pts_us)) * 1000000
    ticks = [int(t * scale / 1000000) for t in main.pts_us]
    clock = dict(ticks=ticks, time_base=str(Fraction(1, scale)), fps=str(main.rate), duration_s=main.duration_s, first_pts=0)
    group = 'Dex' if profile in ('Dex', 'Tianji') else 'YAM'
    episode = dict(episode_id='upload', source_id=main.sha256, group=group, units=[])
    prepared, rows, audit = ([], [], [])
    for i, s in enumerate(segments):
        unit = dict(target_id=f'event_{i:04d}', unit_id=f'prediction:{i}', native_scope_s=[s['start_s'], s['end_s']])
        try:
            plan = plans_for_episode(dict(episode, units=[unit]), {role: clock for role in videos})[0]
            prepared.append((i, plan))
        except AssertionError:
            rows.append((i, dict(start_s=s['start_s'], end_s=s['end_s'], caption='', status='unsupported_target_frames')))
    if prepared:
        model, proc = session.load('captioner')
        for n, (i, plan) in enumerate(prepared):
            progress(0.55 + 0.4 * n / len(prepared), desc=f'Writing captions · {n + 1}/{len(prepared)}')
            wanted = {j for block in plan['blocks'] for j in block['indices']}
            images = frames_for(videos, wanted, mosaic=group)
            geometry = normalized(plan, images)
            inputs, text = three_view_caption_inputs(plan, source_frames(geometry), proc, instruction)
            caption = generate(model, proc, inputs, 512)
            rows.append((i, dict(start_s=plan['native_scope_s'][0], end_s=plan['native_scope_s'][1], caption=caption, status='complete')))
            audit.append(dict(event_id=plan['target_id'], blocks=plan['blocks'], common_geometry=geometry['common_geometry'], camera_layout=group, prompt_sha256=hashlib.sha256(text.encode()).hexdigest()))
    return ([row for _, row in sorted(rows)], audit)

def annotate(videos, profile, long_windows, instruction, progress):
    check_synchronized(videos)
    instruction_suffix(instruction)
    if profile not in PROFILE_POOLS:
        raise InputError('Choose a supported action profile.')
    if len(videos) == 3 and profile in ('Human', 'Human actions'):
        raise InputError('Choose YAM, Dex or Tianji when uploading wrist cameras.')
    if long_windows and len(videos) != 3:
        raise InputError('The 32-second profile requires three synchronized camera views.')
    if long_windows and profile not in ('YAM', 'Robot actions'):
        raise InputError('The 32-second profile is available for YAM videos.')
    pool = 'yam32' if long_windows else PROFILE_POOLS[profile]
    start = time.monotonic()
    session = Session()
    try:
        progress(0.1, desc='Loading the action segmenter')
        segments, temporal_audit = segment(videos, pool, session, instruction, progress)
        runtime_event('segmentation_complete', windows=len(temporal_audit), segments=len(segments))
        progress(0.5, desc='Stitching finished · loading the captioner')
        if len(videos) == 3:
            captions, caption_audit = caption_three(videos, segments, session, instruction, progress, profile=profile)
        else:
            captions, caption_audit = caption_single(videos, segments, profile, session, instruction, progress)
        for row, segment_row in zip(captions, segments, strict=True):
            row['segmentation_provenance'] = segment_row['metadata'].get('segmentation_provenance', 'fine')
        runtime_event('annotation_complete', segments=len(captions), complete_captions=sum((r['status'] == 'complete' for r in captions)), elapsed_s=round(time.monotonic() - start, 3))
        return dict(schema_version='flora_demo_annotation_v1', duration_s=videos['main'].duration_s, segments=captions, elapsed_s=round(time.monotonic() - start, 2), source_clock='seconds relative to the first native source presentation timestamp', camera_roles=list(videos), profile=profile, model_revisions=MODELS, input_sha256={role: video.sha256 for role, video in videos.items()}, pipeline=dict(temporal_windows=temporal_audit, caption_inputs=caption_audit))
    finally:
        session.close()
