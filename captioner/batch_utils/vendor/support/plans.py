import bisect, hashlib
from fractions import Fraction
import u2_contract as U2
ROLES = ('main', 'left_wrist', 'right_wrist')
HEADING = 'Each frame is a synchronized camera mosaic: main camera above, left wrist camera below left, and right wrist camera below right. All panels show the same physical time. Describe each physical operation once across the views.\n\n'
INSTRUCTION = 'Write one concise, self-contained English instruction for the local operation being attempted in TARGET.\nUse a natural imperative, concrete objects and the working hand; include meaningful support by the other hand and necessary source or destination. Say left hand, right hand or both hands. For a handover, name the object and both the giving and receiving hands.\nInfer the local goal from the video sequence. Use BEFORE and AFTER only to disambiguate TARGET, not to add operations occurring outside it. A failed or repeated attempt retains its established instruction; do not replace the goal with an accident, a success claim or a description of mechanical adjustments. Do not invent an object, hand role or hidden goal.\nOutput only the instruction, on one line, without timestamps, execution-quality commentary or the block names.'

def plans_for_episode(ep, clocks):
    clock = clocks['main']
    pts = [float(t * Fraction(clock['time_base'])) for t in clock['ticks']]
    fps = Fraction(clock['fps'])
    plans = []
    for unit in ep['units']:
        start, end = unit['native_scope_s']
        assert 0 <= start < end <= clock['duration_s'] + 1e-05
        blocks = []
        spans = [('BEFORE', max(0.0, start - 2.0), start), ('TARGET', start, end), ('AFTER', end, min(clock['duration_s'], end + 2.0))]
        for role, a, b in spans:
            lo, hi = (bisect.bisect_left(pts, a - 1e-09), bisect.bisect_left(pts, b - 1e-09))
            hi = min(hi, len(pts))
            factor = U2.TARGET_FRAME_FACTOR if role == 'TARGET' else U2.CONTEXT_FRAME_FACTOR
            if hi - lo < factor:
                if role != 'TARGET':
                    continue
                assert hi - lo == 1, (unit['target_id'], 'empty target')
                indices = [lo, lo]
                receipt = U2.repeated_single_target_receipt(lo, source_fps_num=fps.numerator, source_fps_den=fps.denominator)
                padding = [False, True]
            else:
                target = role == 'TARGET'
                indices, receipt = U2.duration_adaptive_sample(lo, hi, source_fps_num=fps.numerator, source_fps_den=fps.denominator, sample_fps=U2.TARGET_SAMPLE_FPS if target else U2.CONTEXT_SAMPLE_FPS, min_frames=U2.TARGET_MIN_FRAMES if target else U2.CONTEXT_MIN_FRAMES, max_frames=U2.TARGET_MAX_FRAMES if target else U2.CONTEXT_MAX_FRAMES, frame_factor=factor)
                padding = [False] * len(indices)
            blocks.append(dict(role=role, scope_s=[a, b], source_start_frame=lo, source_end_frame_exclusive=hi, indices=indices, pts_ticks=[clock['ticks'][i] for i in indices], padding_mask=padding, sampling_receipt=receipt))
        assert any((b['role'] == 'TARGET' for b in blocks))
        assert sum((len(b['indices']) for b in blocks)) <= U2.MAX_TOTAL_FRAMES
        prompt = HEADING + '\n\n'.join((b['role'] + ':\n<video>' for b in blocks)) + '\n\n' + INSTRUCTION
        plans.append(dict(target_id=unit['target_id'], unit_id=unit['unit_id'], episode_id=ep['episode_id'], source_id=ep['source_id'], group=ep['group'], native_scope_s=unit['native_scope_s'], camera_roles=list(ROLES), clock_fps=str(fps), time_base=clock['time_base'], blocks=blocks, prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), context_policy='common_real_up_to_2s_each_side', source_clock_origin=clock['first_pts'], do_sample_frames=False))
    return plans
