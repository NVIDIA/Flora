import bisect
from decimal import Decimal, ROUND_HALF_EVEN

def microseconds(value):
    return int((Decimal(str(value)) * 1000000).to_integral_value(rounding=ROUND_HALF_EVEN))

def frame_span(pts_s, start_us, end_us):
    if type(start_us) is not int or type(end_us) is not int or start_us < 0 or (end_us <= start_us):
        raise ValueError('invalid integer-microsecond source interval')
    times = [microseconds(t) for t in pts_s]
    if not times or any((a >= b for a, b in zip(times, times[1:]))):
        raise ValueError('PTS not strictly increasing after microsecond serialization')
    return (bisect.bisect_left(times, start_us), bisect.bisect_left(times, end_us))

def native_blocks(pts_s, start_us, end_us, duration_us, rate, contract):
    if end_us > duration_us:
        raise ValueError('target beyond source')
    windows = [('BEFORE', max(0, start_us - 2000000), start_us), ('TARGET', start_us, end_us), ('AFTER', end_us, min(duration_us, end_us + 2000000))]
    blocks = []
    for role, a, b in windows:
        if a == b:
            continue
        lo, hi = frame_span(pts_s, a, b)
        target = role == 'TARGET'
        factor = contract.TARGET_FRAME_FACTOR if target else contract.CONTEXT_FRAME_FACTOR
        if hi - lo < factor:
            if target:
                raise ValueError('target_has_fewer_than_native_frame_factor')
            continue
        ids, receipt = contract.duration_adaptive_sample(lo, hi, source_fps_num=rate.numerator, source_fps_den=rate.denominator, sample_fps=contract.TARGET_SAMPLE_FPS if target else contract.CONTEXT_SAMPLE_FPS, min_frames=contract.TARGET_MIN_FRAMES if target else contract.CONTEXT_MIN_FRAMES, max_frames=contract.TARGET_MAX_FRAMES if target else contract.CONTEXT_MAX_FRAMES, frame_factor=factor)
        assert len(set(ids)) == len(ids) and all((lo <= i < hi for i in ids))
        times = [microseconds(pts_s[i]) for i in ids]
        assert all((a <= t < b for t in times))
        blocks.append({'role': role, 'interval_us': [a, b], 'start_frame': lo, 'end_frame_exclusive': hi, 'frame_indices': ids, 'actual_pts_us': times, 'sampling_receipt': receipt, 'model_timestamp_clock': 'native ordinal/average FPS; exact source PTS retained separately'})
    assert sum((len(b['frame_indices']) for b in blocks)) <= contract.MAX_TOTAL_FRAMES
    return blocks
