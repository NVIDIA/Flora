from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import hashlib
import av
from .config import MAX_DURATION_S, MAX_DIMENSION, MAX_FRAME_COUNT

class InputError(ValueError):
    pass

@dataclass(frozen=True)
class Video:
    path: Path
    sha256: str
    rate: Fraction
    pts_us: tuple
    duration_us: int
    size: tuple
    cfr: bool

    @property
    def duration_s(self):
        return self.duration_us / 1000000

def probe(path):
    with av.open(str(path), options={'err_detect': 'explode'}) as container:
        if len(container.streams.video) != 1:
            raise InputError('Use a file containing one video stream.')
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        rate = stream.average_rate
        if not rate or rate < 4:
            raise InputError('The input needs at least 4 source frames per second.')
        size = (stream.codec_context.width, stream.codec_context.height)
        if min(size) <= 0 or max(size) > MAX_DIMENSION:
            raise InputError('Video dimensions must be at most 4096 pixels per side.')
        times, zero, last_length = ([], None, None)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise InputError('Every source frame needs a presentation timestamp.')
            zero = frame.pts if zero is None else zero
            t = Fraction(frame.pts - zero) * stream.time_base * 1000000
            if times and t <= times[-1]:
                raise InputError('Source timestamps must be strictly increasing.')
            if t >= MAX_DURATION_S * 1000000 or len(times) >= MAX_FRAME_COUNT:
                raise InputError(f'Video exceeds {MAX_DURATION_S:g} seconds or {MAX_FRAME_COUNT} frames.')
            times.append(t)
            last_length = Fraction(frame.duration) * stream.time_base * 1000000 if frame.duration else None
        if len(times) < 2:
            raise InputError('The video contains too few source frames.')
        tail = last_length or Fraction(1000000, rate)
        duration = round(times[-1] + tail)
        if duration > MAX_DURATION_S * 1000000:
            raise InputError(f'Video exceeds {MAX_DURATION_S:g} seconds.')
        cfr = all((abs(t - Fraction(i * 1000000, rate)) <= 1 for i, t in enumerate(times)))
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return Video(path, digest, rate, tuple(times), duration, size, cfr)

def check_synchronized(videos):
    main = videos['main']
    if len(videos) == 3:
        for role, video in videos.items():
            if video.rate != main.rate or video.pts_us != main.pts_us or video.duration_us != main.duration_us:
                raise InputError('Three camera files must have identical relative frame timestamps, frame counts and duration.')
        if not main.cfr:
            raise InputError('The three-view profile requires constant-rate synchronized source videos.')

def decode_selected(video, wanted, canonical=False):
    wanted = set(wanted)
    result = {}
    if not wanted:
        return result
    if min(wanted) < 0 or max(wanted) >= len(video.pts_us):
        raise InputError('A requested source frame is outside the video.')
    with av.open(str(video.path), options={'err_detect': 'explode'}) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        zero = None
        for i, frame in enumerate(container.decode(stream)):
            zero = frame.pts if zero is None else zero
            if i in wanted:
                if Fraction(frame.pts - zero) * stream.time_base * 1000000 != video.pts_us[i]:
                    raise InputError('Source frame timestamps changed while decoding.')
                use_canonical = canonical and frame.format.name == 'yuv420p' and (frame.width % 16 == 0) and (frame.height % 2 == 0) and (int(frame.colorspace) in (2, 5, 6)) and (int(frame.color_range) in (0, 1))
                if use_canonical:
                    from canonical_rgb import canonical_rgb
                    result[i] = canonical_rgb(frame)
                else:
                    result[i] = frame.to_image().convert('RGB')
            if i >= max(wanted):
                break
    if set(result) != wanted:
        raise InputError('A requested source frame could not be decoded.')
    return result

def frames_for(videos, wanted, mosaic='YAM'):
    frames = {role: decode_selected(video, wanted, canonical=len(videos) == 3) for role, video in videos.items()}
    if len(videos) == 1:
        return frames['main']
    from composers import compose_top_dominant_mosaic, compose_frame
    if mosaic == 'YAM':
        return {i: compose_top_dominant_mosaic(frames['main'][i], frames['left_wrist'][i], frames['right_wrist'][i]) for i in wanted}
    if mosaic == 'Dex':
        return {i: compose_frame(dict(ego=frames['main'][i], left_wrist=frames['left_wrist'][i], right_wrist=frames['right_wrist'][i]), ('ego', 'left_wrist', 'right_wrist')) for i in wanted}
    raise InputError('Choose a supported camera layout.')
