from pathlib import Path
import math
import threading
from .config import MAX_UPLOAD_BYTES, MODELS
_REQUEST_LOCK = threading.RLock()
_PROFILES = {'human': 'Human', 'yam': 'YAM', 'dex': 'Dex', 'tianji': 'Tianji'}

class DataEnginePipeline:

    @staticmethod
    def _inputs(video, profile, left_wrist, right_wrist, yam32, instruction):
        from .media import InputError, probe, check_synchronized
        if not isinstance(profile, str) or profile.lower() not in _PROFILES:
            raise InputError('profile must be human, yam, dex, or tianji')
        canonical = _PROFILES[profile.lower()]
        if bool(left_wrist) != bool(right_wrist):
            raise InputError('Provide both left and right wrist videos together.')
        if canonical == 'Human' and left_wrist:
            raise InputError('The human profile accepts one video.')
        if not isinstance(yam32, bool):
            raise InputError('yam32 must be true or false.')
        if yam32 and (canonical != 'YAM' or not left_wrist):
            raise InputError('yam32 requires YAM and all three camera views.')
        if not isinstance(instruction, str) or len(instruction.strip()) > 2000:
            raise InputError('instruction must be a string of at most 2000 characters.')
        paths = {'main': video}
        if left_wrist:
            paths.update(left_wrist=left_wrist, right_wrist=right_wrist)
        videos = {}
        for role, value in paths.items():
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise InputError('Each camera path must be a nonempty local path.')
            path = Path(value).expanduser().resolve(strict=True)
            if not path.is_file() or not 0 < path.stat().st_size <= MAX_UPLOAD_BYTES:
                raise InputError('Each input must be a nonempty local video of at most 256 MiB.')
            videos[role] = probe(path)
        check_synchronized(videos)
        return (videos, canonical)

    def inspect_inputs(self, video, *, profile='human', left_wrist=None, right_wrist=None, yam32=False, instruction=''):
        videos, canonical = self._inputs(video, profile, left_wrist, right_wrist, yam32, instruction)
        return {'status': 'inputs_validated', 'profile': canonical, 'yam32': yam32, 'duration_s': videos['main'].duration_s, 'model_revisions': MODELS, 'new_model_calls': 0, 'cameras': {role: {'sha256': v.sha256, 'source_fps': str(v.rate), 'frames': len(v.pts_us), 'size': list(v.size), 'constant_frame_rate': v.cfr, 'duration_s': v.duration_s} for role, v in videos.items()}}

    def annotate(self, video, *, profile='human', left_wrist=None, right_wrist=None, yam32=False, instruction='', progress=None, include_audit=False):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for inference. Use --validate-only to check inputs on CPU.')
        from . import pipeline
        with _REQUEST_LOCK:
            videos, canonical = self._inputs(video, profile, left_wrist, right_wrist, yam32, instruction)
            result = pipeline.annotate(videos, canonical, yam32, instruction, progress or (lambda *a, **k: None))
        duration = result['duration_s']
        for segment in result['segments']:
            if not (math.isfinite(segment['start_s']) and math.isfinite(segment['end_s']) and (0 <= segment['start_s'] < segment['end_s'] <= duration + 0.001)):
                raise RuntimeError('The pipeline returned an invalid caption interval.')
        if not include_audit:
            result.pop('pipeline', None)
        return result
