from fractions import Fraction
import hashlib
import math
from pathlib import Path
from PIL import Image
INTENT_INSTRUCTION = 'Write one concise, self-contained English instruction for the local operation being attempted in TARGET.\nUse a natural imperative, concrete objects and the working hand; include meaningful support by the other hand and necessary source or destination. Say left hand, right hand or both hands. For a handover, name the object and both the giving and receiving hands.\nInfer the local goal from the video sequence. Use BEFORE and AFTER only to disambiguate TARGET, not to add operations occurring outside it. A failed or repeated attempt retains its established instruction; do not replace the goal with an accident, a success claim or a description of mechanical adjustments. Do not invent an object, hand role or hidden goal.\nOutput only the instruction, on one line, without timestamps, execution-quality commentary or the block names.'

def validate(sample):
    blocks = sample['blocks']
    roles = [b['role'] for b in blocks]
    if roles not in [['TARGET'], ['BEFORE', 'TARGET'], ['TARGET', 'AFTER'], ['BEFORE', 'TARGET', 'AFTER']]:
        raise ValueError('Missing TARGET, duplicate or reordered temporal role')
    if sample['camera_count'] not in (1, 3):
        raise ValueError('Unimplemented camera renderer')
    for b in blocks:
        start, end = (b['source_start_frame'], b['source_end_frame_exclusive'])
        indices = b['block_local_timestamp_indices']
        n = len(b['paths'])
        fps = Fraction(b['clock_fps_num'], b['clock_fps_den'])
        if fps <= 0 or end <= start or (not n) or (n != len(indices)) or (n != len(b['frames'])):
            raise ValueError('Invalid exact frame plan')
        if any((i < 0 or i >= end - start for i in indices)):
            raise ValueError('Frame outside block clock')
        if any((a > b for a, b in zip(indices, indices[1:]))):
            raise ValueError('Non-monotone timestamps')
        original = b.get('original_descriptor', {})
        if 'frame_indices' in original and original['frame_indices'] != [start + i for i in indices]:
            raise ValueError('Source frame ordinals disagree with block-local clock')
        if len(set(indices)) != n and (not any(original.get('frame_cache_padding_mask', []))):
            raise ValueError('Repeated frame requires explicit padding evidence')
        if Fraction(b['effective_fps_num'], b['effective_fps_den']) != Fraction(n) * fps / (end - start):
            raise ValueError('Effective sampling rate does not match block duration')
        if n % 2:
            raise ValueError('No implicit temporal pair padding')
    expected = sample['request']['target']['native_span']
    target = next((b for b in blocks if b['role'] == 'TARGET'))
    if expected['start_frame'] is not None and (target['source_start_frame'], target['source_end_frame_exclusive']) != (expected['start_frame'], expected['end_frame_exclusive']):
        raise ValueError('Accepted target span changed')
    text = sample['request']['target']['text']
    if hashlib.sha256(text.encode()).hexdigest() != sample['request']['target']['sha256']:
        raise ValueError('Accepted target changed')

def prompt(sample, video_token):
    validate(sample)
    heading = ''
    if sample['camera_count'] == 3:
        if sample['camera_roles'] != ['main', 'left_wrist', 'right_wrist']:
            raise ValueError('Camera roles do not match the mosaic renderer')
        heading = 'Each frame is a synchronized camera mosaic: main camera above, left wrist camera below left, and right wrist camera below right. All panels show the same physical time. Describe each physical operation once across the views.\n\n'
    return heading + '\n\n'.join((b['role'] + ':\n' + video_token for b in sample['blocks'])) + '\n\n' + INTENT_INSTRUCTION

def load_frames(sample, root, preserve_canvas=True):
    validate(sample)
    videos = []
    sizes = []
    max_pixels = 524288 if preserve_canvas and sample['camera_count'] == 3 else 262144
    for b in sample['blocks']:
        frames = []
        shapes = []
        for path, receipt in zip(b['paths'], b['frames'], strict=True):
            p = Path(root) / path
            if not p.resolve().is_relative_to(Path(root).resolve()):
                raise ValueError('Unowned frame path')
            raw = p.read_bytes()
            if hashlib.sha256(raw).hexdigest() != receipt['png_sha256']:
                raise ValueError('PNG changed')
            with Image.open(p) as im:
                im.load()
                frame = im.convert('RGB')
            if list(frame.size) != receipt['size']:
                raise ValueError('Frame dimensions changed')
            original = list(frame.size)
            area = frame.width * frame.height
            if area > max_pixels:
                scale = math.sqrt(max_pixels / area)
                frame = frame.resize((int(frame.width * scale), int(frame.height * scale)))
            frames.append(frame)
            shapes.append(dict(original=original, regularized=list(frame.size)))
        videos.append(frames)
        sizes.append(shapes)
    return (videos, sizes)

def processor_inputs(sample, root, processor, preserve_canvas=True):
    from transformers.video_utils import VideoMetadata
    videos, sizes = load_frames(sample, root, preserve_canvas)
    metadata = [VideoMetadata(total_num_frames=len(frames), fps=float(Fraction(b['clock_fps_num'], b['clock_fps_den'])), frames_indices=list(b['block_local_timestamp_indices'])) for frames, b in zip(videos, sample['blocks'], strict=True)]
    text = processor.apply_chat_template([dict(role='user', content=prompt(sample, processor.video_token))], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=[text], videos=videos, video_metadata=metadata, do_sample_frames=False, return_metadata=True, return_tensors='pt')
    return (inputs, sizes, text)
