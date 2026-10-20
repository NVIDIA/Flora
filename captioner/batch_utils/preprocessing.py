from fractions import Fraction
import io
import json
from PIL import Image
from .bootstrap import production as a

def instruction_suffix(instruction):
    value = instruction.strip()
    if not value:
        return ''
    if len(value) > 2000:
        raise ValueError('Keep the optional task instruction below 2000 characters.')
    return '\n' + json.dumps({'episode_instruction': value}, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def human_transport(frames):
    from flora_runtime.exact_frame_serving import encode_regularized_image
    output = {}
    for index, image in frames.items():
        payload = encode_regularized_image(image, encoding='jpeg_q98_444')['payload']
        with Image.open(io.BytesIO(payload)) as decoded:
            output[index] = decoded.convert('RGB')
    return output

def temporal_inputs(frames, window, pool, proc, roles, instruction=''):
    from transformers.video_utils import VideoMetadata
    from flora_runtime.exact_frame_serving import regularized_resize_dims
    profile = window.window_profile
    if len(frames) != profile.slot_count:
        raise ValueError('Incorrect native temporal slot count')
    regularized = []
    for image in frames:
        image = image.convert('RGB')
        height, width = regularized_resize_dims(height=image.height, width=image.width, min_pixels=256, max_pixels=40000 if len(roles) == 1 else 160000)
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.BICUBIC)
        regularized.append(image)
    prompt = a.temporal_prompt(window, pool) + instruction_suffix(instruction)
    text = proc.apply_chat_template([dict(role='user', content=prompt.replace('<video>', proc.video_token))], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = proc(text=[text], videos=[regularized], video_metadata=[VideoMetadata(total_num_frames=profile.slot_count, fps=profile.slot_fps, frames_indices=list(range(profile.slot_count)))], do_sample_frames=False, return_metadata=True, return_tensors='pt')
    inputs.pop('video_metadata', None)
    return (inputs, text)

def three_view_caption_inputs(plan, frames, proc, instruction=''):
    from transformers.video_utils import VideoMetadata
    from plans import HEADING, INSTRUCTION
    time_base = Fraction(plan['time_base'])
    content, videos, metadata = ([HEADING], [], [])
    for block in plan['blocks']:
        times = ', '.join((f'{float(t * time_base):.9f}' for t in block['pts_ticks']))
        content.append(block['role'] + ':\nFrames are in this order; native source timestamps in seconds: [' + times + '].\n' + proc.video_token)
        videos.append([frames[i] for i in block['indices']])
        metadata.append(VideoMetadata(total_num_frames=len(block['indices']), fps=float(Fraction(plan['clock_fps'])), frames_indices=[i - block['source_start_frame'] for i in block['indices']]))
    content.append(INSTRUCTION + instruction_suffix(instruction))
    text = proc.apply_chat_template([dict(role='user', content=''.join(content))], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    result = proc(text=[text], videos=videos, video_metadata=metadata, do_sample_frames=False, return_tensors='pt', padding=True)
    return (result, text)
