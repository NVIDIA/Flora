import hashlib, io
from PIL import Image
PACKED_IMAGE_LIMIT_BYTES = 21_866_496
SCALES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

def sha(b):
    return hashlib.sha256(b).hexdigest()

def png(im):
    b = io.BytesIO()
    im.save(b, format='PNG', compress_level=1, optimize=False)
    return b.getvalue()

def normalized(plan, frames, expected=None):
    factors = [expected['common_geometry']['scale']] if expected else SCALES
    for factor in factors:
        blocks = []
        count = 0
        for block in plan['blocks']:
            assets = []
            for start in range(0, len(block['indices']), 2):
                indices = block['indices'][start:start + 2]
                originals = [frames[i] for i in indices]
                w, h = originals[0].size
                nw = round(w * factor)
                nh = round(h * factor)
                packed = Image.new('RGB', (nw * len(indices), nh))
                rgb = []
                for pos, image in enumerate(originals):
                    if factor != 1:
                        image = image.resize((nw, nh), Image.Resampling.LANCZOS)
                    packed.paste(image, (pos * nw, 0))
                    rgb.append(sha(image.tobytes()))
                blob = png(packed)
                count += len(blob) + (-len(blob) % 3)
                assets.append(dict(indices=indices, size=list(packed.size), sha256=sha(blob), rgb_sha256=rgb, bytes=len(blob), _blob=blob))
            blocks.append(dict(role=block['role'], images=assets))
        if count <= PACKED_IMAGE_LIMIT_BYTES:
            break
    else:
        raise RuntimeError('No common caption input geometry fits')
    if expected:
        for old, new in zip(expected['frame_blocks'], blocks):
            assert old['role'] == new['role'] and len(old['images']) == len(new['images'])
            for a, b in zip(old['images'], new['images']):
                assert a['indices'] == b['indices'] and a['rgb_sha256'] == b['rgb_sha256'] and (a['size'] == b['size'])
    return dict(plan, frame_blocks=blocks, common_geometry=dict(scale=factor, source_pixel_unchanged=factor == 1, aligned_image_bytes=count, image_size_limit_bytes=PACKED_IMAGE_LIMIT_BYTES, sampling_frames_removed=0))

def source_frames(normalized):
    out = {}
    for block in normalized['frame_blocks']:
        for a in block['images']:
            image = Image.open(io.BytesIO(a['_blob'])).convert('RGB')
            n = len(a['indices'])
            w = image.width // n
            for pos, i in enumerate(a['indices']):
                frame = image.crop((pos * w, 0, (pos + 1) * w, image.height))
                assert sha(frame.tobytes()) == a['rgb_sha256'][pos]
                out[i] = frame
    return out
