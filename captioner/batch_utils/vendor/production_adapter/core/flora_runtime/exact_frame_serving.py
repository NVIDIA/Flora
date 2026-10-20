# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import hashlib
import io
import math
from typing import Any
PIXEL_HASH_SCHEMA = 'rgb_u8_width_height_v1'

class ExactFrameServingError(ValueError):
    pass

def pixel_sha256(image: Any) -> str:
    rgb = image.convert('RGB')
    header = f'{PIXEL_HASH_SCHEMA}\x00RGB\x00{rgb.width}\x00{rgb.height}\x00'.encode('ascii')
    return hashlib.sha256(header + rgb.tobytes()).hexdigest()

def regularized_resize_dims(*, height: int, width: int, min_pixels: int=16 * 16, max_pixels: int=512 * 512) -> tuple[int, int]:
    if min(height, width, min_pixels, max_pixels) <= 0 or min_pixels > max_pixels:
        raise ExactFrameServingError('regularized resize arguments are invalid')
    area = height * width
    if area > max_pixels:
        scale = math.sqrt(max_pixels / area)
        return (int(height * scale), int(width * scale))
    if area < min_pixels:
        scale = math.sqrt(min_pixels / area)
        return (int(height * scale), int(width * scale))
    return (height, width)

def regularize_image(image: Any) -> Any:
    from PIL import Image
    source = image.convert('RGB')
    height, width = regularized_resize_dims(height=source.height, width=source.width)
    if source.size == (width, height):
        return source
    return source.resize((width, height), resample=Image.Resampling.BICUBIC)

def encode_regularized_image(image: Any, *, encoding: str) -> dict[str, Any]:
    regularized = regularize_image(image)
    stream = io.BytesIO()
    if encoding == 'png_level6':
        regularized.save(stream, format='PNG', optimize=False, compress_level=6)
    elif encoding == 'jpeg_q98_444':
        regularized.save(stream, format='JPEG', quality=98, subsampling=0, optimize=False)
    else:
        raise ExactFrameServingError(f'unsupported exact-frame cache encoding: {encoding}')
    return {'payload': stream.getvalue(), 'width': regularized.width, 'height': regularized.height, 'rgb_pixel_sha256': pixel_sha256(regularized)}
