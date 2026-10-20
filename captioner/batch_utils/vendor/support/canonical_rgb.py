import numpy as np
from PIL import Image

def canonical_rgb(frame):
    assert frame.format.name == 'yuv420p'
    assert frame.width % 16 == 0 and frame.height % 2 == 0
    assert int(frame.colorspace) in (2, 5, 6), ('unsupported colorspace', frame.colorspace)
    assert int(frame.color_range) in (0, 1), ('unsupported range', frame.color_range)
    planes = []
    for p in frame.planes:
        a = np.frombuffer(p, dtype=np.uint8).reshape(p.height, p.line_size)[:, :p.width].astype(np.int32)
        planes.append(a)
    y, u, v = planes
    u = u.repeat(2, axis=0).repeat(2, axis=1)
    v = v.repeat(2, axis=0).repeat(2, axis=1)

    def q13(c):
        return c * 8192 + 32768 >> 16
    y = (y * 8 - 128) * q13(65536 * 255 // 219) >> 16
    u = u * 8 - 1024
    v = v * 8 - 1024
    r = y + (v * q13(104597) >> 16)
    g = y + (u * q13(-25675) >> 16) + (v * q13(-53279) >> 16)
    b = y + (u * q13(132201) >> 16)
    return Image.fromarray(np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8))
