from PIL import Image, ImageOps
from typing import Any
VIEWS = ('ego', 'left_wrist', 'right_wrist')
CAPTION_TOP_SIZE = (768, 432)
CAPTION_WRIST_SIZE = (384, 216)
CAPTION_MOSAIC_WIDTH = 768
CAPTION_MOSAIC_HEIGHT = 648

def compose_frame(frames, views):
    canvas = Image.new('RGB', (640, 720))
    for view, (x, y, w, h) in zip(VIEWS, [(0, 0, 640, 480), (0, 480, 320, 240), (320, 480, 320, 240)]):
        if view in views:
            panel = ImageOps.contain(frames[view].convert('RGB'), (w, h), Image.Resampling.BICUBIC)
            canvas.paste(panel, (x + (w - panel.width) // 2, y + (h - panel.height) // 2))
    return canvas

def compose_top_dominant_mosaic(top: Any, left: Any, right: Any) -> Any:
    from PIL import Image
    top_panel = top.convert('RGB').resize(CAPTION_TOP_SIZE, Image.Resampling.BICUBIC)
    left_panel = left.convert('RGB').resize(CAPTION_WRIST_SIZE, Image.Resampling.BICUBIC)
    right_panel = right.convert('RGB').resize(CAPTION_WRIST_SIZE, Image.Resampling.BICUBIC)
    output = Image.new('RGB', (CAPTION_MOSAIC_WIDTH, CAPTION_MOSAIC_HEIGHT))
    output.paste(top_panel, (0, 0))
    output.paste(left_panel, (0, CAPTION_TOP_SIZE[1]))
    output.paste(right_panel, (CAPTION_WRIST_SIZE[0], CAPTION_TOP_SIZE[1]))
    return output
