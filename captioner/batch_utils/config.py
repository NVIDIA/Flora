from pathlib import Path
import json
import os
ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / 'vendor'
MODELS = json.loads((ROOT / 'models.json').read_text())
MODEL_CACHE = Path(os.environ.get('FLORA_MODEL_CACHE', Path.home() / '.cache' / 'flora' / 'models'))
MAX_DURATION_S = 120
MAX_UPLOAD_BYTES = 256 * 1024 * 1024
MAX_DIMENSION = 4096
MAX_FRAME_COUNT = 7200
PROFILE_POOLS = {'Human': 'human', 'Human actions': 'human', 'YAM': 'yam', 'Dex': 'dex', 'Tianji': 'tianji', 'Robot actions': 'yam'}
