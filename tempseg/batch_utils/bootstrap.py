import hashlib
import json
import sys
from .config import VENDOR

def initialize():
    manifest = json.loads((VENDOR / 'MANIFEST.json').read_text())
    for relative, expected in manifest.items():
        path = VENDOR / relative
        if not path.resolve().is_relative_to(VENDOR.resolve()):
            raise RuntimeError('Invalid runtime manifest path')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Production code changed: {relative}')
    for path in [VENDOR, VENDOR / 'support', VENDOR / 'production_adapter']:
        sys.path.insert(0, str(path))
    import adapter
    return adapter
production = initialize()
