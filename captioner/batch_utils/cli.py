from pathlib import Path
import argparse
import json
import re
import sys
from .api import DataEnginePipeline

def parser():
    p = argparse.ArgumentParser(description='FLORA: TempSeg -> Captioner with native video preprocessing.')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--video', type=Path, help='Main/egocentric local video.')
    source.add_argument('--input-jsonl', type=Path, help='One video job per line; paths relative to this JSONL.')
    p.add_argument('--profile', choices=['human', 'yam', 'dex', 'tianji'], default='human')
    p.add_argument('--left-wrist', type=Path)
    p.add_argument('--right-wrist', type=Path)
    p.add_argument('--yam32', action='store_true', help='Use the native 32-second three-view YAM temporal profile.')
    p.add_argument('--instruction', default='', help='Optional episode instruction; blank by default.')
    p.add_argument('--output', type=Path, help='Single-video result; default annotations.json.')
    p.add_argument('--output-dir', type=Path, help='Required with --input-jsonl for inference.')
    p.add_argument('--validate-only', action='store_true', help='Check media/timestamps on CPU; no weight downloads or generation.')
    p.add_argument('--include-audit', action='store_true', help='Include temporal/model-input audit records in the result.')
    p.add_argument('--overwrite', action='store_true', help='Explicitly allow replacing existing output files.')
    return p

def jobs(args, p):
    if args.video:
        if args.output_dir:
            p.error('--output-dir is for JSONL jobs; use --output for a single video')
        return [(dict(video=args.video, profile=args.profile, left_wrist=args.left_wrist, right_wrist=args.right_wrist, yam32=args.yam32, instruction=args.instruction), args.output or Path('annotations.json'))]
    if args.output or args.left_wrist or args.right_wrist or args.yam32 or args.instruction or (args.profile != 'human'):
        p.error('Put per-video profile/cameras/instruction in JSONL rows.')
    if not args.output_dir and (not args.validate_only):
        p.error('--output-dir is required with --input-jsonl')
    source = args.input_jsonl.expanduser().resolve(strict=True)
    items, ids = ([], set())
    allowed = {'id', 'video', 'profile', 'left_wrist', 'right_wrist', 'yam32', 'instruction'}
    for line_number, line in enumerate(source.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) - allowed or 'video' not in row:
            raise ValueError(f'Invalid JSONL job on line {line_number}.')
        name = row.pop('id', f'video_{line_number:06d}')
        if not isinstance(name, str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', name) or name in ids:
            raise ValueError(f'Invalid or duplicate job id on line {line_number}.')
        ids.add(name)
        for key in ['video', 'left_wrist', 'right_wrist']:
            if row.get(key) is not None:
                if not isinstance(row[key], str) or not row[key].strip():
                    raise ValueError(f'Invalid camera path on JSONL line {line_number}.')
                value = Path(row[key]).expanduser()
                row[key] = value if value.is_absolute() else source.parent / value
        items.append((row, (args.output_dir or Path('.')) / (name + '.json')))
    if not items:
        raise ValueError('The JSONL contains no video jobs.')
    return items

def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        items = jobs(args, p)
        if not args.validate_only and (not args.overwrite):
            for _, output in items:
                if output.exists():
                    raise FileExistsError(f'Output already exists: {output}. Use --overwrite to replace it.')
        engine = DataEnginePipeline()
        for kwargs, output in items:
            if args.validate_only:
                print(json.dumps(engine.inspect_inputs(**kwargs), allow_nan=False))
                continue
            result = engine.annotate(**kwargs, include_audit=args.include_audit, progress=lambda fraction, desc='': print(f'{fraction:.0%} {desc}', file=sys.stderr, flush=True))
            payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('w' if args.overwrite else 'x') as f:
                f.write(payload)
            print(json.dumps({'output': str(output), 'segments': len(result['segments'])}))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f'FLORA inference failed: {error}', file=sys.stderr)
        return 1
