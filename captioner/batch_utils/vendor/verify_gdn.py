from collections import Counter, defaultdict
import hashlib, json
from pathlib import Path

def verify(model, path, *, peft_prefix=False):
    import torch
    from safetensors import safe_open
    path = Path(path)
    mapping = json.loads((path / 'model.safetensors.index.json').read_text())['weight_map']
    wanted = {name for name in mapping if name.endswith(('.A_log', '.dt_bias'))}
    assert wanted
    actual = {}
    for name, p in model.named_parameters():
        if not name.endswith(('.A_log', '.dt_bias')):
            continue
        if peft_prefix:
            assert name.startswith('base_model.model.')
            name = name[len('base_model.model.'):]
        assert name not in actual
        actual[name] = p
    assert set(actual) == wanted, {'missing': sorted(wanted - set(actual)), 'extra': sorted(set(actual) - wanted)}
    groups = defaultdict(list)
    for name in wanted:
        groups[mapping[name]].append(name)
    evidence = []
    for filename, names in sorted(groups.items()):
        with safe_open(str(path / filename), framework='pt', device='cpu') as source:
            for name in sorted(names):
                saved = source.get_tensor(name)
                p = actual[name]
                assert p.dtype == torch.float32, (name, str(p.dtype), 'must preserve FP32 before loading')
                assert p.device.type == 'cuda', (name, 'must verify final device placement')
                value = p.detach().cpu()
                assert torch.equal(value, saved.float()), (name, 'checkpoint bits changed; upcasting rounded BF16 is insufficient')
                evidence.append(dict(name=name, elements=p.numel(), source_dtype=str(saved.dtype), runtime_dtype=str(p.dtype), device=str(p.device), fp32_value_sha256=hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(), saved_value_exact=True))
    return dict(status='PASS_NATIVE_GDN_FP32_CHECKPOINT_VALUES', tensors=len(evidence), source_dtypes=dict(Counter((r['source_dtype'] for r in evidence))), all_runtime_fp32=True, all_values_equal_saved_checkpoint=True, checked_after_final_device_placement=True, values=evidence)
