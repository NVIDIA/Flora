from collections import Counter
import hashlib
from pathlib import Path
import types

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def load(spec, save):
    import torch
    from transformers import AutoModelForImageTextToText
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel
    from safetensors.torch import load_file
    from peft import PeftConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from production_gdn_fix import _keep_native_gdn_fp32
    from verify_gdn import verify
    _keep_native_gdn_fp32(Qwen3_5PreTrainedModel)
    base, info = AutoModelForImageTextToText.from_pretrained(spec['base'], local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa', output_loading_info=True)
    assert not any((info.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'])), info
    path = Path(spec['export']) / 'adapter_model.safetensors'
    assert digest(path) == spec['adapter_sha256']
    source = load_file(str(path), device='cpu')
    assert len(source) == 496 and sum((v.numel() for v in source.values())) == 86556672
    assert all((v.dtype == torch.float32 for v in source.values())), 'Expected corrected FP32 checkpoint'
    model = get_peft_model(base, PeftConfig.from_pretrained(spec['export'], local_files_only=True), adapter_name='candidate', autocast_adapter_dtype=True)
    initial = get_peft_model_state_dict(model, adapter_name='candidate')
    assert set(initial) == set(source)
    assert all((v.dtype == torch.float32 for v in initial.values())), 'PEFT must create FP32 adapter holders'
    del initial
    load_info = set_peft_model_state_dict(model, source, adapter_name='candidate')
    assert not load_info.unexpected_keys, load_info
    actual = get_peft_model_state_dict(model, adapter_name='candidate')
    assert set(actual) == set(source)
    assert all((actual[k].dtype == v.dtype and torch.equal(actual[k].detach().cpu(), v) for k, v in source.items())), 'Saved FP32 LoRA values must be loaded exactly'
    del actual
    model.to('cuda:0').eval()
    actual = get_peft_model_state_dict(model, adapter_name='candidate')
    assert all((actual[k].dtype == torch.float32 and torch.equal(actual[k].detach().cpu(), v) for k, v in source.items())), 'Final device placement must preserve saved FP32 values'
    save('ADAPTER_VALUE_VERIFICATION.json', dict(tensors=496, elements=86556672, source_dtypes=dict(Counter((str(v.dtype) for v in source.values()))), holders_fp32_before_assignment=True, all_saved_values_exact_before_and_after_placement=True, adapter_sha256=spec['adapter_sha256'], merged=False))
    del source, actual
    gdn = verify(model, spec['base'], peft_prefix=True)
    save('GDN_VALUE_VERIFICATION.json', gdn)
    original_generate = model.generate

    def generate_bf16(self, *args, **kwargs):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            return original_generate(*args, **kwargs)
    model.generate = types.MethodType(generate_bf16, model)
    save('MODEL_LOADED.json', dict(adapter_persistent_dtype='torch.float32', compute_autocast='torch.bfloat16', gdn_values_exact=True, base_loading_info=info, parameter_dtypes=dict(Counter((str(p.dtype) for p in model.parameters()))), input_output_generation_policy_changed=False))
    return model
