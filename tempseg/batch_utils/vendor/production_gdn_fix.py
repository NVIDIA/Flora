def _keep_native_gdn_fp32(model_class: type) -> None:
    model_class._keep_in_fp32_modules_strict = sorted(set(getattr(model_class, '_keep_in_fp32_modules_strict', None) or ()) | {'A_log', 'dt_bias'})
