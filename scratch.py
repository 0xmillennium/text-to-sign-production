import re

with open('/home/enes/Projects/text-to-sign-production/src/text_to_sign_production/workflows/model/processing/calibration.py', 'r') as f:
    text = f.read()

# Remove from ProviderRealCalibrationTrace dataclass fields
text = re.sub(r'    actual_provider_model_used: bool\n', '', text)
text = re.sub(r'    actual_text_backbone_class: str \| None = None\n', '', text)
text = re.sub(r'    actual_text_backbone_module: str \| None = None\n', '', text)
text = re.sub(r'    actual_provider_backbone_used: bool = False\n', '', text)
text = re.sub(r'    provider_real_authoritative: bool = False\n', '', text)
text = re.sub(r'    actual_provider_loss_used: bool\n', '', text)

# Remove from validate_provider_real_trace checks
text = re.sub(r'    if trace\.actual_provider_model_used is not True:\n        raise ModelWorkflowInvariantError\("provider-real trace did not use an actual provider model\."\)\n', '', text)
text = re.sub(r'    if trace\.provider_key == "base_direct":\n        if trace\.actual_provider_backbone_used is not True:\n            raise ModelWorkflowInvariantError\("base_direct provider-real trace did not use provider text backbone\."\)\n        if trace\.provider_real_authoritative is not True:\n            raise ModelWorkflowInvariantError\("base_direct provider-real trace is not authoritative\."\)\n        if trace\.actual_text_backbone_class != "FlanT5TextBackbone":\n            raise ModelWorkflowInvariantError\("base_direct provider-real trace did not use FlanT5TextBackbone\."\)\n', '', text)
text = re.sub(r'        if trace\.actual_provider_loss_used is not True:\n            raise ModelWorkflowInvariantError\("provider-real training trace did not use provider loss\."\)\n', '', text)

# Add dataloader options check
import_stmt = '''    if context is not None:
        from text_to_sign_production.modeling.candidates.calibration_runtime import (
            calibration_reader_runtime_options,
        )
        runtime_options = calibration_reader_runtime_options(context.effective_config)'''
text = text.replace('    if context is not None:\n', import_stmt + '\n')
check_stmt = '''        if trace.surface_provider_config_sha256 != context.provider_config_hash:
            raise ModelWorkflowInvariantError("provider-real trace surface provider config hash mismatch.")
        if trace.calibration_surface_reader_num_workers_used is not None:
            if trace.calibration_surface_reader_num_workers_used != runtime_options.num_workers:
                raise ModelWorkflowInvariantError(
                    "provider-real trace reader num_workers mismatch with effective config."
                )
            expected_mode = "multiprocess" if runtime_options.num_workers > 0 else "single_process"
            if trace.calibration_surface_reader_worker_mode != expected_mode:
                raise ModelWorkflowInvariantError(
                    "provider-real trace reader worker_mode mismatch with effective config."
                )
            if trace.calibration_surface_reader_prefetch_factor_used != runtime_options.prefetch_factor:
                raise ModelWorkflowInvariantError(
                    "provider-real trace reader prefetch_factor mismatch with effective config."
                )
            if trace.calibration_surface_reader_persistent_workers_used != runtime_options.persistent_workers:
                raise ModelWorkflowInvariantError(
                    "provider-real trace reader persistent_workers mismatch with effective config."
                )'''
text = text.replace('        if trace.surface_provider_config_sha256 != context.provider_config_hash:\n            raise ModelWorkflowInvariantError("provider-real trace surface provider config hash mismatch.")', check_stmt)


with open('/home/enes/Projects/text-to-sign-production/src/text_to_sign_production/workflows/model/processing/calibration.py', 'w') as f:
    f.write(text)
