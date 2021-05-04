def get_custom_model(name, **kwargs):
    if name.startswith('wide_resnet28'):
        if name.endswith('_tf') or name.endswith('_fixmatch'):
            # TensorFlow or FixMatch version
            from .wrn28 import build_wide_resnet28_tf
            return build_wide_resnet28_tf(name, **kwargs)
        else:
            from .wrn28 import build_wide_resnet28
            return build_wide_resnet28(name, **kwargs)
    raise ValueError(f"Unsupported model: {name}")
