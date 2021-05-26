def get_custom_model(name, **kwargs):
    if 'wide_resnet28' in name:
        from .wrn28 import build_wide_resnet28
        return build_wide_resnet28(name, **kwargs)
    raise ValueError(f"Unsupported model: {name}")
