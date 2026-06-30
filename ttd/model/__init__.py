import importlib

REGISTRY = {
    "monai":      "ttd.model.monai",
    "monai_mini": "ttd.model.monai_mini",
}


def build(model_type: str, **kwargs) -> "torch.nn.Module":
    """Instantiate TooltipDetector for the given model_type."""
    if model_type not in REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. Choose from {list(REGISTRY)}"
        )
    module = importlib.import_module(REGISTRY[model_type])
    return module.TooltipDetector(**kwargs)
