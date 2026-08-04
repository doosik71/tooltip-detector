"""Checkpoint / results path conventions shared by all training and inference
scripts, so a single layout definition can't drift out of sync between them.

Layout
------
  data/models/<target-mode>/<model-type>/{best.pt,last.pt}
  data/results/<target-mode>/<model-type>/{summary.json,per_tip.csv}

`target-mode` (see ttd.dataset.TARGET_MODES) comes first because it is the
axis that decides which training data a checkpoint needed (segmentation
masks or tip points only); `model-type` (see ttd.model.REGISTRY) is the
network architecture trained on that data.
"""

import os

from ttd.dataset import DEFAULT_TARGET_MODE


def model_dir(
    model_type: str,
    target_mode: str = DEFAULT_TARGET_MODE,
    root: str = "data/models",
) -> str:
    return os.path.join(root, target_mode, model_type)


def default_model_path(
    model_type: str,
    target_mode: str = DEFAULT_TARGET_MODE,
    root: str = "data/models",
) -> str:
    return os.path.join(model_dir(model_type, target_mode, root), "best.pt")


def default_results_dir(
    model_type: str,
    target_mode: str = DEFAULT_TARGET_MODE,
    root: str = "data/results",
) -> str:
    return os.path.join(root, target_mode, model_type)
