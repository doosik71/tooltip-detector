"""Checkpoint / results path conventions shared by all training and inference
scripts, so a single layout definition can't drift out of sync between them.

Layout
------
  data/models/<dataset-name>/<target-mode>/<model-type>/{best.pt,last.pt}
  data/results/<dataset-name>/<target-mode>/<model-type>/{summary.json,per_tip.csv}

`dataset-name` (see ttd.dataset.DATASETS) comes first because it is the axis
that decides which raw frames a checkpoint was trained/evaluated on;
`target-mode` (see ttd.dataset.TARGET_MODES) comes next because it decides
which training data was used within that dataset (segmentation masks or tip
points only); `model-type` (see ttd.model.REGISTRY) is the network
architecture trained on that data.
"""

import os

from ttd.dataset import DEFAULT_TARGET_MODE


def model_dir(
    model_type: str,
    dataset_name: str,
    target_mode: str = DEFAULT_TARGET_MODE,
    root: str = "data/models",
) -> str:
    return os.path.join(root, dataset_name, target_mode, model_type)


def default_model_path(
    model_type: str,
    dataset_name: str,
    target_mode: str = DEFAULT_TARGET_MODE,
    root: str = "data/models",
) -> str:
    return os.path.join(model_dir(model_type, dataset_name, target_mode, root), "best.pt")


def default_results_dir(
    model_type: str,
    dataset_name: str,
    target_mode: str = DEFAULT_TARGET_MODE,
    root: str = "data/results",
) -> str:
    return os.path.join(root, dataset_name, target_mode, model_type)
