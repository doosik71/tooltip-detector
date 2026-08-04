"""Shared albumentations pipelines for evaluation-mode inference.

Used by scripts/train-model.py (validation split), scripts/eval-model.py,
scripts/compare-speed.py, scripts/tooltip-detector.py and
scripts/tooltip-tracker.py so all of them preprocess frames identically to
how the model was trained.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _eval_transform(image_size=(480, 736)):
    h, w = image_size
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
