"""Shared dataset path resolution for multi-dataset support.

Data layout convention:

    data/dataset-src/<name>/            read-only original source (videos)
    data/dataset/<name>/progressive/    deinterlaced/reencoded videos
    data/dataset/<name>/images/         extracted frames (train/val/test)
    data/dataset/<name>/segmentation/   binary masks (train/val/test)
    data/dataset/<name>/annotation/     bbox/tip JSON (train/val/test)
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

DATA_ROOT = Path("./data")


def dataset_src_dir(name: str) -> Path:
    return DATA_ROOT / "dataset-src" / name


def progressive_dir(name: str) -> Path:
    return DATA_ROOT / "dataset" / name / "progressive"


def images_dir(name: str) -> Path:
    return DATA_ROOT / "dataset" / name / "images"


def segmentation_dir(name: str) -> Path:
    return DATA_ROOT / "dataset" / name / "segmentation"


def annotation_dir(name: str) -> Path:
    return DATA_ROOT / "dataset" / name / "annotation"


def list_datasets() -> list[str]:
    """Return the sorted union of dataset names under data/dataset-src and data/dataset."""
    names: set[str] = set()
    for root in (DATA_ROOT / "dataset-src", DATA_ROOT / "dataset"):
        if root.is_dir():
            names.update(path.name for path in root.iterdir() if path.is_dir())
    return sorted(names)


def resolve_path(
    explicit: Path | None,
    dataset: str | None,
    factory: Callable[[str], Path],
    flag: str,
) -> Path:
    """Resolve a CLI path argument from an explicit value or a --dataset default.

    An explicit value always wins. If neither an explicit value nor a dataset
    name is given, raise a clear error naming the missing flag.
    """
    if explicit is not None:
        return explicit
    if dataset is None:
        raise ValueError(f"Either --dataset or {flag} must be provided.")
    return factory(dataset)


def resolve_video_input(explicit: Path | None, dataset: str | None) -> Path:
    """Resolve generate_dataset.py's video input directory.

    An explicit value always wins. Otherwise, prefer the dataset's progressive
    output if it exists and is non-empty (deinterlaced/reencoded videos),
    falling back to the read-only dataset-src videos when no progressive step
    was run for this dataset.
    """
    if explicit is not None:
        return explicit
    if dataset is None:
        raise ValueError("Either --dataset or --input must be provided.")

    progressive = progressive_dir(dataset)
    if progressive.is_dir() and any(progressive.iterdir()):
        return progressive
    return dataset_src_dir(dataset)
