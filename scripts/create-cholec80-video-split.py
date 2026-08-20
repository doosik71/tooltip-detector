#!/usr/bin/env python3
"""Expose Cholec80 as a video-disjoint symlink dataset.

The source dataset's train/val/test directories are frame-disjoint but every
video occurs in every split. This script regroups those existing files without
copying or regenerating labels: videos 01--32 become train, 33--40 validation,
and 41--80 test.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

SPLITS = ("train", "val", "test")
KINDS = ("images", "segmentation", "annotation")


def target_split(filename: str) -> str:
    stem = Path(filename).stem
    prefix, _, _ = stem.partition("_")
    if not prefix.startswith("video") or not prefix[5:].isdigit():
        raise ValueError(f"Expected videoNN_frame filename, got {filename!r}")
    number = int(prefix[5:])
    if 1 <= number <= 32:
        return "train"
    if 33 <= number <= 40:
        return "val"
    if 41 <= number <= 80:
        return "test"
    raise ValueError(f"Video number outside 01--80: {filename!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/dataset"))
    parser.add_argument("--source", default="cholec80")
    parser.add_argument("--output", default="cholec80-vs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = args.data_root / args.source
    output = args.data_root / args.output
    if not source.is_dir():
        parser.error(f"Source dataset does not exist: {source}")
    if output.exists():
        parser.error(f"Refusing to overwrite existing output dataset: {output}")

    planned: dict[str, Counter[str]] = {kind: Counter() for kind in KINDS}
    names: dict[tuple[str, str], set[str]] = {(kind, split): set() for kind in KINDS for split in SPLITS}
    for kind in KINDS:
        for old_split in SPLITS:
            directory = source / kind / old_split
            if not directory.is_dir():
                parser.error(f"Missing source directory: {directory}")
            for item in directory.iterdir():
                if not item.is_file():
                    continue
                split = target_split(item.name)
                if item.name in names[(kind, split)]:
                    parser.error(f"Duplicate destination filename: {kind}/{split}/{item.name}")
                names[(kind, split)].add(item.name)
                planned[kind][split] += 1

    for split in SPLITS:
        counts = {kind: planned[kind][split] for kind in KINDS}
        if len(set(counts.values())) != 1:
            parser.error(f"Modality count mismatch for {split}: {counts}")
    if args.dry_run:
        print(json.dumps({kind: dict(counts) for kind, counts in planned.items()}, indent=2))
        return

    for kind in KINDS:
        for old_split in SPLITS:
            for item in (source / kind / old_split).iterdir():
                if not item.is_file():
                    continue
                split = target_split(item.name)
                destination = output / kind / split / item.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(item.resolve())

    manifest = {
        "source": str(source), "split_rule": "video01-32 train, video33-40 val, video41-80 test",
        "counts": {kind: dict(counts) for kind, counts in planned.items()},
    }
    (output / "split-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Created video-disjoint dataset at {output}")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
