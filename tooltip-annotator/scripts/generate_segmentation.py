from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from tooltip.dataset_paths import images_dir, resolve_path, segmentation_dir

try:
    import cv2
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None

try:
    import monai
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    monai = None
    MONAI_IMPORT_ERROR = exc
else:
    MONAI_IMPORT_ERROR = None

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    np = None
    NUMPY_IMPORT_ERROR = exc
else:
    NUMPY_IMPORT_ERROR = None

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


SPLIT_NAMES = ("train", "val", "test")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MODEL_INPUT_WIDTH = 736
MODEL_INPUT_HEIGHT = 480
MASK_SUFFIX = ".png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate binary semantic-segmentation masks for dataset images "
            "using the downloaded MONAI model."
        )
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Dataset name, e.g. 'erop' or 'cholec80'. Used to resolve default "
            "--input/--output when not given explicitly."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Root directory containing train/val/test image folders. Defaults to data/dataset/<dataset>/images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Root directory where train/val/test mask folders will be saved. Defaults to data/dataset/<dataset>/segmentation.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("./temp/models/model.pt"),
        help="Path to the downloaded MONAI checkpoint.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to use, for example cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output masks instead of skipping them.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if CV2_IMPORT_ERROR is not None:
        raise RuntimeError(
            "opencv-python is required to read and write images. "
            "Install it and run the script again."
        ) from CV2_IMPORT_ERROR
    if MONAI_IMPORT_ERROR is not None:
        raise RuntimeError(
            "monai is required to restore the segmentation network. "
            "Install it and run the script again."
        ) from MONAI_IMPORT_ERROR
    if NUMPY_IMPORT_ERROR is not None:
        raise RuntimeError(
            "numpy is required to prepare image tensors. "
            "Install it and run the script again."
        ) from NUMPY_IMPORT_ERROR
    if TORCH_IMPORT_ERROR is not None:
        raise RuntimeError(
            "torch is required to run inference. "
            "Install it and run the script again."
        ) from TORCH_IMPORT_ERROR

    if not args.input.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input}")
    if not args.input.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {args.input}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {args.model}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model checkpoint is not a file: {args.model}")

    for split_name in SPLIT_NAMES:
        split_dir = args.input / split_name
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
        if not split_dir.is_dir():
            raise NotADirectoryError(f"Split path is not a directory: {split_dir}")


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def ensure_output_dirs(output_root: Path) -> None:
    for split_name in SPLIT_NAMES:
        (output_root / split_name).mkdir(parents=True, exist_ok=True)


def _all_outputs_exist(input_root: Path, output_root: Path) -> bool:
    for split_name in SPLIT_NAMES:
        for image_path in list_images(input_root / split_name):
            if not (output_root / split_name / f"{image_path.stem}{MASK_SUFFIX}").exists():
                return False
    return True


def build_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    model = monai.networks.nets.FlexibleUNet(
        in_channels=3,
        out_channels=2,
        backbone="efficientnet-b2",
        pretrained=False,
        spatial_dims=2,
        is_pad=False,
        pre_conv=None,
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def preprocess_image(image_path: Path) -> tuple[np.ndarray, torch.Tensor]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        image_rgb,
        (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
    return image_rgb, tensor


def postprocess_mask(logits: torch.Tensor, output_size: tuple[int, int]) -> np.ndarray:
    prediction = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    mask = prediction * 255
    restored = cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)
    return restored


def infer_image(
    image_path: Path,
    output_path: Path,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    image_rgb, tensor = preprocess_image(image_path)
    original_height, original_width = image_rgb.shape[:2]

    with torch.inference_mode():
        logits = model(tensor.to(device))

    mask = postprocess_mask(logits, (original_width, original_height))
    if not cv2.imwrite(str(output_path), mask):
        raise RuntimeError(f"Failed to write segmentation mask: {output_path}")


def process_split(
    split_name: str,
    input_root: Path,
    output_root: Path,
    model: torch.nn.Module,
    device: torch.device,
    overwrite: bool,
) -> tuple[int, int]:
    input_dir = input_root / split_name
    output_dir = output_root / split_name
    image_paths = list_images(input_dir)

    saved_count = 0
    skipped_count = 0

    for image_path in tqdm(image_paths, desc=split_name, unit="image", ascii=True, ncols=100):
        output_path = output_dir / f"{image_path.stem}{MASK_SUFFIX}"
        if output_path.exists() and not overwrite:
            skipped_count += 1
            continue

        infer_image(
            image_path=image_path,
            output_path=output_path,
            model=model,
            device=device,
        )
        saved_count += 1

    return saved_count, skipped_count


def main() -> int:
    args = parse_args()
    args.input = resolve_path(args.input, args.dataset, images_dir, "--input")
    args.output = resolve_path(args.output, args.dataset, segmentation_dir, "--output")
    validate_args(args)

    device = resolve_device(args.device)
    ensure_output_dirs(args.output)

    if not args.overwrite and _all_outputs_exist(args.input, args.output):
        print("All segmentation masks already exist. Nothing to do.")
        return 0

    model = build_model(args.model, device)

    totals = {
        split_name: {"saved": 0, "skipped": 0}
        for split_name in SPLIT_NAMES
    }

    for split_name in SPLIT_NAMES:
        saved_count, skipped_count = process_split(
            split_name=split_name,
            input_root=args.input,
            output_root=args.output,
            model=model,
            device=device,
            overwrite=args.overwrite,
        )
        totals[split_name]["saved"] = saved_count
        totals[split_name]["skipped"] = skipped_count

    print("Segmentation generation complete.")
    for split_name in SPLIT_NAMES:
        print(
            f"{split_name}: "
            f"saved={totals[split_name]['saved']}, "
            f"skipped={totals[split_name]['skipped']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
