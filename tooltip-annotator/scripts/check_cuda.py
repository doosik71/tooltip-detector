from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the current GPU runtime for PyTorch/MONAI inference, "
            "including CUDA/cuDNN compatibility and basic runtime smoke tests."
        )
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch CUDA device used for smoke tests. Default: cuda:0",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full report as JSON in addition to the human-readable report.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "command": command,
        }

    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "command": command,
    }


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def find_torch_cuda_library() -> Path | None:
    try:
        import torch
    except ImportError:
        return None

    candidate = Path(torch.__file__).resolve().parent / "lib" / "libtorch_cuda.so"
    return candidate if candidate.exists() else None


def parse_libtorch_links() -> dict[str, str]:
    libtorch_cuda = find_torch_cuda_library()
    if libtorch_cuda is None:
        return {}

    result = run_command(["ldd", str(libtorch_cuda)])
    if not result["ok"]:
        return {}

    links: dict[str, str] = {}
    for line in result["stdout"].splitlines():
        if "=>" not in line:
            continue
        left, right = line.split("=>", 1)
        library_name = left.strip()
        target = right.strip().split("(", 1)[0].strip()
        if target:
            links[library_name] = target
    return links


def list_nvidia_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if not name:
            continue
        normalized = name.lower().replace("_", "-")
        if normalized.startswith("nvidia-"):
            versions[name] = version
    return dict(sorted(versions.items()))


def format_cudnn_version(raw_version: int | None) -> str | None:
    if raw_version is None or raw_version <= 0:
        return None
    major = raw_version // 10000
    minor = (raw_version % 10000) // 100
    patch = raw_version % 100
    return f"{major}.{minor}.{patch}"


def inspect_torch(device_name: str) -> dict[str, Any]:
    report: dict[str, Any] = {"import_ok": False}

    try:
        import torch
    except ImportError as exc:
        report["error"] = str(exc)
        return report

    report["import_ok"] = True
    report["version"] = torch.__version__
    report["torch_cuda_version"] = torch.version.cuda
    report["cuda_available"] = torch.cuda.is_available()
    report["cudnn_enabled"] = bool(torch.backends.cudnn.enabled)
    report["cudnn_version_raw"] = torch.backends.cudnn.version()
    report["cudnn_version_text"] = format_cudnn_version(report["cudnn_version_raw"])
    report["device_count"] = torch.cuda.device_count() if report["cuda_available"] else 0
    report["requested_device"] = device_name

    if report["cuda_available"]:
        try:
            device = torch.device(device_name)
            report["resolved_device"] = str(device)
            report["device_name"] = torch.cuda.get_device_name(device)
            report["device_capability"] = ".".join(
                str(part) for part in torch.cuda.get_device_capability(device)
            )
        except Exception as exc:  # pragma: no cover - environment-specific
            report["device_error"] = str(exc)

    return report


def collect_environment() -> dict[str, str]:
    keys = [
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "LD_LIBRARY_PATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "PATH",
    ]
    return {key: os.environ.get(key, "") for key in keys}


def gather_nvidia_smi() -> dict[str, Any]:
    return run_command(["nvidia-smi"])


def gather_nvcc() -> dict[str, Any]:
    return run_command(["nvcc", "--version"])


def minimal_cuda_test(device_name: str, use_cudnn: bool) -> dict[str, Any]:
    code = f"""
import json
import torch

result = {{
    "cuda_available": torch.cuda.is_available(),
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "requested_device": {device_name!r},
    "use_cudnn": {use_cudnn!r},
}}

try:
    torch.backends.cudnn.enabled = {use_cudnn!r}
    device = torch.device({device_name!r})
    x = torch.randn(1, 3, 736, 480, device=device)
    conv = torch.nn.Conv2d(3, 32, 3).to(device)
    y = conv(x)
    result["ok"] = True
    result["output_shape"] = tuple(y.shape)
except Exception as exc:
    result["ok"] = False
    result["error_type"] = type(exc).__name__
    result["error"] = str(exc)

print(json.dumps(result))
"""
    completed = run_command([sys.executable, "-c", code])
    payload: dict[str, Any] = {
        "ok": False,
        "command_ok": completed["ok"],
        "returncode": completed["returncode"],
        "stdout": completed["stdout"],
        "stderr": completed["stderr"],
    }
    if completed["stdout"]:
        try:
            payload.update(json.loads(completed["stdout"].splitlines()[-1]))
        except json.JSONDecodeError:
            pass
    return payload


def check_cudnn_alignment(torch_report: dict[str, Any], package_versions: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": True,
        "issues": [],
    }

    torch_cuda = torch_report.get("torch_cuda_version")
    cudnn_raw = torch_report.get("cudnn_version_raw")
    cudnn_text = torch_report.get("cudnn_version_text")
    package_name = None
    package_version = None

    for name, version in package_versions.items():
        if name.lower().replace("_", "-").startswith("nvidia-cudnn-cu"):
            package_name = name
            package_version = version
            break

    result["torch_cuda_version"] = torch_cuda
    result["torch_cudnn_version"] = cudnn_text
    result["package_name"] = package_name
    result["package_version"] = package_version

    if package_name is None:
        result["ok"] = False
        result["issues"].append("No nvidia-cudnn package is installed in the current Python environment.")
        return result

    if torch_cuda is not None and "cu13" in package_name.lower() and not str(torch_cuda).startswith("13"):
        result["ok"] = False
        result["issues"].append(
            f"PyTorch reports CUDA {torch_cuda}, but the installed cuDNN wheel is {package_name}."
        )

    if cudnn_raw is None:
        result["ok"] = False
        result["issues"].append("PyTorch could not report a cuDNN version.")
        return result

    if package_version is not None and cudnn_text is not None:
        package_major_minor = ".".join(package_version.split(".")[:2])
        torch_major_minor = ".".join(cudnn_text.split(".")[:2])
        if package_major_minor != torch_major_minor:
            result["ok"] = False
            result["issues"].append(
                f"PyTorch reports cuDNN {cudnn_text}, but the installed wheel version is {package_version}."
            )

    return result


def evaluate_findings(report: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    findings: list[str] = []
    actions: list[str] = []
    status = "PASS"

    torch_report = report["torch"]
    if not torch_report.get("import_ok"):
        status = "FAIL"
        findings.append(f"PyTorch import failed: {torch_report.get('error', 'unknown error')}")
        actions.append("Install a working PyTorch build before attempting GPU inference.")
        return status, findings, actions

    if not torch_report.get("cuda_available"):
        status = "FAIL"
        findings.append("PyTorch cannot see a CUDA device.")
        actions.append("Check the NVIDIA driver, container/host GPU passthrough, and CUDA-enabled PyTorch build.")
        return status, findings, actions

    cudnn_alignment = report["cudnn_alignment"]
    if not cudnn_alignment["ok"]:
        status = "FAIL"
        findings.extend(cudnn_alignment["issues"])

    cudnn_on = report["smoke_tests"]["cudnn_enabled"]
    cudnn_off = report["smoke_tests"]["cudnn_disabled"]

    if not cudnn_on.get("ok") and cudnn_off.get("ok"):
        status = "FAIL"
        findings.append(
            "CUDA works when cuDNN is disabled, but fails when cuDNN is enabled. "
            "This isolates the problem to the cuDNN runtime stack."
        )
        links = report.get("libtorch_links", {})
        nvjitlink_path = links.get("libnvJitLink.so.13")
        if nvjitlink_path:
            findings.append(f"libtorch_cuda.so currently resolves libnvJitLink.so.13 to: {nvjitlink_path}")
            if "/usr/local/cuda" in nvjitlink_path:
                findings.append(
                    "libnvJitLink is being resolved from the system CUDA installation rather than the Python environment."
                )
        actions.append(
            "Prefer the virtual-environment NVIDIA libraries over /usr/local/cuda when launching Python."
        )
        actions.append(
            "Run with LD_LIBRARY_PATH pointing first to "
            ".venv/lib/python3.12/site-packages/nvidia/cu13/lib and "
            ".venv/lib/python3.12/site-packages/nvidia/cudnn/lib."
        )
        actions.append(
            "If the mismatch persists, reinstall PyTorch and the nvidia-* CUDA/cuDNN wheels as one consistent set."
        )

    elif not cudnn_on.get("ok"):
        status = "FAIL"
        findings.append(
            f"Minimal CUDA + cuDNN smoke test failed: {cudnn_on.get('error_type', 'Error')}: {cudnn_on.get('error', '')}"
        )
        actions.append("Inspect the CUDA/cuDNN library set loaded by the current environment.")
        actions.append("Reinstall a consistent PyTorch CUDA build and matching nvidia-* wheels.")

    elif cudnn_on.get("ok"):
        findings.append("Minimal CUDA + cuDNN smoke test passed.")

    if report["nvidia_smi"]["ok"]:
        findings.append("nvidia-smi completed successfully.")
    else:
        status = "FAIL"
        findings.append("nvidia-smi failed or is unavailable.")
        actions.append("Install or repair the NVIDIA driver tools.")

    if status == "PASS":
        actions.append("Current GPU runtime appears usable for PyTorch + cuDNN inference.")

    return status, findings, actions


def build_report(device_name: str) -> dict[str, Any]:
    torch_report = inspect_torch(device_name)
    package_versions = list_nvidia_package_versions()
    report = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "environment": collect_environment(),
        "torch": torch_report,
        "nvidia_packages": package_versions,
        "cudnn_alignment": check_cudnn_alignment(torch_report, package_versions),
        "nvidia_smi": gather_nvidia_smi(),
        "nvcc": gather_nvcc(),
        "libtorch_cuda_path": str(find_torch_cuda_library()) if find_torch_cuda_library() else None,
        "libtorch_links": parse_libtorch_links(),
        "smoke_tests": {
            "cudnn_enabled": minimal_cuda_test(device_name, use_cudnn=True),
            "cudnn_disabled": minimal_cuda_test(device_name, use_cudnn=False),
        },
    }
    status, findings, actions = evaluate_findings(report)
    report["status"] = status
    report["findings"] = findings
    report["recommended_actions"] = actions
    return report


def print_report(report: dict[str, Any]) -> None:
    print(f"GPU Check Status: {report['status']}")
    print()

    print("Environment")
    print(f"- Python: {report['python_executable']} ({report['python_version']})")
    print(f"- VIRTUAL_ENV: {report['environment']['VIRTUAL_ENV'] or '<unset>'}")
    print(f"- CUDA_HOME: {report['environment']['CUDA_HOME'] or '<unset>'}")
    print(f"- LD_LIBRARY_PATH: {report['environment']['LD_LIBRARY_PATH'] or '<unset>'}")
    print()

    torch_report = report["torch"]
    print("PyTorch")
    print(f"- torch: {torch_report.get('version', '<unavailable>')}")
    print(f"- torch CUDA: {torch_report.get('torch_cuda_version', '<unavailable>')}")
    print(f"- CUDA available: {torch_report.get('cuda_available')}")
    print(f"- cuDNN version: {torch_report.get('cudnn_version_text', '<unavailable>')}")
    if "device_name" in torch_report:
        print(f"- Device: {torch_report.get('resolved_device')} ({torch_report.get('device_name')})")
    if "device_error" in torch_report:
        print(f"- Device error: {torch_report['device_error']}")
    print()

    alignment = report["cudnn_alignment"]
    print("cuDNN Alignment")
    print(f"- Installed wheel: {alignment.get('package_name') or '<not found>'}")
    print(f"- Wheel version: {alignment.get('package_version') or '<not found>'}")
    print(f"- PyTorch reported cuDNN: {alignment.get('torch_cudnn_version') or '<unavailable>'}")
    print(f"- Alignment OK: {alignment.get('ok')}")
    print()

    print("Smoke Tests")
    for label, payload in (
        ("cuDNN enabled", report["smoke_tests"]["cudnn_enabled"]),
        ("cuDNN disabled", report["smoke_tests"]["cudnn_disabled"]),
    ):
        if payload.get("ok"):
            print(f"- {label}: PASS")
        else:
            print(
                f"- {label}: FAIL: "
                f"{payload.get('error_type', 'Error')}: {payload.get('error', '<no message>')}"
            )
    print()

    links = report.get("libtorch_links", {})
    if links:
        print("Linked CUDA Libraries")
        for name in ("libcudnn.so.9", "libcudart.so.13", "libcublas.so.13", "libnvJitLink.so.13"):
            if name in links:
                print(f"- {name}: {links[name]}")
        print()

    print("Findings")
    for finding in report["findings"]:
        print(f"- {finding}")
    print()

    print("Recommended Actions")
    for action in report["recommended_actions"]:
        print(f"- {action}")
    print()

    if report["nvidia_smi"]["ok"]:
        first_line = report["nvidia_smi"]["stdout"].splitlines()[0] if report["nvidia_smi"]["stdout"] else ""
        print("nvidia-smi")
        print(f"- {first_line}")
        print()
    else:
        print("nvidia-smi")
        print(f"- Failed: {report['nvidia_smi']['stderr'] or report['nvidia_smi']['stdout']}")
        print()

    if report["nvcc"]["ok"]:
        nvcc_lines = report["nvcc"]["stdout"].splitlines()
        if nvcc_lines:
            print("nvcc")
            for line in nvcc_lines[-2:]:
                print(f"- {line}")
            print()


def main() -> int:
    args = parse_args()
    report = build_report(args.device)
    print_report(report)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))

    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
