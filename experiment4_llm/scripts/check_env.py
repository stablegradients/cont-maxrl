#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import sys
from dataclasses import asdict, dataclass


REQUIRED_MODULES = [
    "torch",
    "numpy",
    "scipy",
    "matplotlib",
    "yaml",  # PyYAML
    "datasets",
    "transformers",
    "accelerate",
]

OPTIONAL_MODULES = [
    "verl",
    "vllm",
    "pytest",
]


@dataclass
class ModuleStatus:
    name: str
    required: bool
    installed: bool
    version: str | None
    error: str | None


def probe_module(name: str, required: bool) -> ModuleStatus:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", None)
        if version is None and name == "yaml":
            try:
                import importlib.metadata as metadata

                version = metadata.version("PyYAML")
            except Exception:
                version = None
        return ModuleStatus(
            name=name,
            required=required,
            installed=True,
            version=version,
            error=None,
        )
    except Exception as exc:
        return ModuleStatus(
            name=name,
            required=required,
            installed=False,
            version=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def gather_status() -> dict[str, object]:
    rows: list[ModuleStatus] = []
    for name in REQUIRED_MODULES:
        rows.append(probe_module(name, required=True))
    for name in OPTIONAL_MODULES:
        rows.append(probe_module(name, required=False))

    torch_cuda_available = None
    torch_cuda_device_count = None
    if any(r.name == "torch" and r.installed for r in rows):
        import torch

        torch_cuda_available = bool(torch.cuda.is_available())
        torch_cuda_device_count = int(torch.cuda.device_count())

    return {
        "python": {
            "version": sys.version.split("\n")[0],
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "tools": {
            "docker": shutil.which("docker") is not None,
        },
        "modules": [asdict(r) for r in rows],
        "torch": {
            "cuda_available": torch_cuda_available,
            "cuda_device_count": torch_cuda_device_count,
        },
    }


def print_report(report: dict[str, object]) -> int:
    py = report["python"]
    print(f"Python: {py['version']}")
    print(f"Executable: {py['executable']}")
    print(f"Platform: {py['platform']}")
    print(f"Docker found: {report['tools']['docker']}")

    print("\nModule checks:")
    missing_required = []
    for row in report["modules"]:
        required = row["required"]
        prefix = "REQ" if required else "OPT"
        if row["installed"]:
            version = row["version"] or "unknown"
            print(f"  [{prefix}] OK      {row['name']} ({version})")
        else:
            print(f"  [{prefix}] MISSING {row['name']} ({row['error']})")
            if required:
                missing_required.append(row["name"])

    torch_info = report["torch"]
    if torch_info["cuda_available"] is not None:
        print(
            f"\nTorch CUDA: available={torch_info['cuda_available']} "
            f"devices={torch_info['cuda_device_count']}"
        )

    if missing_required:
        print("\nMissing required packages:")
        print("  " + " ".join(missing_required))
        pip_names = ["pyyaml" if n == "yaml" else n for n in missing_required]
        print("Install command:")
        print("  python -m pip install " + " ".join(pip_names))
        return 1

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Experiment 4 Python environment")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON report in addition to text",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Return exit code 0 even if required modules are missing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = gather_status()
    code = print_report(report)

    if args.json:
        print("\nJSON report:")
        print(json.dumps(report, indent=2))

    if code != 0 and not args.allow_missing:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
