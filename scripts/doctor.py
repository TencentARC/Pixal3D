#!/usr/bin/env python3
"""Inspect whether this Mac is ready for quality-first TRELLIS.2 inference."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
from typing import Any


METAL_MODULES = ("flex_gemm", "cumesh", "mtldiffrast", "o_voxel")


def _command(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _module_status(name: str) -> dict[str, str | bool]:
    try:
        module = importlib.import_module(name)
        return {"available": True, "version": str(getattr(module, "__version__", "unknown"))}
    except Exception as exc:  # An incompatible metallib should be reported too.
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def collect_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "macos": _command("sw_vers", "-productVersion"),
        "metal_sdk": _command("xcrun", "--sdk", "macosx", "--show-sdk-version"),
        "modules": {name: _module_status(name) for name in METAL_MODULES},
    }
    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
        }
    except Exception as exc:
        report["torch"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print compact JSON only.")
    parser.add_argument("--require-metal", action="store_true", help="Exit non-zero unless all Metal modules work.")
    args = parser.parse_args()

    report = collect_report()
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))

    if args.require_metal:
        torch_ok = bool(report.get("torch", {}).get("mps_available"))
        modules_ok = all(bool(report["modules"][name].get("available")) for name in METAL_MODULES)
        if not torch_ok or not modules_ok:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
