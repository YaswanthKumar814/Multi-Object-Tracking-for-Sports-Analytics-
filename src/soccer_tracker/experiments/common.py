"""Small helpers shared by the experiment stages: result files and run metadata."""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable


def write_csv(path: Path, rows: Iterable[dict], columns: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        for r in rows:
            for k in r:
                if k not in columns:
                    columns.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})


def read_csv(path: Path) -> list[dict]:
    """Read a CSV written by :func:`write_csv`, converting numeric strings back to numbers."""
    def conv(v: str) -> Any:
        if v == "":
            return None
        if v in ("True", "False"):
            return v == "True"
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v

    with Path(path).open(encoding="utf-8", newline="") as fh:
        return [{k: conv(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_default)


def read_json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(o)


def _cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def system_info(device: str | None = None) -> dict:
    """Hardware and software the experiments ran on (recorded with every result set)."""
    info: dict[str, Any] = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "cpu": _cpu_name(),
        "cpu_logical_cores": os.cpu_count(),
        "python": platform.python_version(),
        "device": device or "auto",
    }
    try:
        import psutil

        info["cpu_physical_cores"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        # Laptops throttle the CPU on battery, which changes every timing by a large factor.
        battery = psutil.sensors_battery()
        info["power_source"] = ("no battery (desktop)" if battery is None
                                else "AC power" if battery.power_plugged else "battery (CPU may be throttled)")
        freq = psutil.cpu_freq()
        info["cpu_mhz_current"] = round(freq.current) if freq else None
    except ImportError:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    for mod in ("ultralytics", "trackers", "supervision", "cv2", "numpy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip() or None
    except Exception:
        info["git_commit"] = None
    return info
