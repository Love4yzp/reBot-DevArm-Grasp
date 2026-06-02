"""Configuration loading for voice command scripts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_VOICE_CONFIG = Path(__file__).resolve().parents[2] / "config" / "voice.yaml"


def load_voice_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path).expanduser() if path else DEFAULT_VOICE_CONFIG
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _merge_defaults(data)


def _merge_defaults(data: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "input": {"mode": "record", "text": None, "audio": None, "record_only": False},
        "recording": {"seconds": 3.0, "countdown": True, "temp_audio": None},
        "asr": {
            "language": "zh",
            "model_dir": "iic/SenseVoiceSmall",
            "device": None,
            "timeout_s": 120.0,
            "allow_incomplete_model": False,
            "verbose": False,
        },
        "output": {"json": False, "compact": False},
        "grasp": {"config": "config/default.yaml", "execute": False},
    }
    merged = deepcopy(defaults)
    for section, values in data.items():
        if isinstance(values, dict) and isinstance(merged.get(section), dict):
            merged[section].update(values)
        else:
            merged[section] = values
    return merged
