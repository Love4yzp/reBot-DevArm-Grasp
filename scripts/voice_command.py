#!/usr/bin/env python3
"""Record/transcribe a short command and print a semantic JSON candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.voice.asr import model_cache_status, transcribe_with_timeout
from utils.voice.config import load_voice_config
from utils.voice.recorder import record_wav
from utils.voice.semantic import parse_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SenseVoice command parser for robot grasping.")
    parser.add_argument("--voice-config", default=None, help="Voice YAML config path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_voice_config(_resolve_project_path(args.voice_config) if args.voice_config else None)
    input_cfg = cfg["input"]
    rec_cfg = cfg["recording"]
    asr_cfg = cfg["asr"]
    out_cfg = cfg["output"]
    mode = str(input_cfg.get("mode", "record")).lower()
    seconds = float(rec_cfg["seconds"])
    language = str(asr_cfg["language"])
    model_dir = str(asr_cfg["model_dir"])
    device = asr_cfg.get("device")
    asr_timeout = float(asr_cfg["timeout_s"])
    allow_incomplete = bool(asr_cfg.get("allow_incomplete_model", False))
    verbose = bool(asr_cfg.get("verbose", False))
    output_json = bool(out_cfg.get("json", False))
    compact_json = bool(out_cfg.get("compact", False))
    temp_audio: Path | None = None

    try:
        if mode == "text":
            text = str(input_cfg.get("text") or "")
            transcript = {"raw_text": text, "text": text, "clean_text": text}
        else:
            audio_value = input_cfg.get("audio")
            audio_path = Path(str(audio_value)).expanduser() if mode == "audio" and audio_value else None
            if mode == "record":
                temp = tempfile.NamedTemporaryFile(prefix="voice_command_", suffix=".wav", delete=False)
                temp.close()
                temp_audio = Path(temp.name)
                if rec_cfg.get("countdown", True):
                    _countdown()
                _status(f"[REC] start {seconds:.0f}s")
                audio_path = record_wav(temp_audio, seconds=seconds)
                _status("[REC] end")
                if input_cfg.get("record_only", False):
                    _clear_status()
                    print(str(audio_path))
                    temp_audio = None
                    return 0
            if audio_path is None:
                raise ValueError("voice input.mode must be record, text, or audio with input.audio set")

            cache_status = model_cache_status(model_dir)
            if cache_status.get("checked") and not cache_status.get("complete") and not allow_incomplete:
                raise RuntimeError(
                    "SenseVoice model cache looks incomplete: "
                    + json.dumps(cache_status, ensure_ascii=False)
                    + ". Finish downloading the model, or set asr.allow_incomplete_model: true in voice.yaml."
                )

            _status("[ASR] running")
            transcript = transcribe_with_timeout(
                audio_path,
                model_dir=model_dir,
                device=device,
                language=language,
                timeout_s=asr_timeout,
                verbose=verbose,
            )
            _status("[ASR] done")

        intent = parse_command(transcript["clean_text"])
        _clear_status()
        if output_json:
            payload = intent.to_dict()
            payload["transcript"] = transcript
            json.dump(payload, sys.stdout, ensure_ascii=False, indent=None if compact_json else 2)
            sys.stdout.write("\n")
        else:
            print(transcript["text"])
        return 0
    finally:
        if temp_audio is not None:
            try:
                temp_audio.unlink(missing_ok=True)
            except OSError:
                pass


def _countdown() -> None:
    for value in (3, 2, 1):
        _status(f"[REC] {value}")
        time.sleep(1)


def _status(message: str) -> None:
    if sys.stderr.isatty():
        print(f"\r{message:<48}", end="", file=sys.stderr, flush=True)
    else:
        print(message, file=sys.stderr, flush=True)


def _clear_status() -> None:
    if sys.stderr.isatty():
        print("\r" + " " * 80 + "\r", end="", file=sys.stderr, flush=True)


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
