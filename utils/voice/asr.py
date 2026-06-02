"""Small wrapper around SenseVoice/FunASR transcription."""

from __future__ import annotations

import contextlib
import io
import multiprocessing as mp
from pathlib import Path
import os
import queue
import re
from typing import Any


_RICH_TAG_RE = re.compile(r"<\|[^|]+?\|>")


class SenseVoiceRecognizer:
    def __init__(
        self,
        model_dir: str = "iic/SenseVoiceSmall",
        device: str | None = None,
        remote_code: str | Path | None = None,
    ) -> None:
        self.model_dir = resolve_model_dir(model_dir)
        self.device = device or os.getenv("SENSEVOICE_DEVICE", "cpu")
        self.remote_code = Path(remote_code) if remote_code else _default_remote_code()
        self._model: Any | None = None
        self._postprocess = None

    def load(self) -> None:
        if self._model is not None:
            return

        from funasr import AutoModel
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        self._postprocess = rich_transcription_postprocess
        self._model = AutoModel(
            model=self.model_dir,
            trust_remote_code=True,
            remote_code=str(self.remote_code),
            device=self.device,
        )

    def transcribe(self, audio_path: str | Path, language: str = "auto", use_itn: bool = True) -> dict[str, str]:
        self.load()
        assert self._model is not None
        assert self._postprocess is not None

        result = self._model.generate(
            input=str(audio_path),
            cache={},
            language=language,
            use_itn=use_itn,
            batch_size=1,
        )
        raw_text = result[0].get("text", "") if result else ""
        processed = self._postprocess(raw_text)
        clean_text = _RICH_TAG_RE.sub("", processed).strip()
        return {"raw_text": raw_text, "text": processed, "clean_text": clean_text}


def transcribe_with_timeout(
    audio_path: str | Path,
    *,
    model_dir: str = "iic/SenseVoiceSmall",
    device: str | None = None,
    language: str = "auto",
    timeout_s: float = 120.0,
    verbose: bool = False,
) -> dict[str, str]:
    """Run SenseVoice in a child process so model loading cannot hang the CLI."""
    output: mp.Queue = mp.Queue(maxsize=1)
    proc = mp.Process(
        target=_transcribe_worker,
        args=(str(audio_path), model_dir, device, language, verbose, output),
        daemon=True,
    )
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        raise RuntimeError(f"SenseVoice ASR timed out after {timeout_s:.0f}s")

    try:
        message = output.get_nowait()
    except queue.Empty as exc:
        raise RuntimeError(f"SenseVoice ASR exited without output, code={proc.exitcode}") from exc

    if verbose and message.get("logs"):
        print(message["logs"], end="")
    if message["ok"]:
        return message["result"]
    raise RuntimeError(message["error"])


def model_cache_status(model_dir: str) -> dict[str, Any]:
    """Return a lightweight completeness hint for ModelScope SenseVoice cache."""
    if model_dir != "iic/SenseVoiceSmall":
        return {"checked": False, "complete": None, "reason": "custom model path or id"}

    cache_dir = Path.home() / ".cache" / "modelscope" / "hub" / "models" / "iic" / "SenseVoiceSmall"
    if not cache_dir.exists():
        return {"checked": True, "complete": False, "path": str(cache_dir), "reason": "cache directory missing"}

    weight_files = [
        path
        for pattern in ("*.pt", "*.pth", "*.bin", "*.safetensors", "*.onnx")
        for path in cache_dir.rglob(pattern)
    ]
    total_weight_mb = sum(path.stat().st_size for path in weight_files) / (1024 * 1024)
    complete = bool(weight_files) and total_weight_mb > 100
    return {
        "checked": True,
        "complete": complete,
        "path": str(cache_dir),
        "weight_files": [str(path) for path in weight_files[:5]],
        "weight_mb": round(total_weight_mb, 1),
        "reason": "ok" if complete else "model weights missing or incomplete",
    }


def resolve_model_dir(model_dir: str) -> str:
    """Prefer a complete local ModelScope cache over a remote model id."""
    status = model_cache_status(model_dir)
    if status.get("checked") and status.get("complete") and status.get("path"):
        return str(status["path"])
    return model_dir


def _transcribe_worker(
    audio_path: str,
    model_dir: str,
    device: str | None,
    language: str,
    verbose: bool,
    output: mp.Queue,
) -> None:
    logs = io.StringIO()
    try:
        with contextlib.redirect_stdout(logs), contextlib.redirect_stderr(logs):
            recognizer = SenseVoiceRecognizer(model_dir=model_dir, device=device)
            result = recognizer.transcribe(audio_path, language=language)
        output.put({"ok": True, "result": result, "logs": logs.getvalue() if verbose else ""})
    except Exception as exc:
        output.put({"ok": False, "error": str(exc), "logs": logs.getvalue() if verbose else ""})


def _default_remote_code() -> Path:
    override = os.getenv("SENSEVOICE_REMOTE_CODE")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[2] / "sdk" / "SenseVoice" / "model.py"
