"""Microphone recording utilities."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import subprocess


def record_wav(path: str | Path, seconds: float = 3.0, sample_rate: int = 16000) -> Path:
    """Record a mono 16-bit PCM wav file with arecord."""
    arecord = shutil.which("arecord")
    if arecord is None:
        raise RuntimeError("arecord was not found; install alsa-utils or pass --audio/--text instead.")

    output = Path(path)
    duration = max(1, int(round(seconds)))
    arecord_cmd = [
        arecord,
        "-f",
        "S16_LE",
        "-r",
        str(sample_rate),
        "-c",
        "1",
        "-d",
        str(duration),
        str(output),
    ]

    timeout_bin = shutil.which("timeout")
    if timeout_bin:
        timeout_seconds = duration + 2
        cmd = [
            timeout_bin,
            "--signal=INT",
            "--kill-after=2",
            str(timeout_seconds),
            *arecord_cmd,
        ]
    else:
        cmd = arecord_cmd

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=duration + 5)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(proc)
        stdout, stderr = proc.communicate()
        raise RuntimeError(f"recording did not stop after {duration}s") from exc

    if proc.returncode != 0:
        message = (stderr or stdout or "").strip()
        raise RuntimeError(f"arecord failed with code {proc.returncode}: {message}")

    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"recording produced no audio: {output}")

    return output


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=2)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
