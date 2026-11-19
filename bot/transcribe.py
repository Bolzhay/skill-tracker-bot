"""Speech-to-text for voice/audio/video proofs via local faster-whisper:
free, and offline once the model is cached (the first run downloads it)."""

import asyncio
import logging
import os
import tempfile
import threading

import numpy as np

from . import config

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless on Windows, just noisy

_SAMPLE_RATE = 16000  # what Whisper expects
_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel  # heavy import, only when needed

        logging.info("Loading Whisper model %r (the first run downloads it)", config.WHISPER_MODEL)
        _model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    return _model


def warmup() -> None:
    with _lock:
        _get_model()


def decode_audio(path: str) -> np.ndarray:
    """Any audio/video container -> 16 kHz mono float32, via the FFmpeg that
    ships inside PyAV (so ogg/opus voice notes and mp4 videos both work).

    Done here instead of by faster-whisper: its own decoder passes
    `metadata_errors=` to av.open, which the newest PyAV rejects, and that
    broke every transcription. Raises if the file has no audio track."""
    import av

    chunks = []
    with av.open(path) as container:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=_SAMPLE_RATE)
        for frame in container.decode(audio=0):
            chunks.extend(piece.to_ndarray().reshape(-1) for piece in resampler.resample(frame))
        chunks.extend(piece.to_ndarray().reshape(-1) for piece in resampler.resample(None))  # flush

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def transcribe_file(path: str) -> str:
    audio = decode_audio(path)
    if audio.size == 0:
        return ""

    # One transcription at a time: the model isn't meant for concurrent calls.
    with _lock:
        segments, _info = _get_model().transcribe(audio, vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments).strip()


async def transcribe_bytes(data: bytes, suffix: str = ".ogg") -> str:
    """Runs in a thread — decoding and Whisper are CPU-heavy."""

    def run() -> str:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
        try:
            return transcribe_file(tmp.name)
        finally:
            os.unlink(tmp.name)

    return await asyncio.to_thread(run)
