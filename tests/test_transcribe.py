import math
import os
import struct
import wave
from types import SimpleNamespace

import av
import numpy as np
import pytest

from bot import transcribe


def _sine_wav(path, seconds=1.0, rate=22050, channels=2, freq=440):
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * n / rate))) * channels
            for n in range(int(rate * seconds))
        )
        out.writeframes(frames)


def _reencode(src, dst, fmt, codec, rate):
    """Re-encode the wav the way Telegram delivers voice notes / videos."""
    with av.open(str(src)) as inp, av.open(str(dst), "w", format=fmt) as out:
        stream = out.add_stream(codec, rate=rate)
        stream.layout = "mono"
        resampler = av.AudioResampler(format=stream.format.name, layout="mono", rate=rate)
        for frame in inp.decode(audio=0):
            for piece in resampler.resample(frame):
                out.mux(stream.encode(piece))
        out.mux(stream.encode(None))


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "tone.wav"
    _sine_wav(path)
    return path


def _assert_one_second_of_whisper_input(audio):
    assert audio.dtype == np.float32
    assert abs(audio.size - transcribe._SAMPLE_RATE) < 0.05 * transcribe._SAMPLE_RATE  # ~1 s at 16 kHz
    assert 0.2 < np.abs(audio).max() <= 1.0  # the tone survived and is normalised


def test_wav_is_decoded_to_16khz_mono_float(wav):
    _assert_one_second_of_whisper_input(transcribe.decode_audio(str(wav)))


@pytest.mark.parametrize(
    "name, fmt, codec, rate",
    [("voice.ogg", "ogg", "libopus", 48000), ("clip.mp4", "mp4", "aac", 44100)],
)
def test_telegram_voice_and_video_containers_are_decoded(wav, tmp_path, name, fmt, codec, rate):
    target = tmp_path / name
    _reencode(wav, target, fmt, codec, rate)
    _assert_one_second_of_whisper_input(transcribe.decode_audio(str(target)))


def test_a_file_that_is_not_audio_raises(tmp_path):
    junk = tmp_path / "junk.ogg"
    junk.write_bytes(b"this is not media")
    with pytest.raises(Exception):
        transcribe.decode_audio(str(junk))


class FakeModel:
    def __init__(self, texts):
        self.texts = texts
        self.seen = []

    def transcribe(self, audio, vad_filter):
        self.seen.append((audio, vad_filter))
        return iter(SimpleNamespace(text=text) for text in self.texts), None


def test_segments_are_joined_into_one_transcript(wav, monkeypatch):
    model = FakeModel([" Привет, ", "мир. "])
    monkeypatch.setattr(transcribe, "_get_model", lambda: model)

    assert transcribe.transcribe_file(str(wav)) == "Привет, мир."
    audio, vad = model.seen[0]
    assert isinstance(audio, np.ndarray) and vad is True  # speech-only, so silence isn't "heard"


def test_silence_with_no_segments_gives_an_empty_transcript(wav, monkeypatch):
    monkeypatch.setattr(transcribe, "_get_model", lambda: FakeModel([]))
    assert transcribe.transcribe_file(str(wav)) == ""


async def test_bytes_go_through_a_temp_file_that_is_cleaned_up(monkeypatch):
    seen = {}

    def fake_transcribe_file(path):
        seen["path"] = path
        seen["content"] = open(path, "rb").read()
        seen["suffix"] = os.path.splitext(path)[1]
        return "текст"

    monkeypatch.setattr(transcribe, "transcribe_file", fake_transcribe_file)
    assert await transcribe.transcribe_bytes(b"RAW", ".ogg") == "текст"

    assert seen["content"] == b"RAW" and seen["suffix"] == ".ogg"
    assert not os.path.exists(seen["path"])


async def test_the_temp_file_is_removed_even_if_transcription_fails(monkeypatch):
    seen = {}

    def boom(path):
        seen["path"] = path
        raise RuntimeError("decoder failed")

    monkeypatch.setattr(transcribe, "transcribe_file", boom)
    with pytest.raises(RuntimeError):
        await transcribe.transcribe_bytes(b"RAW", ".mp4")
    assert not os.path.exists(seen["path"])
