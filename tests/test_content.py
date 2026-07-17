from __future__ import annotations

import base64
import io
import wave

import numpy as np

from rolloutlib import content


def _decode_data_url(url: str) -> bytes:
    return base64.b64decode(url.split(",", maxsplit=1)[1])


def test_content_helpers_build_text_and_json_parts() -> None:
    assert content.text("hello") == {"type": "text", "text": "hello"}
    assert content.json({"turn": "white"}) == {
        "type": "text",
        "text": '{"turn":"white"}',
    }


def test_image_helper_encodes_numpy_arrays_as_png_data_urls() -> None:
    pixels = np.zeros((2, 3, 3), dtype=np.uint8)

    part = content.image(pixels, alt="board")

    assert part["type"] == "image"
    assert part.get("alt") == "board"
    assert part["url"].startswith("data:image/png;base64,")
    assert _decode_data_url(part["url"]).startswith(b"\x89PNG\r\n\x1a\n")


def test_audio_helper_encodes_numpy_arrays_as_wav_data_urls() -> None:
    samples = np.array([0.0, 0.5, -0.5], dtype=np.float32)

    part = content.audio(samples, sample_rate=16_000)

    assert part["type"] == "audio"
    assert part.get("format") == "wav"
    assert part["url"].startswith("data:audio/wav;base64,")
    with wave.open(io.BytesIO(_decode_data_url(part["url"])), "rb") as wav:
        assert wav.getframerate() == 16_000
        assert wav.getnchannels() == 1
        assert wav.getnframes() == 3
