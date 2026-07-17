"""Construct backend-neutral text, image, and audio message content."""

from __future__ import annotations

import base64
import importlib
import io
import json as json_module
import wave
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from .types import (
    AudioContentPart,
    ImageContentPart,
    TextContentPart,
)


def data_url(data: bytes, media_type: str) -> str:
    """Encode bytes as a data URL."""

    if not media_type:
        raise ValueError("media_type must not be empty")
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def text(value: object) -> TextContentPart:
    """Construct a text content part."""

    return {"type": "text", "text": str(value)}


def json(
    value: Mapping[str, Any] | Sequence[Any],
    *,
    indent: int | None = None,
) -> TextContentPart:
    """Serialize a structured value into a text content part."""

    return {
        "type": "text",
        "text": json_module.dumps(
            value,
            ensure_ascii=False,
            separators=None if indent is not None else (",", ":"),
            indent=indent,
        ),
    }


def image(
    value: str | bytes | np.ndarray[Any, Any] | Any,
    *,
    format: str = "png",
    alt: str | None = None,
) -> ImageContentPart:
    """Construct an image content part from a URL, bytes, or image array.

    Array and image-object encoding uses Pillow, which is available through the
    ``media`` and ``openai`` optional dependency extras.
    """

    normalized_format = format.lower()
    if isinstance(value, str):
        url = value
    elif isinstance(value, bytes):
        url = data_url(value, f"image/{normalized_format}")
    else:
        try:
            pil_image = importlib.import_module("PIL.Image")
        except ImportError as error:
            raise ImportError(
                "encoding image arrays requires Pillow; install rolloutlib[media]"
            ) from error
        resolved = (
            pil_image.fromarray(value)
            if isinstance(value, np.ndarray)
            else value
        )
        output = io.BytesIO()
        resolved.save(output, format=normalized_format.upper())
        url = data_url(output.getvalue(), f"image/{normalized_format}")
    part: ImageContentPart = {"type": "image", "url": url}
    if alt is not None:
        part["alt"] = alt
    return part


def audio(
    value: str | bytes | np.ndarray[Any, Any],
    *,
    format: str = "wav",
    sample_rate: int | None = None,
) -> AudioContentPart:
    """Construct an audio content part from a URL, bytes, or PCM samples.

    NumPy arrays are encoded as 16-bit WAV and therefore require
    ``sample_rate``. Floating-point samples are clipped to ``[-1, 1]``.
    """

    normalized_format = format.lower()
    if isinstance(value, str):
        url = value
    elif isinstance(value, bytes):
        url = data_url(value, f"audio/{normalized_format}")
    else:
        if normalized_format != "wav":
            raise ValueError("NumPy audio arrays currently support only WAV")
        if sample_rate is None or sample_rate <= 0:
            raise ValueError("sample_rate must be positive for audio arrays")
        samples = np.asarray(value)
        if samples.ndim not in (1, 2):
            raise ValueError("audio arrays must have shape (frames,) or (frames, channels)")
        if np.issubdtype(samples.dtype, np.floating):
            pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
        elif samples.dtype == np.int16:
            pcm = samples.astype("<i2", copy=False)
        else:
            raise ValueError("audio arrays must contain float or int16 samples")
        channels = 1 if pcm.ndim == 1 else pcm.shape[1]
        output = io.BytesIO()
        with wave.open(cast(Any, output), "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes())
        url = data_url(output.getvalue(), "audio/wav")
    return {"type": "audio", "url": url, "format": normalized_format}


__all__ = ["audio", "data_url", "image", "json", "text"]
