"""
Shared Gemini helpers for the rolling-dice pipeline.

Mirrors the conventions of the form-board sft_pipeline: dotenv loading, PNG
parts as inline base64 blobs, JSON-mode generation, and a retry/backoff wrapper
around generate_content.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-3.1-pro-preview"
MAX_RETRIES = 4
RETRY_BACKOFF = 5.0


def load_dotenv() -> None:
    """Walk upward from this file loading the first .env found."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


def make_client() -> genai.Client:
    """Build a Gemini client, failing with a clear message if the key is absent."""
    load_dotenv()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit(
            "GEMINI_API_KEY is not set. Add it to .env in the project root:\n"
            "    GEMINI_API_KEY=your-key-here"
        )
    return genai.Client(api_key=key)


def png_part(path: Path) -> types.Part:
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return types.Part(inline_data=types.Blob(mime_type="image/png", data=data))


def png_part_from_bytes(raw: bytes) -> types.Part:
    data = base64.b64encode(raw).decode("ascii")
    return types.Part(inline_data=types.Blob(mime_type="image/png", data=data))


def text_part(text: str) -> types.Part:
    return types.Part(text=text)


def generate_json(
    client: genai.Client,
    model: str,
    parts: List[types.Part],
    required_fields: Optional[List[str]] = None,
    schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Call generate_content in JSON mode and return the parsed object.

    Retries on transport errors, empty responses, unparseable output, and
    missing required fields. Raises the last error if every attempt fails.
    """
    config_kwargs: Dict[str, Any] = {"response_mime_type": "application/json"}
    if schema is not None:
        config_kwargs["response_schema"] = schema

    last_err: Optional[Exception] = None
    backoff = RETRY_BACKOFF

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(**config_kwargs),
            )
            text = resp.text
            if not text:
                raise RuntimeError("empty response text")
            data = json.loads(text)
            if isinstance(data, list) and data:
                data = data[0]
            if not isinstance(data, dict):
                raise RuntimeError(f"unexpected shape: {type(data).__name__}")
            if required_fields:
                missing = [f for f in required_fields if f not in data]
                if missing:
                    raise RuntimeError(f"missing required fields: {missing}")
            return data
        except Exception as exc:  # noqa: BLE001 - retry on anything transient
            last_err = exc
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2

    raise RuntimeError(f"generate_content failed after {MAX_RETRIES} attempts: {last_err}")


def iter_puzzle_dirs(
    output_dir: Path, variant: Optional[str] = None, level: Optional[int] = None
) -> Iterator[Path]:
    """Yield every puzzle directory under output_dir, optionally filtered."""
    # "octahedron" is a sibling task rather than one of MIRA's own variants,
    # but it writes the same directory layout, so it walks the same way.
    variants = [variant] if variant else ["top", "sum", "two", "octahedron"]
    for v in variants:
        variant_dir = output_dir / v
        if not variant_dir.is_dir():
            continue
        levels = (
            [variant_dir / f"level_{level:02d}"]
            if level
            else sorted(variant_dir.glob("level_*"))
        )
        for level_dir in levels:
            if not level_dir.is_dir():
                continue
            for puzzle_dir in sorted(level_dir.glob("puzzle_*")):
                yield puzzle_dir
