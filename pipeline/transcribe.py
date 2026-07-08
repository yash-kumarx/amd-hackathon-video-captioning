"""Audio transcription via gemini-2.5-flash (OpenAI-compat input_audio).

Fireworks deprecated all audio inference on 2026-06-10 (Whisper endpoints return 401),
so RESEARCH.md's Whisper plan is dead — Gemini flash free tier replaces it. Its quota
is per-model, so this does not compete with Gemma's quota.
"""
import base64
import logging
from typing import Optional

import httpx

from . import config
from .util import chat_completion, message_content

log = logging.getLogger("pipeline.transcribe")

PROMPT = (
    "Transcribe this audio verbatim in its original language. "
    "If there is no intelligible speech, reply with exactly NO_SPEECH followed by a 3-6 word "
    "description of what the audio contains (e.g. 'NO_SPEECH ambient street noise, distant traffic')."
)


async def transcribe(client: httpx.AsyncClient, wav_path: str) -> Optional[str]:
    """Returns transcript text, 'NO_SPEECH …' marker, or None on failure."""
    try:
        with open(wav_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        payload = {
            "model": config.AUDIO_MODEL,
            "max_tokens": 1200,
            "temperature": 0.0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
                ],
            }],
        }
        resp = await chat_completion(
            client, config.GEMINI_OPENAI_BASE, config.GEMINI_API_KEY,
            payload, timeout=config.AUDIO_TIMEOUT, retries=1, provider="gemini",
        )
        text = message_content(resp).strip()
        return text or None
    except Exception as e:  # audio is a booster, never sink the clip
        log.warning("transcription failed: %s", str(e)[:150])
        return None
