"""Stage 2: grounded-facts JSON from keyframes (+transcript +OCR) via Kimi K2.6 on Fireworks.

RESEARCH.md specified Qwen3-VL, but no Qwen-VL model exists on Fireworks serverless as of
2026-07-08 (verified by probe). kimi-k2p6 is the strongest serverless VLM available;
reasoning_effort="none" is required to keep its thinking out of content.
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional

import httpx

from . import config
from .schemas import GroundedFacts
from .util import chat_completion, extract_json, message_content

log = logging.getLogger("pipeline.grounding")

SYSTEM = (
    "You are a meticulous visual analyst. You are given ordered keyframes sampled across a "
    "{duration:.0f}-second video, plus an audio transcript and OCR text detected on-screen (when available). "
    "Report ONLY what is visually or audibly verifiable. Never invent brand names, numbers, or dialogue. "
    "Describe the depicted scene itself — do NOT comment on encoding, playback speed, filters, "
    "camera pans, or missing audio (such observations belong in uncertainty_notes if essential). "
    "If unsure, put it in uncertainty_notes. For ambiguous close-ups prefer the most common everyday "
    "interpretation (e.g. keyboard keys, not staples) and list alternatives in uncertainty_notes. "
    "Only state a mood if it is plainly evident on screen; otherwise leave mood empty. "
    "MOTION: compare consecutive keyframes before writing actions — note what moves and how "
    "(subject position/posture changes, approaching or receding, camera motion). Never call a "
    "subject stationary unless it truly holds the same pose and position across the frames. "
    "Be concrete: colors, counts, actions, location cues, motion over time. "
    "Be terse. Output STRICT JSON only, no markdown fences, exactly this shape: "
    '{{"subjects": [], "actions": [], "setting": "", "on_screen_text": [], "mood": "", '
    '"audio_summary": "", "temporal_arc": "", "salient_objects": [], "uncertainty_notes": []}}'
)


def _user_content(frames: List[Dict], transcript: Optional[str], ocr_text: List[str]) -> List[Dict]:
    parts: List[Dict] = []
    lines = [f"Keyframes (in order, timestamps in seconds): {', '.join(str(f['t']) for f in frames)}"]
    if transcript:
        lines.append(f"AUDIO TRANSCRIPT: {transcript[:2000]}")
    else:
        lines.append("AUDIO TRANSCRIPT: (none — video has no audio track or no speech)")
    if ocr_text:
        lines.append("OCR ON-SCREEN TEXT (verified locally): " + " | ".join(ocr_text))
    lines.append("Analyze the video and return the strict JSON now.")
    parts.append({"type": "text", "text": "\n".join(lines)})
    for f in frames:
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f['b64']}"}})
    return parts


async def _ground_once(client: httpx.AsyncClient, payload_base: dict, model: str,
                       timeout: float) -> Optional[GroundedFacts]:
    payload = dict(payload_base, model=model)
    resp = await chat_completion(
        client, config.FIREWORKS_BASE, config.FIREWORKS_API_KEY,
        payload, timeout=timeout, retries=0,
    )
    data = extract_json(message_content(resp))
    if isinstance(data, dict):
        facts = GroundedFacts(**{k: v for k, v in data.items() if k in GroundedFacts.model_fields})
        if facts.compact():
            return facts
    log.warning("grounding via %s returned unparseable JSON", model)
    return None


async def ground(
    client: httpx.AsyncClient,
    frames: List[Dict],
    duration: float,
    transcript: Optional[str],
    ocr_text: List[str],
) -> GroundedFacts:
    """Primary attempt on k2p6; if it misses its window, race a k2p6 retry against
    k2p5 in parallel (Fireworks allows concurrency; latency variance is the enemy)."""
    payload_base = {
        "max_tokens": 600,
        "temperature": 0.1,
        "reasoning_effort": "none",
        "messages": [
            {"role": "system", "content": SYSTEM.format(duration=duration)},
            {"role": "user", "content": _user_content(frames, transcript, ocr_text)},
        ],
    }
    try:
        facts = await _ground_once(client, payload_base, config.VISION_MODEL,
                                   timeout=config.GROUNDING_TIMEOUT * 0.8)
        if facts:
            return facts
    except Exception as e:
        log.warning("grounding primary failed: %s", str(e)[:120])

    racers = [
        asyncio.ensure_future(_ground_once(client, payload_base, m, timeout=config.GROUNDING_TIMEOUT))
        for m in (config.VISION_MODEL, config.VISION_MODEL_FALLBACK)
    ]
    try:
        for fut in asyncio.as_completed(racers, timeout=config.GROUNDING_TIMEOUT + 1):
            try:
                facts = await fut
                if facts:
                    return facts
            except Exception as e:
                log.warning("grounding racer failed: %s", str(e)[:120])
    except asyncio.TimeoutError:
        log.warning("grounding race timed out")
    finally:
        for r in racers:
            if not r.done():
                r.cancel()

    # Text-only degraded grounding: transcript + OCR become the facts
    return GroundedFacts(
        audio_summary=(transcript or "")[:500],
        on_screen_text=ocr_text[:10],
        uncertainty_notes=["visual grounding unavailable; facts from audio/OCR only"],
    )
