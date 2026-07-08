"""Stages 3-5: Gemma styled generation (best-of-N) + judge-replica rerank + critique/repair.

Gemma (gemma-4-31b-it via Gemini API free tier) owns ALL THREE stages — this is the
load-bearing Gemma role for the partner prize. gemma-4 always emits <thought>…</thought>
before the answer (cannot be disabled) → strip_thought() everywhere, generous max_tokens.
response_format=json_object 503s on this endpoint (probed) → JSON enforced via prompt.
"""
import asyncio
import json
import logging
import random
from typing import Dict, List, Optional

import httpx

from . import config
from .schemas import GroundedFacts
from .util import (chat_completion, clamp_words, extract_json, message_content,
                   mostly_english, strip_thought)

log = logging.getLogger("pipeline.style")

RUBRICS = {
    "formal": (
        "Neutral, precise, documentary register. Third-person. No slang, no jokes, "
        "no first person, no exclamation marks."
    ),
    "sarcastic": (
        "Dry, ironic understatement or mock-praise; humor comes from the incongruity between "
        "a grand tone and mundane content. Deadpan delivery. Never mean-spirited, never breaks the facts. "
        "Anchor the sarcasm in an emotion the viewer would plausibly feel (boredom, faux awe, mock suspense)."
    ),
    "humorous_tech": (
        "Playful humor built from tech/programming/engineering metaphors (bugs, latency, deploys, APIs, "
        "buffering, CPU, render, updates). The joke must map onto what actually happens in the video."
    ),
    "humorous_non_tech": (
        "Warm, everyday, relatable humor — witty-friend narration. Absolutely no tech jargon. "
        "Humor from everyday incongruity and charm."
    ),
}

EXEMPLARS = {
    "formal": "A young orange kitten explores a sunlit garden, pausing to bat at a low-hanging leaf before settling on a stone path.",
    "sarcastic": "Ah yes, another thrilling episode of a man heroically battling his inbox — truly the stuff of legend.",
    "humorous_tech": "The city rendered its autumn update at last — foliage textures loading in 4K, pedestrians running on background threads.",
    "humorous_non_tech": "This kitten has decided the entire garden is a personal obstacle course, and that leaf? Public enemy number one.",
}

GEN_SYSTEM = (
    "You write video captions in requested styles. Every caption MUST be factually grounded in the "
    "provided FACTS json — do not add events, objects, brands, or claims not present in FACTS. "
    "Never mention missing audio, playback speed, or video/encoding artifacts. "
    "English only. Each caption is 1-2 sentences, {wmin}-{wmax} words. "
    "Do NOT deliberate — keep any thinking under 40 tokens total, then answer. "
    "Return STRICT JSON only (no markdown fences): {shape} "
    "with exactly {n} caption(s) per style."
)

JUDGE_SYSTEM = (
    "You are the evaluation judge for video captions. For each candidate, score two independent axes "
    "from 0.0 to 1.0: accuracy = fidelity to FACTS (penalize any hallucinated or contradicted detail); "
    "style_match = how well it embodies the target style per its rubric. Do NOT reward length. "
    "Keep your thinking very brief. Return STRICT JSON only: "
    '{"<style>": {"best_index": <int>, "scores": [[accuracy, style_match], ...]}} for every style given.'
)

CRITIQUE_SYSTEM = (
    "You are a caption quality controller. For each style's caption check: (a) no factual claim outside FACTS; "
    "(b) style compliance per rubric; (c) %d-%d words; (d) English only. "
    "If a caption passes, return it unchanged. If it fails, rewrite it minimally to pass. "
    "Keep your thinking very brief. Return STRICT JSON only: "
    '{"formal": "...", "sarcastic": "...", "humorous_tech": "...", "humorous_non_tech": "..."} '
    "(only the styles given to you)."
)


def _facts_str(facts: GroundedFacts) -> str:
    return json.dumps(facts.compact(), ensure_ascii=False)


def _gen_shape(styles: List[str], n: int) -> str:
    slot = '"..."'
    inner = ", ".join('"{}": [{}]'.format(s, ", ".join([slot] * n)) for s in styles)
    return "{" + inner + "}"


_gemma_sem: Optional[asyncio.Semaphore] = None
_last_gemma_start = 0.0


def init_concurrency() -> None:
    """Called once inside the running event loop. The semaphore caps concurrent Gemma
    HTTP calls only — never the surrounding styling stage — so clips don't starve.

    Gemini free tier hard-rejects concurrent Gemma requests with 500 INTERNAL (verified:
    two simultaneous calls both fail instantly), so this MUST stay at 1 and calls are
    additionally spaced by GEMMA_MIN_GAP seconds between starts."""
    global _gemma_sem
    _gemma_sem = asyncio.Semaphore(config.GEMMA_CONCURRENCY)


async def _respect_gap() -> None:
    global _last_gemma_start
    import time
    now = time.monotonic()
    wait = config.GEMMA_MIN_GAP - (now - _last_gemma_start)
    if wait > 0:
        await asyncio.sleep(wait)
    _last_gemma_start = time.monotonic()


async def _gemma_call(client: httpx.AsyncClient, messages: List[Dict], temperature: float,
                      timeout: float, max_tokens: Optional[int] = None,
                      started: Optional[asyncio.Event] = None) -> str:
    """One Gemma call with model fallback chain. Returns thought-stripped content.
    `started` fires once the serial lane is acquired and the HTTP call actually begins —
    callers time their grace window from that moment, not from task creation."""
    last_exc: Optional[Exception] = None
    models = [m for m in (config.GEMMA_MODEL, config.GEMMA_MODEL_FALLBACK) if m]
    for model in models:
        payload = {
            "model": model,
            "max_tokens": max_tokens or config.GEMMA_MAX_TOKENS,
            "temperature": temperature,
            "top_p": 0.95,
            "messages": messages,
        }
        try:
            if _gemma_sem is not None:
                async with _gemma_sem:
                    await _respect_gap()
                    if started is not None:
                        started.set()
                    resp = await chat_completion(
                        client, config.GEMINI_OPENAI_BASE, config.GEMINI_API_KEY,
                        payload, timeout=timeout, retries=2, provider="gemini",
                    )
            else:
                await _respect_gap()
                if started is not None:
                    started.set()
                resp = await chat_completion(
                    client, config.GEMINI_OPENAI_BASE, config.GEMINI_API_KEY,
                    payload, timeout=timeout, retries=2, provider="gemini",
                )
            content = strip_thought(message_content(resp))
            if content:
                return content
            log.warning("gemma %s returned empty content after thought-strip", model)
        except Exception as e:
            last_exc = e
            log.warning("gemma %s failed: %s", model, str(e)[:150])
    raise last_exc if last_exc else RuntimeError("gemma returned empty on all models")


async def generate_candidates(
    client: httpx.AsyncClient, facts: GroundedFacts, styles: List[str], n: int, timeout: float,
    started: Optional[asyncio.Event] = None,
) -> Dict[str, List[str]]:
    """One Gemma call → n candidates per style. Raises on total failure."""
    rubric_lines = "\n".join(
        f"- {s}: {RUBRICS[s]}\n  Example of the register (do NOT copy content): {EXEMPLARS[s]}"
        for s in styles
    )
    sys_msg = GEN_SYSTEM.format(
        wmin=config.WORD_MIN, wmax=config.WORD_MAX, n=n, shape=_gen_shape(styles, n)
    )
    user = (
        f"FACTS: {_facts_str(facts)}\n\nSTYLE RUBRICS:\n{rubric_lines}\n\n"
        f"Write {n} candidate caption(s) for each of: {', '.join(styles)}. STRICT JSON now."
    )
    content = await _gemma_call(
        client,
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
        temperature=config.GEMMA_TEMP_GEN if n > 1 else 0.7,
        timeout=timeout,
        started=started,
    )
    data = extract_json(content)
    if not isinstance(data, dict):
        raise ValueError("gemma generation JSON unparseable")
    # Tolerate wrappers ({"captions": {...}}) and cosmetic key drift ("Humorous Tech")
    lowered = {str(k).strip().lower().replace(" ", "_").replace("-", "_"): v for k, v in data.items()}
    if not any(s in lowered for s in styles):
        for v in data.values():
            if isinstance(v, dict):
                inner = {str(k).strip().lower().replace(" ", "_").replace("-", "_"): vv for k, vv in v.items()}
                if any(s in inner for s in styles):
                    lowered = inner
                    break
    out: Dict[str, List[str]] = {}
    for s in styles:
        v = lowered.get(s)
        if isinstance(v, str):
            v = [v]
        if isinstance(v, list):
            cands = [str(c).strip() for c in v if str(c).strip()]
            if cands:
                out[s] = cands
    if not out:
        raise ValueError("gemma generation returned no usable styles")
    return out


async def judge_rerank(
    client: httpx.AsyncClient, facts: GroundedFacts, candidates: Dict[str, List[str]], timeout: float,
) -> Dict[str, str]:
    """Gemma judge-replica: pick best candidate per style. Falls back to first candidate."""
    fallback = {s: c[0] for s, c in candidates.items()}
    multi = {s: c for s, c in candidates.items() if len(c) > 1}
    if not multi:
        return fallback
    # Shuffle to suppress position bias; remember mapping
    perms: Dict[str, List[int]] = {}
    shuffled: Dict[str, List[str]] = {}
    for s, cands in multi.items():
        idx = list(range(len(cands)))
        random.shuffle(idx)
        perms[s] = idx
        shuffled[s] = [cands[i] for i in idx]
    user = (
        f"FACTS: {_facts_str(facts)}\n\nRUBRICS: {json.dumps({s: RUBRICS[s] for s in multi})}\n\n"
        f"CANDIDATES: {json.dumps(shuffled, ensure_ascii=False)}\n\nScore and pick best per style. STRICT JSON now."
    )
    try:
        content = await _gemma_call(
            client,
            [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
            temperature=config.GEMMA_TEMP_JUDGE,
            timeout=timeout,
        )
        data = extract_json(content)
        if isinstance(data, dict):
            for s in multi:
                entry = data.get(s)
                if isinstance(entry, dict):
                    bi = entry.get("best_index")
                    if isinstance(bi, int) and 0 <= bi < len(shuffled[s]):
                        fallback[s] = shuffled[s][bi]
    except Exception as e:
        log.warning("judge rerank failed, using first candidates: %s", str(e)[:150])
    return fallback


async def critique_repair(
    client: httpx.AsyncClient, facts: GroundedFacts, winners: Dict[str, str], timeout: float,
) -> Dict[str, str]:
    """Gemma critique/repair pass; falls back to input winners untouched."""
    user = (
        f"FACTS: {_facts_str(facts)}\n\nRUBRICS: {json.dumps({s: RUBRICS[s] for s in winners})}\n\n"
        f"CAPTIONS: {json.dumps(winners, ensure_ascii=False)}\n\nCheck and repair. STRICT JSON now."
    )
    try:
        content = await _gemma_call(
            client,
            [{"role": "system",
              "content": CRITIQUE_SYSTEM % (config.WORD_MIN, config.WORD_MAX)},
             {"role": "user", "content": user}],
            temperature=0.2,
            timeout=timeout,
        )
        data = extract_json(content)
        if isinstance(data, dict):
            repaired = {}
            for s, orig in winners.items():
                v = data.get(s)
                repaired[s] = str(v).strip() if isinstance(v, str) and str(v).strip() else orig
            return repaired
    except Exception as e:
        log.warning("critique failed, keeping winners: %s", str(e)[:150])
    return winners


async def style_via_fireworks_fallback(
    client: httpx.AsyncClient, facts: GroundedFacts, styles: List[str], timeout: float,
    frames: Optional[List[Dict]] = None,
) -> Dict[str, str]:
    """Backup racer when Gemma misses its window. When grounding produced nothing,
    Kimi grounds AND styles in one multimodal call from raw frames. Weakens the
    Gemma-prize story — logged loudly so it's visible in any run report."""
    log.error("GEMMA UNAVAILABLE — styling via Fireworks fallback (%s)", config.TEXT_FALLBACK_MODEL)
    rubric_lines = "\n".join(f"- {s}: {RUBRICS[s]}" for s in styles)
    shape = "{" + ", ".join('"{}": "..."'.format(s) for s in styles) + "}"
    have_facts = bool(facts.subjects or facts.setting or facts.actions)
    if have_facts or not frames:
        user_content: object = f"FACTS: {_facts_str(facts)}\n\nRUBRICS:\n{rubric_lines}\n\nSTRICT JSON now."
        ground_clause = "strictly factual to FACTS"
    else:
        parts: List[Dict] = [{
            "type": "text",
            "text": ("No pre-computed facts available. Look at these ordered keyframes from the video "
                     f"and caption what you actually see.\n\nRUBRICS:\n{rubric_lines}\n\nSTRICT JSON now."),
        }]
        for f in frames[:4]:
            parts.append({"type": "image_url",
                          "image_url": {"url": "data:image/jpeg;base64," + f["b64"]}})
        user_content = parts
        ground_clause = "strictly grounded in the provided keyframes"
    payload = {
        "model": config.TEXT_FALLBACK_MODEL,
        "max_tokens": 700,
        "temperature": 0.7,
        "reasoning_effort": "none",
        "messages": [
            {"role": "system",
             "content": f"You write grounded video captions. English only, {config.WORD_MIN}-{config.WORD_MAX} "
                        f"words each (hard cap 45), {ground_clause}. Never mention missing audio, "
                        "playback speed, or video/encoding artifacts — describe the scene itself. "
                        f"Return STRICT JSON only: {shape}"},
            {"role": "user", "content": user_content},
        ],
    }
    resp = await chat_completion(
        client, config.FIREWORKS_BASE, config.FIREWORKS_API_KEY, payload,
        timeout=timeout, retries=1,
    )
    data = extract_json(message_content(resp)) or {}
    return {s: str(data.get(s, "")).strip() for s in styles if str(data.get(s, "")).strip()}


def degraded_captions(facts: GroundedFacts, styles: List[str], duration: float) -> Dict[str, str]:
    """Deterministic last-rung ladder — always valid, always non-empty, style-differentiated.
    Built ONLY from observed facts (no canned answers about content we didn't see)."""
    bits = []
    if facts.subjects:
        bits.append(", ".join(facts.subjects[:3]))
    if facts.actions:
        bits.append(" and ".join(facts.actions[:2]))
    if facts.setting:
        bits.append(f"in {facts.setting}")
    scene = " ".join(bits).strip() or "a short video scene"
    dur = f"{duration:.0f}-second" if duration else "short"
    return {
        "formal": clamp_words(
            f"This {dur} clip documents {scene}, presented without narration and recorded in a single continuous take for review.",
            config.WORD_MAX),
        "sarcastic": clamp_words(
            f"Behold: {scene}. Roughly {max(duration,10):.0f} seconds of unfiltered cinema that absolutely nobody demanded, and yet here we all are, watching it anyway.",
            config.WORD_MAX),
        "humorous_tech": clamp_words(
            f"System log: {scene} loaded successfully, ran for {max(duration,10):.0f} seconds, and exited with code 0. No bugs reported, though QA remains suspicious.",
            config.WORD_MAX),
        "humorous_non_tech": clamp_words(
            f"So basically, {scene} — for {max(duration,10):.0f} whole seconds. Honestly, it's the most commitment any of us has shown all week.",
            config.WORD_MAX),
    }


def sanitize(captions: Dict[str, str], facts: GroundedFacts, styles: List[str], duration: float) -> Dict[str, str]:
    """Final gate: every requested style non-empty, English, within length cap."""
    ladder = degraded_captions(facts, styles, duration)
    out = {}
    for s in styles:
        c = (captions.get(s) or "").strip()
        if not c or not mostly_english(c):
            c = ladder[s]
        out[s] = clamp_words(c, config.WORD_MAX + 8)  # hard cap with small slack over soft band
    return out
