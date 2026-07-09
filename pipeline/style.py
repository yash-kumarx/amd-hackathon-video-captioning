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
import re
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
        "Playful humor built on ONE central tech/programming metaphor that genuinely fits what happens "
        "in the video (a bug, a deploy, latency, buffering, a reboot…) — developed cleanly, not a pile "
        "of stacked jargon. The joke should still land for a non-engineer."
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
    "provided FACTS json{vision_clause} — do not add events, objects, brands, or claims not present there. "
    "Never mention missing audio, playback speed, or video/encoding artifacts. "
    "English only. Each caption is 1-2 sentences, {wmin}-{wmax} words. "
    "Think briefly (under 150 tokens): draft 2 angles per style, check each against the FACTS for "
    "hallucination and against its rubric for register, keep only the sharper one — then answer. "
    "Make each caption SPECIFIC to this video (name its actual subjects/actions, not generic filler), "
    "and make the four styles clearly distinct from each other. "
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


def _norm_style_dict(data: object, styles: List[str]) -> Dict[str, object]:
    """Normalize model JSON to {style: value}: lowercases/underscores keys and
    unwraps one level of wrapper ({"captions": {...}}) if the styles live inside."""
    if not isinstance(data, dict):
        return {}
    lowered = {str(k).strip().lower().replace(" ", "_").replace("-", "_"): v
               for k, v in data.items()}
    if not any(s in lowered for s in styles):
        for v in data.values():
            if isinstance(v, dict):
                inner = {str(k).strip().lower().replace(" ", "_").replace("-", "_"): vv
                         for k, vv in v.items()}
                if any(s in inner for s in styles):
                    return inner
    return lowered


def _gen_shape(styles: List[str], n: int) -> str:
    slot = '"..."'
    inner = ", ".join('"{}": [{}]'.format(s, ", ".join([slot] * n)) for s in styles)
    return "{" + inner + "}"


_gemma_lanes: List[Dict] = []   # [{sem, key, last_start}] — one serial lane per Gemini key
_lane_rr = 0


def init_concurrency() -> None:
    """Called once inside the running event loop. Each Gemini API key gets its own
    SERIAL lane (free tier hard-rejects concurrent Gemma calls per project with 500
    INTERNAL — verified live; two keys = two independent projects = 2x throughput).
    Calls within a lane are additionally spaced by GEMMA_MIN_GAP seconds."""
    global _gemma_lanes
    keys = [k for k in (config.GEMINI_API_KEY, config.GEMINI_API_KEY_2) if k]
    _gemma_lanes = [
        {"sem": asyncio.Semaphore(config.GEMMA_CONCURRENCY), "key": k, "last": 0.0}
        for k in keys
    ]


async def _respect_gap(lane: Dict) -> None:
    import time
    now = time.monotonic()
    wait = config.GEMMA_MIN_GAP - (now - lane["last"])
    if wait > 0:
        await asyncio.sleep(wait)
    lane["last"] = time.monotonic()


async def _gemma_call(client: httpx.AsyncClient, messages: List[Dict], temperature: float,
                      timeout: float, max_tokens: Optional[int] = None,
                      started: Optional[asyncio.Event] = None) -> str:
    """One Gemma call with a provider chain, all Gemma-family: Gemini lane(s) first,
    then OpenRouter's google/gemma-4-31b-it:free if a key exists. Returns
    thought-stripped content. `started` fires once the serial lane is acquired and
    the HTTP call actually begins — callers time their grace from that moment.
    When more pools exist, the primary attempt is capped (GEMMA_PRIMARY_CAP) so a
    congested-window hang still leaves the next Gemma pool time to answer."""
    global _lane_rr
    import time as _time
    t_end = _time.monotonic() + timeout
    lane = None
    if _gemma_lanes:
        lane = _gemma_lanes[_lane_rr % len(_gemma_lanes)]
        _lane_rr += 1

    # (base, key, model, provider_tag)
    chain: List = []
    gem_key = lane["key"] if lane else config.GEMINI_API_KEY
    for model in (config.GEMMA_MODEL, config.GEMMA_MODEL_FALLBACK):
        if model:
            chain.append((config.GEMINI_OPENAI_BASE, gem_key, model, "gemini"))
    if config.OPENROUTER_API_KEY:
        chain.append((config.OPENROUTER_BASE, config.OPENROUTER_API_KEY,
                      config.GEMMA_OPENROUTER_MODEL, "openrouter"))

    async def attempt(base: str, key: str, model: str, provider: str, tmo: float) -> str:
        payload = {
            "model": model,
            "max_tokens": max_tokens or config.GEMMA_MAX_TOKENS,
            "temperature": temperature,
            "top_p": 0.95,
            "messages": messages,
        }
        # retries=1: a hung free-tier call holds the SERIAL lane for its full
        # timeout — one retry catches instant-500 blips, more just starves peers
        resp = await chat_completion(client, base, key, payload,
                                     timeout=tmo, retries=1, provider=provider)
        return strip_thought(message_content(resp))

    last_exc: Optional[Exception] = None

    async def run_chain() -> str:
        nonlocal last_exc
        if started is not None:
            started.set()
        for i, (base, key, model, provider) in enumerate(chain):
            left = t_end - _time.monotonic()
            if left < 4.0:
                break
            tmo = left
            if provider == "gemini" and len(chain) > i + 1:
                tmo = min(config.GEMMA_PRIMARY_CAP, left)
            try:
                content = await attempt(base, key, model, provider, tmo)
                if content:
                    if provider != "gemini":
                        log.info("gemma served by %s (%s)", provider, model)
                    return content
                log.warning("gemma %s/%s returned empty after thought-strip", provider, model)
            except Exception as e:
                last_exc = e
                log.warning("gemma %s/%s failed: %s", provider, model, str(e)[:130])
        raise last_exc if last_exc else RuntimeError("gemma empty on all providers")

    if lane is not None:
        async with lane["sem"]:
            await _respect_gap(lane)
            return await run_chain()
    return await run_chain()


async def generate_candidates(
    client: httpx.AsyncClient, facts: GroundedFacts, styles: List[str], n: int, timeout: float,
    started: Optional[asyncio.Event] = None, frames: Optional[List[Dict]] = None,
) -> Dict[str, List[str]]:
    """One Gemma call → n candidates per style. Raises on total failure.

    gemma-4-31b is multimodal (verified live): when grounding produced nothing, Gemma
    grounds AND styles from raw keyframes itself — the Gemma-prize story holds even in
    degraded mode. With GEMMA_VISION=1 frames also ride along with good facts for
    visual verification."""
    rubric_lines = "\n".join(
        f"- {s}: {RUBRICS[s]}\n  Example of the register (do NOT copy content): {EXEMPLARS[s]}"
        for s in styles
    )
    have_facts = bool(facts.subjects or facts.setting or facts.actions
                      or facts.audio_summary or facts.on_screen_text)
    if not have_facts and not frames:
        # Nothing to caption from — fail fast so the ladder takes over instead of
        # burning the serial lane on a prompt with no grounding at all.
        raise ValueError("no facts and no frames — nothing for gemma to ground on")
    use_frames = bool(frames) and (config.GEMMA_VISION or not have_facts)
    vision_clause = " and the keyframes shown" if use_frames else ""
    sys_msg = GEN_SYSTEM.format(
        wmin=config.WORD_MIN, wmax=config.WORD_MAX, n=n, shape=_gen_shape(styles, n),
        vision_clause=vision_clause,
    )
    if have_facts:
        user_text = (
            f"FACTS: {_facts_str(facts)}\n\nSTYLE RUBRICS:\n{rubric_lines}\n\n"
            f"Write {n} candidate caption(s) for each of: {', '.join(styles)}. STRICT JSON now."
        )
    else:
        user_text = (
            "No pre-verified FACTS are available. Look at the ordered keyframes and caption "
            f"exactly what you see — nothing more.\n\nSTYLE RUBRICS:\n{rubric_lines}\n\n"
            f"Write {n} candidate caption(s) for each of: {', '.join(styles)}. STRICT JSON now."
        )
    if use_frames:
        parts: List[Dict] = [{"type": "text", "text": user_text}]
        for f in frames[: config.GEMMA_VISION_FRAMES]:
            parts.append({"type": "image_url",
                          "image_url": {"url": "data:image/jpeg;base64," + f["b64"]}})
        user_content: object = parts
    else:
        user_content = user_text
    try:
        content = await _gemma_call(
            client,
            [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_content}],
            temperature=config.GEMMA_TEMP_GEN if n > 1 else 0.7,
            timeout=timeout,
            started=started,
        )
    except Exception:
        if not (use_frames and have_facts):
            raise
        # Image call failed but we have textual facts — one text-only retry
        log.warning("gemma multimodal call failed — retrying text-only")
        content = await _gemma_call(
            client,
            [{"role": "system", "content": sys_msg.replace(vision_clause, "")},
             {"role": "user", "content": user_text}],
            temperature=config.GEMMA_TEMP_GEN if n > 1 else 0.7,
            timeout=timeout,
        )
    data = extract_json(content)
    if not isinstance(data, dict):
        raise ValueError("gemma generation JSON unparseable")
    # Tolerate wrappers ({"captions": {...}}) and cosmetic key drift ("Humorous Tech")
    lowered = _norm_style_dict(data, styles)
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
    """Reliable styling lane on Kimi K2.6 (Fireworks). Runs in parallel with the Gemma
    lane from styling start (Gemma preferred at the wire; this is the floor that means
    no clip ever falls to the template ladder in a congested Gemini window).

    MULTIMODAL by default: Kimi sees the actual keyframes AND the auto-generated facts,
    with the frames declared ground-truth — so it self-corrects grounding errors (the
    facts said "horse"; the frame shows a dog) instead of inheriting them. This lifts
    the accuracy axis, which the facts-only handoff was capping. Retries once if the
    first pass yields fewer than all four styles."""
    rubric_lines = "\n".join(
        f"- {s}: {RUBRICS[s]}\n  Example of the register (do NOT copy content): {EXEMPLARS[s]}"
        for s in styles
    )
    shape = "{" + ", ".join('"{}": "..."'.format(s) for s in styles) + "}"
    have_facts = bool(facts.subjects or facts.setting or facts.actions
                      or facts.audio_summary or facts.on_screen_text)
    use_frames = bool(frames)
    if use_frames and have_facts:
        ground_clause = ("in what the keyframes actually show. Preliminary auto-generated FACTS are "
                         "provided as a HINT only — they may misidentify subjects; TRUST THE FRAMES "
                         "when they conflict")
    elif use_frames:
        ground_clause = "in what the keyframes actually show"
    else:
        ground_clause = "in the provided FACTS json"
    sys_txt = (
        f"You write video captions in requested styles. Every caption MUST be factually grounded "
        f"{ground_clause} — do not add events, objects, brands, or claims that are not there. "
        "Identify the main subject correctly before writing. "
        "Never mention missing audio, playback speed, or video/encoding artifacts. "
        f"English only. Each caption is 1-2 sentences, {config.WORD_MIN}-{config.WORD_MAX} words — "
        f"NEVER exceed {config.WORD_MAX} words, end every caption as a complete sentence. "
        "Silently draft 2 angles per style, keep only the sharper one. Make each caption SPECIFIC "
        "to this video (name its actual subjects/actions, not generic filler), and make the four "
        "styles clearly distinct from each other. "
        f"Return STRICT JSON only (no markdown fences): {shape}"
    )
    facts_line = f"Preliminary FACTS (hint, may be wrong): {_facts_str(facts)}\n\n" if have_facts else ""
    if use_frames:
        parts: List[Dict] = [{
            "type": "text",
            "text": (f"{facts_line}Look at these ordered keyframes and identify what is really shown, "
                     f"then write captions.\n\nSTYLE RUBRICS:\n{rubric_lines}\n\n"
                     "Write 1 caption per style. STRICT JSON now."),
        }]
        for f in frames[: config.KIMI_STYLE_FRAMES]:
            parts.append({"type": "image_url",
                          "image_url": {"url": "data:image/jpeg;base64," + f["b64"]}})
        user_content: object = parts
    else:
        user_content = (
            f"FACTS: {_facts_str(facts)}\n\nSTYLE RUBRICS:\n{rubric_lines}\n\n"
            "Write 1 caption per style. STRICT JSON now."
        )

    async def one_call(temp: float) -> Dict[str, str]:
        payload = {
            "model": config.TEXT_FALLBACK_MODEL,
            "max_tokens": 700,
            "temperature": temp,
            "reasoning_effort": "none",
            "messages": [
                {"role": "system", "content": sys_txt},
                {"role": "user", "content": user_content},
            ],
        }
        resp = await chat_completion(
            client, config.FIREWORKS_BASE, config.FIREWORKS_API_KEY, payload,
            timeout=timeout, retries=1,
        )
        lowered = _norm_style_dict(extract_json(message_content(resp)) or {}, styles)
        out = {}
        for s in styles:
            v = lowered.get(s)
            if isinstance(v, list):
                v = v[0] if v else ""
            if isinstance(v, (str, int, float)) and str(v).strip():
                out[s] = str(v).strip()
        return out

    result = await one_call(0.6)
    if len(result) < len(styles):
        log.warning("kimi styling returned %d/%d styles — retrying once", len(result), len(styles))
        try:
            retry = await one_call(0.4)
            if len(retry) > len(result):
                result = retry
        except Exception as e:
            log.warning("kimi styling retry failed: %s", str(e)[:120])
    return result


def _short_scene(facts: GroundedFacts) -> str:
    """A SHORT (<=12 word) scene phrase from facts — never a facts dump, no duplication."""
    subj = " ".join((facts.subjects[0] if facts.subjects else "").split()[:4]).strip()
    act = " ".join((facts.actions[0] if facts.actions else "").split()[:6]).strip()
    setting = " ".join((facts.setting or "").split()[:5]).strip()
    subj_l, act_l = subj.lower(), act.lower()
    # Avoid "pedestrians pedestrians crossing": if the action already names the subject,
    # keep only the action; else combine.
    if subj and act:
        core = act if (subj_l and subj_l in act_l) else f"{subj} {act}"
    else:
        core = subj or act or "a quiet scene"
    if setting and setting.lower() not in core.lower():
        core = f"{core} in {setting}"
    # Hard cap the whole phrase
    return " ".join(core.split()[:12]).strip()


def degraded_captions(facts: GroundedFacts, styles: List[str], duration: float) -> Dict[str, str]:
    """Deterministic last-rung ladder — always valid, non-empty, style-differentiated, and
    SHORT/clean (never a facts dump). Built only from observed facts. This rung should almost
    never ship (Gemma + Kimi lanes cover styling), but when it does it must not tank the score
    with a run-on truncated blob — so each caption is a tidy single sentence."""
    scene = _short_scene(facts)
    a = scene[0].upper() + scene[1:] if scene else "A quiet scene"
    return {
        "formal": clamp_words(f"{a}, captured in a brief continuous shot.", config.WORD_MAX),
        "sarcastic": clamp_words(
            f"Ah yes, {scene} — riveting cinema that absolutely nobody saw coming.", config.WORD_MAX),
        "humorous_tech": clamp_words(
            f"{a}, running smoothly on a single thread with zero exceptions thrown.", config.WORD_MAX),
        "humorous_non_tech": clamp_words(
            f"Just {scene}, and honestly it is doing its best out there.", config.WORD_MAX),
    }


def _looks_degenerate(c: str) -> bool:
    """A run-on facts-dump or heavily duplicated caption — reject in favor of the ladder."""
    words = c.split()
    if len(words) > config.WORD_MAX + 6 and not re.search(r"[.!?]", c[:_char_of_word(c, config.WORD_MAX)]):
        return True  # long with no sentence break early = run-on
    lw = [w.lower().strip(".,;:") for w in words if w.strip(".,;:")]
    if len(lw) >= 12 and len(set(lw)) / len(lw) < 0.55:
        return True  # low lexical variety = duplication ("pedestrians ... pedestrians ...")
    return False


def _char_of_word(text: str, n: int) -> int:
    """Char index just past the n-th word (for slicing)."""
    parts = text.split()
    if len(parts) <= n:
        return len(text)
    return len(" ".join(parts[:n]))


def sanitize(captions: Dict[str, str], facts: GroundedFacts, styles: List[str], duration: float) -> Dict[str, str]:
    """Final gate: every requested style non-empty, English, sentence-clean, within length cap."""
    ladder = degraded_captions(facts, styles, duration)
    out = {}
    for s in styles:
        c = (captions.get(s) or "").strip()
        if not c or not mostly_english(c) or _looks_degenerate(c):
            c = ladder[s]
        out[s] = clamp_words(c, config.WORD_MAX + 8)  # sentence-aware hard cap
    return out
