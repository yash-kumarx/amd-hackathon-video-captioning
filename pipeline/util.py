import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import httpx

log = logging.getLogger("pipeline")

_spend_lock = asyncio.Lock()
SPEND = {"fireworks_prompt": 0, "fireworks_completion": 0, "gemini_calls": 0}


async def record_usage(provider: str, usage: Optional[dict]) -> None:
    if not usage:
        return
    async with _spend_lock:
        if provider == "fireworks":
            SPEND["fireworks_prompt"] += usage.get("prompt_tokens", 0) or 0
            SPEND["fireworks_completion"] += usage.get("completion_tokens", 0) or 0
        else:
            SPEND["gemini_calls"] += 1


def spend_summary() -> str:
    # kimi-k2p6 serverless: $0.95/M in, $4.00/M out (cached in $0.16/M — count worst case)
    usd = SPEND["fireworks_prompt"] / 1e6 * 0.95 + SPEND["fireworks_completion"] / 1e6 * 4.00
    return (f"fireworks tokens in/out={SPEND['fireworks_prompt']}/{SPEND['fireworks_completion']}"
            f" (~${usd:.4f}); gemini free-tier calls={SPEND['gemini_calls']}")


async def chat_completion(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout: float,
    retries: int = 2,
    provider: str = "fireworks",
) -> Dict[str, Any]:
    """OpenAI-compatible chat call with backoff. Raises on final failure."""
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            r = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=timeout,
            )
            if r.status_code in (429, 500, 502, 503, 504):
                raise TransientAPIError(f"HTTP {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            data = r.json()
            # Gemini compat wraps errors in a list sometimes
            if isinstance(data, list):
                data = data[0] if data else {}
            if "error" in data:
                raise TransientAPIError(str(data["error"])[:200])
            await record_usage(provider, data.get("usage"))
            return data
        except (TransientAPIError, httpx.TimeoutException, httpx.TransportError) as e:
            last_exc = e
            if attempt < retries:
                # 5xx/timeouts: retry near-immediately (capacity blips / latency variance);
                # 429 rate limit: brief pause
                s = str(e)
                delay = 1.2 if "429" in s else 0.3
                log.warning("chat_completion retry %d after %.1fs (%s)", attempt + 1, delay, errstr(e))
                await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


def errstr(e: BaseException) -> str:
    """str(TimeoutError()) is empty — always include the type name."""
    return f"{type(e).__name__}: {str(e)[:160]}"


class TransientAPIError(Exception):
    pass


_THOUGHT_RE = re.compile(r"<thought>.*?</thought>", re.DOTALL)


def strip_thought(text: str) -> str:
    """gemma-4 thinking cannot be disabled; final answer follows </thought>."""
    if not text:
        return ""
    out = _THOUGHT_RE.sub("", text)
    # Unclosed thought block (truncation): drop everything from <thought>
    if "<thought>" in out:
        out = out.split("<thought>")[0]
    return out.strip()


def message_content(resp: Dict[str, Any]) -> str:
    choices = resp.get("choices") or [{}]
    return (choices[0].get("message") or {}).get("content") or ""


def extract_json(text: str) -> Optional[Any]:
    """Best-effort JSON extraction from model output (fences, prose, trailing text)."""
    if not text:
        return None
    text = text.strip()
    # Try fenced block first
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Raw parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Outermost braces, then progressively trim trailing garbage
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break
    return None


def clamp_words(text: str, max_words: int) -> str:
    """Cut overlong text at a sentence boundary when one exists reasonably deep in,
    so judges never see a mid-thought amputation ("Spoiler: it.")."""
    words = text.split()
    if len(words) <= max_words:
        return text
    prefix = " ".join(words[:max_words])
    ends = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", prefix)]
    if ends:
        candidate = prefix[: ends[-1]].strip()
        # Only accept the sentence cut if it keeps a substantial caption
        if len(candidate.split()) >= max(8, int(max_words * 0.5)):
            return candidate
    out = prefix.rstrip(",;:— ")
    if not out.endswith((".", "!", "?")):
        out += "."
    return out


def mostly_english(text: str) -> bool:
    if not text:
        return False
    ascii_letters = sum(1 for c in text if c.isascii())
    return ascii_letters / max(len(text), 1) > 0.9


class Stopwatch:
    def __init__(self, budget: float):
        self.t0 = time.monotonic()
        self.budget = budget

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    @property
    def remaining(self) -> float:
        return max(0.0, self.budget - self.elapsed)
