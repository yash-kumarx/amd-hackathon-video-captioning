"""Orchestrator: tasks.json → results.json with per-clip deadline and failure ladder.

Crash-safety: results.json is pre-filled with valid degraded captions for ALL tasks
immediately, then rewritten after each clip completes — a hard kill mid-run still
leaves a complete, schema-valid file (all four styles, never empty).
"""
import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional

import httpx

from . import config, extract, grounding, ocr, style, transcribe
from .schemas import Captions, GroundedFacts, Result, Task
from .util import Stopwatch, chat_completion, errstr, spend_summary

log = logging.getLogger("pipeline.run")

_results_lock = asyncio.Lock()


def load_tasks(path: str) -> List[Task]:
    with open(path) as f:
        raw = json.load(f)
    return [Task(**t) for t in raw]


async def write_results(results: Dict[str, Dict[str, str]], order: List[str]) -> None:
    """Atomic write, stable task order."""
    payload = [
        Result(task_id=tid, captions=Captions(**results[tid])).model_dump()
        for tid in order
    ]
    os.makedirs(os.path.dirname(config.OUTPUT_PATH) or ".", exist_ok=True)
    tmp = config.OUTPUT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, config.OUTPUT_PATH)


async def process_clip(client: httpx.AsyncClient, task: Task,
                       bank=None) -> Dict[str, str]:
    """Full pipeline for one clip; every stage degrades rather than raises.

    Speed-first plan (≤25s wall on a 1-2 vCPU runner): NO full download — ffmpeg
    range-requests frames/audio straight from the URL (~1-3MB per grab instead of a
    10-90MB file). Grounding ∥ transcription ≤9s → Gemma races a banked multimodal
    Kimi result; whichever valid result exists at the wire wins, Gemma preferred.
    """
    sw = Stopwatch(config.CLIP_DEADLINE)
    workdir = os.path.join(config.WORK_DIR, task.task_id)
    os.makedirs(workdir, exist_ok=True)
    styles = [s for s in config.ALL_STYLES]  # contract: always emit all four
    facts = GroundedFacts()
    frames: List[Dict[str, object]] = []
    duration = 0.0
    transcript: Optional[str] = None
    try:
        # --- Stage 0-1: download (network-bound, CPU-cheap), then probe + frames.
        # If the download fails, ffmpeg range-requests the URL directly as a fallback.
        video = os.path.join(workdir, "video.mp4")
        try:
            await extract.download_video(client, task.video_url, video)
        except Exception as de:
            log.warning("[%s] download failed (%s) — streaming frames from URL",
                        task.task_id, errstr(de))
            video = task.video_url
        meta = await extract.probe(video)
        duration = meta["duration"]
        frames = await extract.extract_frames(video, workdir, duration)
        log.info("[%s] %.1fs video, %d frames, audio=%s (t=%.1fs)",
                 task.task_id, duration, len(frames), meta["has_audio"], sw.elapsed)

        # --- Stage 1b/1c ∥ Stage 2: transcription and OCR run concurrent with grounding.
        # Grounding goes frames-only; transcript/OCR enrich the facts afterward.
        async def get_transcript() -> Optional[str]:
            if not (config.ENABLE_AUDIO and meta["has_audio"] and sw.remaining > 12):
                return None
            wav = await extract.extract_audio(video, workdir)
            if not wav:
                return None
            return await transcribe.transcribe(client, wav)

        async def get_ocr() -> List[str]:
            if not config.ENABLE_OCR:
                return []
            return await ocr.ocr_frames([f["path"] for f in frames])

        async def get_grounding(ocr_task: "asyncio.Task[List[str]]") -> GroundedFacts:
            # OCR is local and fast — worth a short wait so Kimi sees verified text
            ocr_text: List[str] = []
            try:
                ocr_text = await asyncio.wait_for(asyncio.shield(ocr_task), timeout=3.0)
            except (asyncio.TimeoutError, Exception):
                pass
            return await grounding.ground(client, frames, duration, None, ocr_text)

        ocr_task = asyncio.ensure_future(get_ocr())
        gtimeout = min(config.GROUNDING_TIMEOUT + 4, max(5.0, sw.remaining - 8))
        g_res, t_res = await asyncio.gather(
            asyncio.wait_for(get_grounding(ocr_task), timeout=gtimeout),
            get_transcript(),
            return_exceptions=True,
        )
        if not ocr_task.done():
            ocr_task.cancel()
        if isinstance(g_res, GroundedFacts):
            facts = g_res
        else:
            log.warning("[%s] grounding failed: %s", task.task_id, errstr(g_res))
        transcript = t_res if isinstance(t_res, str) else None
        if transcript and not transcript.startswith("NO_SPEECH"):
            facts.audio_summary = (facts.audio_summary + " | " if facts.audio_summary else "") + \
                f"speech transcript: {transcript[:800]}"
        elif transcript:
            facts.audio_summary = facts.audio_summary or transcript[:200]
        log.info("[%s] grounded: %s (t=%.1fs)", task.task_id,
                 json.dumps(facts.compact(), ensure_ascii=False)[:180], sw.elapsed)
        if bank is not None:
            # Insurance: bank a facts-aware ladder NOW so even a hard-killed clip shows
            # scene-specific captions instead of the generic pre-fill.
            try:
                await bank(style.degraded_captions(facts, styles, duration))
            except Exception:
                pass
    except Exception as e:
        log.warning("[%s] extraction/grounding degraded: %s (t=%.1fs)", task.task_id, errstr(e), sw.elapsed)

    # --- Stages 3-5: Gemma styling with a Kimi racer banked in parallel ---
    # Both lanes start together: Gemma (serial free-tier lane, preferred) and a quiet
    # Kimi backup (Fireworks, ~5-8s, pennies). Gemma gets nearly ALL residual time —
    # if it lands anywhere inside the budget it wins; otherwise the banked Kimi result
    # is already waiting. No sequential grace-then-backup dead time.
    captions: Dict[str, str] = {}
    styled_by = "ladder"
    gemma_task: Optional[asyncio.Task] = None
    kimi_task: Optional[asyncio.Task] = None
    flash_task: Optional[asyncio.Task] = None
    try:
        n = config.BEST_OF_N
        gemma_started = asyncio.Event()
        gemma_task = asyncio.ensure_future(
            style.generate_candidates(client, facts, styles, n,
                                      timeout=config.STYLE_GRACE + 6,
                                      started=gemma_started, frames=frames)
        )
        kimi_task = asyncio.ensure_future(
            style.style_via_fireworks_fallback(
                client, facts, styles,
                min(config.KIMI_STYLE_TIMEOUT, max(4.0, sw.remaining - 2)),
                frames=frames)
        )
        # Third candidate set on gemini-flash (separate provider AND separate quota —
        # pure parallel upside; fails fast on quota exhaustion and costs nothing).
        if config.ENABLE_FLASH_STYLE and frames:
            flash_task = asyncio.ensure_future(
                style.flash_style_set(
                    client, facts, styles,
                    min(9.0, max(4.0, sw.remaining - 2)), frames=frames))
        cands: Dict[str, List[str]] = {}
        # Wait for the serial Gemma lane up to the point where a minimal Gemma round
        # could still finish before the wire; past that, Gemma can't win anyway.
        lane_wait = max(0.1, sw.remaining - (config.GEMMA_MIN_GRACE + 2.5))
        got_lane = True
        try:
            await asyncio.wait_for(gemma_started.wait(), timeout=lane_wait)
        except asyncio.TimeoutError:
            got_lane = False
            log.warning("[%s] gemma lane busy for %.1fs — settling for Kimi racer",
                        task.task_id, lane_wait)
        if got_lane:
            # Timed from the actual HTTP start; leave ~2.5s to collect the banked
            # Kimi result and sanitize. Judge/critique get their cut only if enabled.
            reserve = 2.5
            if config.ENABLE_JUDGE and n > 1:
                reserve += config.JUDGE_TIMEOUT
            if config.ENABLE_CRITIQUE:
                reserve += config.CRITIQUE_TIMEOUT
            grace = min(config.STYLE_GRACE, max(0.0, sw.remaining - reserve))
            if grace >= config.GEMMA_MIN_GRACE:
                try:
                    cands = await asyncio.wait_for(asyncio.shield(gemma_task), timeout=grace)
                except asyncio.TimeoutError:
                    log.warning("[%s] gemma not done after %.1fs grace", task.task_id, grace)
                except Exception as e:
                    log.warning("[%s] gemma styling failed: %s", task.task_id, errstr(e))
            else:
                log.warning("[%s] only %.1fs grace — Kimi racer stands", task.task_id, grace)

        # Gemma is complete iff it produced all four styles; a partial Gemma result is
        # topped up per-style from the banked Kimi floor rather than discarded.
        if cands and len(cands) == len(styles):
            winners = {s: c[0] for s, c in cands.items()}
            if config.ENABLE_JUDGE and n > 1 and sw.remaining > config.JUDGE_TIMEOUT + 3:
                winners = await style.judge_rerank(client, facts, cands, config.JUDGE_TIMEOUT)
            if config.ENABLE_CRITIQUE and sw.remaining > config.CRITIQUE_TIMEOUT + 3:
                winners = await style.critique_repair(client, facts, winners, config.CRITIQUE_TIMEOUT)
            captions = winners
            styled_by = "gemma"
            log.info("[%s] gemma styled all %d styles (t=%.1fs)",
                     task.task_id, len(styles), sw.elapsed)
        else:
            # Gemma didn't fully land. Give it a short residual chance if still mid-flight,
            # then collect the banked Kimi floor. When Kimi is already banked, cap the
            # residual at 4s — finishing at ~17s beats gambling the whole window on Gemma.
            gemma_extra = max(0.0, sw.remaining - (config.KIMI_STYLE_TIMEOUT + 1.0)) \
                if got_lane and not gemma_task.done() else 0.0
            if kimi_task is not None and kimi_task.done() and not kimi_task.cancelled() \
                    and kimi_task.exception() is None and kimi_task.result():
                gemma_extra = min(gemma_extra, 4.0)
            if gemma_extra > 1.0:
                try:
                    late = await asyncio.wait_for(asyncio.shield(gemma_task), timeout=gemma_extra)
                    if late and len(late) == len(styles):
                        cands = late
                        captions = {s: c[0] for s, c in late.items()}
                        styled_by = "gemma"
                        log.info("[%s] gemma reclaimed the race (t=%.1fs)", task.task_id, sw.elapsed)
                except (asyncio.TimeoutError, Exception):
                    pass
            if not captions:
                if not gemma_task.done():
                    gemma_task.cancel()  # free the serial lane for the next clip
                kimi_res: Dict[str, str] = {}
                try:
                    kimi_res = await asyncio.wait_for(
                        asyncio.shield(kimi_task), timeout=max(2.0, sw.remaining - 0.3))
                except Exception as e2:
                    log.warning("[%s] kimi styling failed: %s", task.task_id, errstr(e2))
                # Merge: Kimi floor, topped up by any partial Gemma styles we did get
                merged = dict(kimi_res)
                for s, c in (cands or {}).items():
                    if s not in merged and c:
                        merged[s] = c[0]
                if merged:
                    captions = merged
                    styled_by = "kimi-backup" if not cands else "gemma+kimi"
                    lvl = log.warning if styled_by == "kimi-backup" else log.info
                    lvl("[%s] GEMMA MISSED — styled via %s, %d/%d styles (t=%.1fs)",
                        task.task_id, styled_by, len(captions), len(styles), sw.elapsed)

        # --- Additive verify step: if the flash set also landed, let a fast frame-
        # grounded verifier pick per style between the primary result and flash's.
        # Strictly optional: any failure or time pressure leaves `captions` untouched.
        if (captions and len(captions) == len(styles) and flash_task is not None
                and sw.remaining > 4.5):
            flash_set: Dict[str, str] = {}
            try:
                flash_set = await asyncio.wait_for(
                    asyncio.shield(flash_task), timeout=max(0.3, sw.remaining - 4.2))
            except (asyncio.TimeoutError, Exception):
                pass
            if flash_set and len(flash_set) == len(styles) and sw.remaining > 4.0:
                pools = {s: [captions[s], flash_set[s]] for s in styles}
                try:
                    picked = await style.pick_from_pools(
                        client, frames, pools, styles,
                        timeout=min(4.5, max(2.0, sw.remaining - 1.0)),
                        provider="gemini")
                    if picked and all(picked.get(s) for s in styles):
                        captions = picked
                        styled_by += "+flash-verified"
                        log.info("[%s] flash verify applied (t=%.1fs)", task.task_id, sw.elapsed)
                except Exception:
                    pass
    except Exception as e:
        log.warning("[%s] styling stage error: %s (t=%.1fs)", task.task_id, errstr(e), sw.elapsed)
    finally:
        for t in (gemma_task, kimi_task, flash_task):
            if t is not None and not t.done():
                t.cancel()

    final = style.sanitize(captions, facts, styles, duration)
    extract.cleanup(workdir)
    log.info("[%s] DONE in %.1fs (styled_by=%s)", task.task_id, sw.elapsed, styled_by)
    return final


async def run() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    t0 = time.monotonic()
    try:
        tasks = load_tasks(config.INPUT_PATH)
    except Exception as e:
        log.error("cannot read %s: %s — writing empty results", config.INPUT_PATH, e)
        os.makedirs(os.path.dirname(config.OUTPUT_PATH) or ".", exist_ok=True)
        with open(config.OUTPUT_PATH, "w") as f:
            json.dump([], f)
        return 0

    order = [t.task_id for t in tasks]
    # Pre-fill: a complete valid file exists from the first second
    results: Dict[str, Dict[str, str]] = {
        t.task_id: style.degraded_captions(GroundedFacts(), config.ALL_STYLES, 0.0) for t in tasks
    }
    await write_results(results, order)

    clip_sem = asyncio.Semaphore(config.CLIP_CONCURRENCY)
    style.init_concurrency()
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)

    # Adaptive stagger: shrink if the task count would push total wall time past
    # GLOBAL_BUDGET (stagger×(n-1) + deadline + startup margin ≤ budget).
    n_tasks = len(tasks)
    stagger = config.CLIP_STAGGER
    if n_tasks > 1:
        max_stagger = (config.GLOBAL_BUDGET - config.CLIP_DEADLINE - 25) / (n_tasks - 1)
        stagger = max(4.0, min(stagger, max_stagger))
    log.info("plan: %d clips, stagger=%.1fs, deadline=%.0fs, est wall=%.0fs",
             n_tasks, stagger, config.CLIP_DEADLINE,
             stagger * max(0, n_tasks - 1) + config.CLIP_DEADLINE)

    async with httpx.AsyncClient(limits=limits) as client:
        async def worker(task: Task, index: int) -> None:
            # Stagger starts to pipeline clips through the serial Gemma lane; the
            # per-clip clock starts at ITS processing start, so waiting here costs
            # nothing against the per-clip budget (total stays under GLOBAL_BUDGET).
            if index and stagger > 0:
                await asyncio.sleep(index * stagger)
            async with clip_sem:
                async def bank(caps: Dict[str, str]) -> None:
                    async with _results_lock:
                        results[task.task_id] = caps
                        await write_results(results, order)

                try:
                    captions = await asyncio.wait_for(
                        process_clip(client, task, bank=bank),
                        timeout=config.CLIP_DEADLINE + 2,  # absolute backstop < 30s
                    )
                except Exception as e:
                    log.error("[%s] clip hard-failed: %s — banked/degraded captions stand",
                              task.task_id, errstr(e))
                    return
                await bank(captions)

        async def warm(base: str, key: str, payload: Dict, provider: str) -> None:
            try:
                await chat_completion(client, base, key, payload, timeout=12, retries=0, provider=provider)
            except Exception:
                pass

        # Fire-and-forget connection warmup (TLS + TTFB) while first downloads run
        asyncio.ensure_future(warm(
            config.GEMINI_OPENAI_BASE, config.GEMINI_API_KEY,
            {"model": config.GEMMA_MODEL, "max_tokens": 32,
             "messages": [{"role": "user", "content": "Say OK"}]}, "gemini"))
        asyncio.ensure_future(warm(
            config.FIREWORKS_BASE, config.FIREWORKS_API_KEY,
            {"model": config.VISION_MODEL, "max_tokens": 8, "reasoning_effort": "none",
             "messages": [{"role": "user", "content": "Say OK"}]}, "fireworks"))

        await asyncio.gather(*(worker(t, i) for i, t in enumerate(tasks)))

    await write_results(results, order)
    log.info("ALL DONE: %d clips in %.1fs. Spend: %s", len(tasks), time.monotonic() - t0, spend_summary())
    return 0
