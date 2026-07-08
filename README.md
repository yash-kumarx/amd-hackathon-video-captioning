# Video Captioning Agent — AMD Developer Hackathon ACT II, Track 2

Grounded, style-controlled video captioning: every clip is analyzed by a vision model into
verifiable facts, then **Google Gemma** writes, selects, and quality-controls four styled
captions (`formal`, `sarcastic`, `humorous_tech`, `humorous_non_tech`).

## Gemma is the load-bearing brain of this system

**Model:** `gemma-4-31b-it` via the Gemini API OpenAI-compatible endpoint
(`https://generativelanguage.googleapis.com/v1beta/openai/`).

Gemma owns the entire language side of the pipeline:

1. **Styled generation (all four styles, one call).** Gemma receives the grounded-facts JSON
   plus per-style rubrics and register exemplars, and writes every caption the system emits.
2. **Test-time selection inside Gemma's thinking.** `gemma-4-31b-it` reasons before answering
   (`<thought>…</thought>`). The generation prompt directs that reasoning: draft angles, check
   each against the FACTS for hallucinations and against the rubric for style fidelity, then
   emit only the winner per style — best-of-N with the judge internalized, at zero extra latency.
3. **Judge-replica reranking (`ENABLE_JUDGE=1`).** For explicit best-of-N (`BEST_OF_N=3`),
   Gemma scores candidates on accuracy-vs-facts and style-match (position-shuffled to suppress
   position bias, instructed not to reward length to suppress verbosity bias) and picks per-style
   winners.
4. **Self-critique / repair (`ENABLE_CRITIQUE=1`).** Gemma re-checks winners for factual
   contradictions, style compliance, length band, and English-only, repairing inline.

Gemma-4 is natively **multimodal** on this endpoint (verified by probe: it correctly described
video keyframes) — the critique pass can be pointed at raw frames for visual verification.

If Gemma cannot answer within the clip's latency budget (free-tier capacity is variable), a
backup lane produces captions so no clip ever fails — but Gemma is always attempted first,
and every fallback is logged loudly (`styled_by=` in the run log tells the truth per clip).

## Architecture

```
tasks.json → per clip (staggered, pipelined):
  1. download + ffprobe
  2. keyframes (ffmpeg, 8-10 @ ≤1024px)         ─┐ concurrent
     audio → gemini-2.5-flash transcription      ─┤
     RapidOCR on-screen text (local, in-image)   ─┘
  3. GROUNDING: Kimi K2.6 (Fireworks serverless) + transcript + OCR
     → strict facts JSON {subjects, actions, setting, on_screen_text, mood, …}
  4. STYLING: ★ GEMMA ★ facts → 4 styled captions (draft-judge-select in thinking)
     [optional: Gemma judge-replica rerank, Gemma critique/repair]
  5. validate (pydantic) → degraded-but-valid ladder → results.json, exit 0
```

| Role | Model | Endpoint |
|---|---|---|
| Styling + selection + judge + critique | **`gemma-4-31b-it`** | Gemini API (OpenAI-compatible), free tier |
| Vision grounding | `kimi-k2p6` (fallback `kimi-k2p5`) | Fireworks serverless |
| Audio transcription | `gemini-2.5-flash` (`input_audio`) | Gemini API free tier |
| On-screen text | RapidOCR (ONNX, baked into image) | local, no network |

### Reliability engineering (why this never zeros a clip)
- `results.json` is **pre-filled valid** at startup and atomically rewritten per clip — a hard
  kill still leaves a complete, schema-valid file with all four styles non-empty.
- Per-clip hard deadline ≈27s (<30s contract) with stage-level budgets; clips are staggered so
  the serialized Gemma lane (Gemini free tier rejects concurrent calls — verified) is free
  exactly when each clip reaches styling.
- Every stage degrades: grounding races two vision models; styling races Gemma against a
  Kimi backup; the last rung templates captions from whatever facts exist. English-only and
  length checks run in the final gate.

## Running

```bash
docker run --rm \
  -e FIREWORKS_API_KEY=... -e GEMINI_API_KEY=... \
  -v /path/to/input:/input -v /path/to/output:/output \
  <image>
```

Reads `/input/tasks.json` (`[{task_id, video_url, styles[]}]`), writes `/output/results.json`
(`[{task_id, captions:{formal, sarcastic, humorous_tech, humorous_non_tech}}]`), exits 0.

Key env knobs (all optional): `BEST_OF_N`, `ENABLE_JUDGE`, `ENABLE_CRITIQUE`, `ENABLE_AUDIO`,
`ENABLE_OCR`, `CLIP_STAGGER`, `STYLE_GRACE`, `GEMMA_MODEL`.

## Design decisions (probed live, 2026-07-08)

- `gemma-3-27b-it` no longer exists on the Gemini API → `gemma-4-31b-it`.
- No Qwen-VL model is serverless on Fireworks → Kimi K2.6 grounds vision.
- Fireworks deprecated audio inference (2026-06-10) → Gemini Flash transcribes audio.
- Gemini free tier 500s on concurrent Gemma calls → serialized lane + clip staggering.
- Gemma-4 thinking can't be disabled → prompt-constrained brief thinking (~11s → ~6-8s calls)
  and thought-stripping; the thinking is put to work as in-context best-of-N selection.
