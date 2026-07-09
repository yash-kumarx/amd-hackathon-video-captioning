"""Central configuration. Everything overridable via env so leaderboard A/B needs no code edits."""
import os

def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")

# --- Secrets (never hardcode) ---
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# Optional second Gemini project key: adds an independent serial Gemma lane (2x throughput)
GEMINI_API_KEY_2 = os.environ.get("GEMINI_API_KEY_2", "")
# Optional OpenRouter key: google/gemma-4-31b-it:free = same Gemma, independent capacity pool
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# --- Endpoints ---
FIREWORKS_BASE = "https://api.fireworks.ai/inference/v1"
GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
GEMMA_OPENROUTER_MODEL = os.environ.get("GEMMA_OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
# When more providers exist in the chain, the primary Gemini attempt is capped so a
# congested-window hang (36s+ measured) leaves time for the next Gemma pool.
GEMMA_PRIMARY_CAP = float(os.environ.get("GEMMA_PRIMARY_CAP", "20"))

# --- Models (verified live 2026-07-08; see STATUS.md) ---
VISION_MODEL = os.environ.get("VISION_MODEL", "accounts/fireworks/models/kimi-k2p6")
VISION_MODEL_FALLBACK = os.environ.get("VISION_MODEL_FALLBACK", "accounts/fireworks/models/kimi-k2p5")
GEMMA_MODEL = os.environ.get("GEMMA_MODEL", "gemma-4-31b-it")
# gemma-4-26b-a4b-it 500'd on every live probe 2026-07-08 — disabled by default;
# re-enable via env if it recovers. Empty string = no second Gemma model in the chain.
GEMMA_MODEL_FALLBACK = os.environ.get("GEMMA_MODEL_FALLBACK", "")
AUDIO_MODEL = os.environ.get("AUDIO_MODEL", "gemini-2.5-flash")  # Fireworks audio deprecated 2026-06-10
# Last-resort styling fallback if Gemini API is entirely down (weakens Gemma story; logged loudly)
TEXT_FALLBACK_MODEL = os.environ.get("TEXT_FALLBACK_MODEL", "accounts/fireworks/models/kimi-k2p6")

# --- Contract paths ---
INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
WORK_DIR = os.environ.get("WORK_DIR", "/tmp/clips")

# --- Styles contract ---
ALL_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

# --- Feature flags (baseline ships conservative; Stage B flips via env, no code change) ---
ENABLE_AUDIO = _flag("ENABLE_AUDIO", "1")     # gemini-2.5-flash transcription (separate quota from Gemma)
ENABLE_OCR = _flag("ENABLE_OCR", "1")         # local RapidOCR; auto-skips if import fails
BEST_OF_N = int(os.environ.get("BEST_OF_N", "1"))          # 1 = banked baseline; 3 = Stage B
ENABLE_JUDGE = _flag("ENABLE_JUDGE", "0")     # Gemma judge-replica rerank (only meaningful if N>1)
ENABLE_CRITIQUE = _flag("ENABLE_CRITIQUE", "0")  # Gemma self-critique/repair pass

# --- Frames ---
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "14"))
MIN_FRAMES = int(os.environ.get("MIN_FRAMES", "8"))
FRAME_LONG_EDGE = int(os.environ.get("FRAME_LONG_EDGE", "1024"))
FRAME_JPEG_Q = int(os.environ.get("FRAME_JPEG_Q", "4"))  # ffmpeg -q:v scale (2..6; lower=better)

# --- Concurrency & budgets (seconds) ---
# Timing model: the enforceable contract is TOTAL runtime ≤ 10 min for the batch
# (container reads tasks.json once — there is no per-request clock at runtime).
# Hidden clips are 30s-2min; the old 27s wall starved them. Per-clip budget is now
# ~48s with staggering sized so total stays < GLOBAL_BUDGET even for large N.
GLOBAL_BUDGET = float(os.environ.get("GLOBAL_BUDGET", "540"))
CLIP_CONCURRENCY = int(os.environ.get("CLIP_CONCURRENCY", "3"))
# Gemini free tier 500s on ANY concurrent Gemma calls (verified live) → serialize + gap
GEMMA_CONCURRENCY = int(os.environ.get("GEMMA_CONCURRENCY", "1"))
GEMMA_MIN_GAP = float(os.environ.get("GEMMA_MIN_GAP", "0.8"))
# Stagger successive clip starts so each clip reaches styling just as the serial
# Gemma lane frees up. Sized for CONGESTED free-tier windows (36s/call measured):
# at 20s stagger the lane still serves most clips; calm windows just idle briefly.
# run() shrinks the stagger automatically if task count would overflow GLOBAL_BUDGET.
CLIP_STAGGER = float(os.environ.get("CLIP_STAGGER", "20"))
CLIP_DEADLINE = float(os.environ.get("CLIP_DEADLINE", "55"))
DOWNLOAD_TIMEOUT = float(os.environ.get("DOWNLOAD_TIMEOUT", "30"))  # 2-min clips can be 30-60MB
GROUNDING_TIMEOUT = float(os.environ.get("GROUNDING_TIMEOUT", "14"))
AUDIO_TIMEOUT = float(os.environ.get("AUDIO_TIMEOUT", "15"))
AUDIO_MAX_SEC = float(os.environ.get("AUDIO_MAX_SEC", "120"))
# Gemma styling gets STYLE_GRACE from its actual HTTP start; a Kimi backup races
# in parallel from styling start, so a long grace risks nothing — Gemma is preferred
# at the wire, the banked Kimi result stands otherwise. Free-tier Gemma latency
# swings ~7s (calm) to 36s+ (congested, measured 2026-07-09) — grace must span both.
STYLE_GRACE = float(os.environ.get("STYLE_GRACE", "38"))
KIMI_STYLE_TIMEOUT = float(os.environ.get("KIMI_STYLE_TIMEOUT", "9"))
BACKUP_RESERVE = float(os.environ.get("BACKUP_RESERVE", "10"))    # time reserved for the Kimi lane
GEMMA_MIN_GRACE = float(os.environ.get("GEMMA_MIN_GRACE", "7"))   # below this, don't even try Gemma
JUDGE_TIMEOUT = float(os.environ.get("JUDGE_TIMEOUT", "10"))
CRITIQUE_TIMEOUT = float(os.environ.get("CRITIQUE_TIMEOUT", "10"))

# --- Gemma decoding ---
GEMMA_MAX_TOKENS = int(os.environ.get("GEMMA_MAX_TOKENS", "2600"))  # directed thinking ~150 + answer; cap doubles as runaway-thought guard
# Attach keyframes to the Gemma styling call (gemma-4-31b is multimodal — verified).
# Frames ALWAYS ride along when grounding failed (facts empty — rescue mode).
# GEMMA_VISION=1 additionally sends frames alongside good facts, but a 3-image call
# measured ~30s on the free tier vs ~10s text-only — it clogs the serial lane, so
# default OFF; facts carry the visual grounding on the fast path.
GEMMA_VISION = _flag("GEMMA_VISION", "0")
GEMMA_VISION_FRAMES = int(os.environ.get("GEMMA_VISION_FRAMES", "3"))
GEMMA_TEMP_GEN = float(os.environ.get("GEMMA_TEMP_GEN", "0.9"))
GEMMA_TEMP_JUDGE = float(os.environ.get("GEMMA_TEMP_JUDGE", "0.0"))

# Caption length band (verbosity-bias defense; RESEARCH.md §6)
WORD_MIN = int(os.environ.get("WORD_MIN", "20"))
WORD_MAX = int(os.environ.get("WORD_MAX", "45"))
