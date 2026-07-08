"""Central configuration. Everything overridable via env so leaderboard A/B needs no code edits."""
import os

def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")

# --- Secrets (never hardcode) ---
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# --- Endpoints ---
FIREWORKS_BASE = "https://api.fireworks.ai/inference/v1"
GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"

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
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "10"))
MIN_FRAMES = int(os.environ.get("MIN_FRAMES", "8"))
FRAME_LONG_EDGE = int(os.environ.get("FRAME_LONG_EDGE", "1024"))
FRAME_JPEG_Q = int(os.environ.get("FRAME_JPEG_Q", "4"))  # ffmpeg -q:v scale (2..6; lower=better)

# --- Concurrency & budgets (seconds) ---
CLIP_CONCURRENCY = int(os.environ.get("CLIP_CONCURRENCY", "3"))
# Gemini free tier 500s on ANY concurrent Gemma calls (verified live) → serialize + gap
GEMMA_CONCURRENCY = int(os.environ.get("GEMMA_CONCURRENCY", "1"))
GEMMA_MIN_GAP = float(os.environ.get("GEMMA_MIN_GAP", "0.8"))
# Stagger successive clip starts so each clip reaches styling just as the serial
# Gemma lane frees up (Gemma call ≈ 11.5s with the brief-thinking prompt)
CLIP_STAGGER = float(os.environ.get("CLIP_STAGGER", "12"))
CLIP_DEADLINE = float(os.environ.get("CLIP_DEADLINE", "27"))       # hard wall < 30s contract (backstop +2)
DOWNLOAD_TIMEOUT = float(os.environ.get("DOWNLOAD_TIMEOUT", "10"))
GROUNDING_TIMEOUT = float(os.environ.get("GROUNDING_TIMEOUT", "9"))
AUDIO_TIMEOUT = float(os.environ.get("AUDIO_TIMEOUT", "6"))
# Gemma styling: free-tier gemma-4-31b measured ~17.5s for the 4-style call (thinking
# is unstrippable). It gets STYLE_GRACE from its start; if it hasn't landed, a fast
# Kimi backup fires and whichever valid result exists at the wire wins (Gemma preferred).
STYLE_GRACE = float(os.environ.get("STYLE_GRACE", "19"))
KIMI_STYLE_TIMEOUT = float(os.environ.get("KIMI_STYLE_TIMEOUT", "8"))
BACKUP_RESERVE = float(os.environ.get("BACKUP_RESERVE", "6.5"))   # time reserved for the Kimi lane
GEMMA_MIN_GRACE = float(os.environ.get("GEMMA_MIN_GRACE", "5.5")) # below this, don't even try Gemma
JUDGE_TIMEOUT = float(os.environ.get("JUDGE_TIMEOUT", "8"))
CRITIQUE_TIMEOUT = float(os.environ.get("CRITIQUE_TIMEOUT", "8"))

# --- Gemma decoding ---
GEMMA_MAX_TOKENS = int(os.environ.get("GEMMA_MAX_TOKENS", "2000"))  # brief-thinking prompt: ~90 thought + ~160 answer; cap doubles as runaway-thought guard
GEMMA_TEMP_GEN = float(os.environ.get("GEMMA_TEMP_GEN", "0.9"))
GEMMA_TEMP_JUDGE = float(os.environ.get("GEMMA_TEMP_JUDGE", "0.0"))

# Caption length band (verbosity-bias defense; RESEARCH.md §6)
WORD_MIN = int(os.environ.get("WORD_MIN", "20"))
WORD_MAX = int(os.environ.get("WORD_MAX", "45"))
