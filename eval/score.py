"""Judge-replica scorer: grade a results.json 0-1 per caption, like the AMD leaderboard.

Two independent vision judges (Kimi K2.6 on Fireworks + gemini-2.5-flash on Gemini API),
each sees 6 keyframes of the actual video plus the caption set, scores every style on:
  accuracy   — is the caption factually true to the video?
  style      — does it nail the requested style register?
  quality    — fluency, specificity, engagement (is it a GOOD caption?)
Overall = mean of the three, averaged across judges. Aggregates → eval/scores_<tag>.json

Usage:
  python3 eval/score.py eval/out_baseline/results.json --tag baseline
Requires .env sourced (FIREWORKS_API_KEY, GEMINI_API_KEY). Clips + frames are cached
in eval/clips/ and eval/frames/ on first run.
"""
import argparse
import asyncio
import base64
import json
import os
import re
import subprocess
import sys
import time

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

RUBRICS = {
    "formal": "Neutral, precise, documentary register. Third-person, no slang, no jokes.",
    "sarcastic": "Dry, ironic understatement or mock-praise; deadpan; humor from incongruity; never breaks facts.",
    "humorous_tech": "Playful humor built on tech/programming metaphors that map onto what happens in the video.",
    "humorous_non_tech": "Warm, everyday relatable humor, witty-friend narration, zero tech jargon.",
}

# Mirrors the REAL Track 2 rubric (Participant Guide): exactly two axes.
JUDGE_PROMPT = """You are the automated judge for a video-captioning contest. You see keyframes from a video and four captions, each targeting a labeled style.

Score EVERY caption on exactly two axes, each 0.0-1.0:
- accuracy: how faithfully the caption reflects the video content (hallucinated subjects/objects/events = heavy penalty; vague-but-true is mid; specific and true is high)
- style: how well the caption matches the requested tone (rubrics below)

Be strict: reserve >0.9 for captions that are both specific and flawless in register.
STYLE RUBRICS: %s

CAPTIONS: %s

Return STRICT JSON only:
{"formal": {"accuracy": x, "style": x}, "sarcastic": {...}, "humorous_tech": {...}, "humorous_non_tech": {...}}"""


def sh(cmd: list) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout


def ensure_frames(task_id: str, url: str, n: int = 6):
    """Download clip once, extract n jpeg frames at 768px. Returns list of b64 strings."""
    clip_dir = os.path.join(HERE, "clips")
    frames_dir = os.path.join(HERE, "frames", task_id)
    os.makedirs(clip_dir, exist_ok=True)
    ext = os.path.splitext(url)[1] or ".mp4"
    vid = os.path.join(clip_dir, task_id + ext)
    if not os.path.exists(vid):
        subprocess.run(["curl", "-sL", "--max-time", "180", "-o", vid, url], check=True)
    if not os.path.isdir(frames_dir) or len(os.listdir(frames_dir)) < n:
        os.makedirs(frames_dir, exist_ok=True)
        dur = float(sh(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", vid]).strip() or "30")
        for i in range(n):
            t = dur * (i + 0.5) / n
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", vid,
                 "-frames:v", "1", "-vf", "scale='min(768,iw)':-2", "-q:v", "5",
                 os.path.join(frames_dir, f"f{i}.jpg")],
                check=False, timeout=60)
    out = []
    for f in sorted(os.listdir(frames_dir)):
        with open(os.path.join(frames_dir, f), "rb") as fh:
            out.append(base64.b64encode(fh.read()).decode())
    return out[:n]


def extract_json(text: str):
    text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.S)
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


_flash_sem = asyncio.Semaphore(1)  # gemini free tier 429s on concurrent judge calls
_flash_last = [0.0]                 # global min-gap pacing for the gemini judge
_FLASH_GAP = 4.0


async def call_judge(client, base, key, model, frames_b64, captions, extra=None):
    parts = [{"type": "text", "text": JUDGE_PROMPT % (
        json.dumps(RUBRICS), json.dumps(captions, ensure_ascii=False))}]
    for b in frames_b64:
        parts.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b}})
    payload = {"model": model, "max_tokens": 1200, "temperature": 0.0,
               "messages": [{"role": "user", "content": parts}]}
    if extra:
        payload.update(extra)
    serial = "generativelanguage" in base
    import time as _t
    for attempt in range(5):
        if serial:
            async with _flash_sem:
                wait = _FLASH_GAP - (_t.monotonic() - _flash_last[0])
                if wait > 0:
                    await asyncio.sleep(wait)
                r = await client.post(f"{base}/chat/completions",
                                      headers={"Authorization": f"Bearer {key}"},
                                      json=payload, timeout=90)
                _flash_last[0] = _t.monotonic()
        else:
            r = await client.post(f"{base}/chat/completions",
                                  headers={"Authorization": f"Bearer {key}"},
                                  json=payload, timeout=90)
        if r.status_code == 429 and attempt < 4:
            await asyncio.sleep(6.0 * (attempt + 1))
            continue
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return extract_json(content)
    return None


async def score_clip(client, task_id, url, captions):
    frames = await asyncio.to_thread(ensure_frames, task_id, url)
    fw_key = os.environ["FIREWORKS_API_KEY"]
    gm_key = os.environ["GEMINI_API_KEY"]
    judges = await asyncio.gather(
        call_judge(client, "https://api.fireworks.ai/inference/v1", fw_key,
                   "accounts/fireworks/models/kimi-k2p6", frames, captions,
                   extra={"reasoning_effort": "none"}),
        call_judge(client, "https://generativelanguage.googleapis.com/v1beta/openai",
                   gm_key, os.environ.get("FLASH_JUDGE_MODEL", "gemini-2.5-flash"), frames, captions),
        return_exceptions=True,
    )
    per_style = {}
    for s in STYLES:
        axes = {"accuracy": [], "style": []}
        for j in judges:
            if isinstance(j, dict) and isinstance(j.get(s), dict):
                for a in axes:
                    v = j[s].get(a)
                    if isinstance(v, (int, float)):
                        axes[a].append(max(0.0, min(1.0, float(v))))
        if all(axes.values()):
            m = {a: sum(v) / len(v) for a, v in axes.items()}
            m["overall"] = sum(m.values()) / 2
            per_style[s] = m
        else:
            per_style[s] = {"accuracy": None, "style": None, "overall": None}
    errs = [str(j)[:100] for j in judges if not isinstance(j, dict)]
    return per_style, errs


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--tasks", default=os.path.join(HERE, "tasks_long.json"))
    args = ap.parse_args()

    results = {r["task_id"]: r["captions"] for r in json.load(open(args.results))}
    tasks = {t["task_id"]: t["video_url"] for t in json.load(open(args.tasks))}

    t0 = time.time()
    out = {"tag": args.tag, "results_file": args.results, "clips": {}, "aggregate": {}}
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(2)

        async def one(tid):
            async with sem:
                per_style, errs = await score_clip(client, tid, tasks[tid], results[tid])
                overall = [v["overall"] for v in per_style.values() if v["overall"] is not None]
                clip_score = sum(overall) / len(overall) if overall else None
                out["clips"][tid] = {"score": clip_score, "styles": per_style, "judge_errors": errs}
                print(f"[{tid}] {clip_score if clip_score is None else round(clip_score,3)} {errs or ''}")

        await asyncio.gather(*(one(tid) for tid in results if tid in tasks))

    scores = [c["score"] for c in out["clips"].values() if c["score"] is not None]
    agg = {"mean": round(sum(scores) / len(scores), 4) if scores else None,
           "n": len(scores),
           "min": round(min(scores), 3) if scores else None,
           "max": round(max(scores), 3) if scores else None}
    for axis in ("accuracy", "style"):
        vals = [st[axis] for c in out["clips"].values() for st in c["styles"].values()
                if st[axis] is not None]
        agg[axis] = round(sum(vals) / len(vals), 4) if vals else None
    per_style_mean = {}
    for s in STYLES:
        vals = [c["styles"][s]["overall"] for c in out["clips"].values()
                if c["styles"].get(s, {}).get("overall") is not None]
        per_style_mean[s] = round(sum(vals) / len(vals), 4) if vals else None
    agg["per_style"] = per_style_mean
    out["aggregate"] = agg

    path = os.path.join(HERE, f"scores_{args.tag}.json")
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print(f"\n== {args.tag} == mean={agg['mean']} accuracy={agg['accuracy']} "
          f"style={agg['style']}\nper-style: {per_style_mean}"
          f"\n({time.time()-t0:.0f}s) -> {path}")


if __name__ == "__main__":
    asyncio.run(main())
