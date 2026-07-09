"""Build a realistic eval set: 30s-2min videos across the 8 eval categories.

Queries Wikimedia Commons for freely-licensed videos, filters by duration,
and writes eval/tasks_long.json with DIRECT remote URLs (so the run exercises
real download timing, like the hidden leaderboard eval does).

Usage: python eval/fetch_eval_clips.py
"""
import json
import os
import sys
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# Mirror the 8 hidden-eval categories from RESEARCH.md
CATEGORY_QUERIES = {
    "nature_wildlife": "filetype:video incategory:\"Videos of birds\"",
    "urban_street": "filetype:video incategory:\"Videos of traffic\"",
    "sports": "filetype:video incategory:\"Videos of sports\"",
    "food_cooking": "filetype:video incategory:\"Videos of cooking\"",
    "technology": "filetype:video incategory:\"Videos of computers\"",
    "music_performance": "filetype:video incategory:\"Videos of music\"",
    "people_talking": "filetype:video incategory:\"Videos of people speaking\"",
    "vehicles": "filetype:video incategory:\"Videos of trains\"",
    # extra breadth
    "animals_other": "filetype:video incategory:\"Videos of cats\"",
    "water_boats": "filetype:video incategory:\"Videos of boats\"",
    "aerial_drone": "filetype:video aerial drone city",
    "crafts_hands": "filetype:video incategory:\"Videos of crafts\"",
}

MIN_DUR, MAX_DUR = 25.0, 135.0
MAX_BYTES = 80 * 1024 * 1024  # keep downloads sane


def api(params: dict) -> dict:
    base = "https://commons.wikimedia.org/w/api.php"
    qs = urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(
        f"{base}?{qs}",
        headers={"User-Agent": "amd-hackathon-eval/1.0 (contact: dev)"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def find_clip(query: str):
    """Return (title, url, duration, size) of the first video matching duration band."""
    data = api({
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6",  # File:
        "gsrlimit": "40",
        "prop": "videoinfo",
        "viprop": "url|size|mediatype|extmetadata",
    })
    pages = (data.get("query") or {}).get("pages") or {}
    candidates = []
    for p in pages.values():
        vi = (p.get("videoinfo") or [{}])[0]
        url = vi.get("url", "")
        if not url.lower().endswith((".webm", ".mp4", ".ogv", ".mov")):
            continue
        size = vi.get("size") or 0
        if size > MAX_BYTES:
            continue
        # duration comes via extmetadata sometimes; fall back to imageinfo duration key
        dur = None
        for k in ("duration",):
            if k in vi:
                dur = float(vi[k])
        if dur is None:
            # videoinfo usually includes 'duration' at top level of vi; if not, skip precise filter
            continue
        if MIN_DUR <= dur <= MAX_DUR:
            candidates.append((p.get("title", "?"), url, dur, size))
    # prefer mid-length
    candidates.sort(key=lambda c: abs(c[2] - 60))
    return candidates[0] if candidates else None


def main() -> int:
    tasks = []
    meta = []
    for cat, q in CATEGORY_QUERIES.items():
        try:
            hit = find_clip(q)
        except Exception as e:
            print(f"[{cat}] API error: {e}", file=sys.stderr)
            hit = None
        if not hit:
            print(f"[{cat}] no clip in band")
            continue
        title, url, dur, size = hit
        tid = f"eval_{cat}"
        tasks.append({
            "task_id": tid,
            "video_url": url,
            "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"],
        })
        meta.append({"task_id": tid, "title": title, "duration": dur, "mb": round(size / 1e6, 1)})
        print(f"[{cat}] {dur:.0f}s {size/1e6:.1f}MB {title}")
        if len(tasks) >= 12:
            break

    with open(os.path.join(HERE, "tasks_long.json"), "w") as f:
        json.dump(tasks, f, indent=1)
    with open(os.path.join(HERE, "clips_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"\nWrote {len(tasks)} tasks to eval/tasks_long.json")
    return 0 if len(tasks) >= 8 else 1


if __name__ == "__main__":
    sys.exit(main())
