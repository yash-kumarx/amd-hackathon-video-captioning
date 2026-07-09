"""Fetch 30s-2min public-domain clips from archive.org into eval/clips/ as a
Wikimedia-independent eval source (archive.org serves direct URLs politely).

Usage: python3 eval/fetch_archive_clips.py
Appends any new clips to eval/tasks_long.json style task list -> eval/tasks_archive.json
"""
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

# Queries chosen for variety of visual content
QUERIES = {
    "arch_cooking": 'mediatype:movies AND subject:"cooking"',
    "arch_sports": 'mediatype:movies AND subject:"sports"',
    "arch_nature": 'mediatype:movies AND subject:"wildlife"',
    "arch_city": 'mediatype:movies AND subject:"city"',
    "arch_trains": 'mediatype:movies AND subject:"railroad"',
    "arch_dance": 'mediatype:movies AND subject:"dance"',
    "arch_science": 'mediatype:movies AND subject:"chemistry"',
    "arch_aviation": 'mediatype:movies AND subject:"aviation"',
}
MIN_DUR, MAX_DUR = 25.0, 135.0
MAX_MB = 70


def jget(url):
    req = urllib.request.Request(url, headers={"User-Agent": "amd-hackathon-eval/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def find_file(identifier):
    meta = jget(f"https://archive.org/metadata/{identifier}")
    for f in meta.get("files", []):
        name = f.get("name", "")
        if not name.lower().endswith((".mp4", ".m4v")):
            continue
        try:
            dur = float(f.get("length") or 0)
            size = int(f.get("size") or 0)
        except (TypeError, ValueError):
            continue
        if MIN_DUR <= dur <= MAX_DUR and 0 < size <= MAX_MB * 1024 * 1024:
            return name, dur, size
    return None


def main():
    tasks = []
    for tid, q in QUERIES.items():
        try:
            res = jget("https://archive.org/advancedsearch.php?" + urllib.parse.urlencode({
                "q": q, "fl[]": "identifier", "rows": "15", "output": "json",
                "sort[]": "downloads desc"}))
        except Exception as e:
            print(f"[{tid}] search failed: {e}")
            continue
        docs = res.get("response", {}).get("docs", [])
        got = False
        for d in docs:
            ident = d["identifier"]
            try:
                hit = find_file(ident)
            except Exception:
                continue
            if not hit:
                continue
            name, dur, size = hit
            url = f"https://archive.org/download/{ident}/{urllib.parse.quote(name)}"
            dest = os.path.join(HERE, "clips", f"{tid}.mp4")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            r = subprocess.run(["curl", "-sL", "--max-time", "300", "-o", dest, url])
            if r.returncode == 0 and os.path.getsize(dest) > 200000:
                print(f"[{tid}] {dur:.0f}s {size/1e6:.0f}MB {ident}/{name}")
                tasks.append({"task_id": tid, "video_url": url, "styles": STYLES})
                got = True
                break
        if not got:
            print(f"[{tid}] nothing suitable")
    json.dump(tasks, open(os.path.join(HERE, "tasks_archive.json"), "w"), indent=1)
    print(f"wrote {len(tasks)} archive tasks")


if __name__ == "__main__":
    main()
