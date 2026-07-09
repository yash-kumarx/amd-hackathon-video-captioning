"""Rewrite tasks_long.json URLs to a local HTTP server serving eval/clips/.

Run the server separately:  python3 -m http.server 8077 -d eval/clips
Then:                       python3 eval/make_local_tasks.py
Writes eval/tasks_local.json (same task_ids, local URLs) — download timing then
mirrors GCS-in-datacenter (fast LAN pull) instead of throttled Wikimedia.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

tasks = json.load(open(os.path.join(HERE, "tasks_long.json")))
out = []
for t in tasks:
    ext = os.path.splitext(t["video_url"])[1] or ".mp4"
    fname = t["task_id"] + ext
    if not os.path.exists(os.path.join(HERE, "clips", fname)):
        print(f"MISSING local clip: {fname} — skipped")
        continue
    out.append({**t, "video_url": f"http://127.0.0.1:8077/{fname}"})
json.dump(out, open(os.path.join(HERE, "tasks_local.json"), "w"), indent=1)
print(f"wrote {len(out)} local tasks")
