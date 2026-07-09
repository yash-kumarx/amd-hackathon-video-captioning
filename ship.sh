#!/usr/bin/env bash
# Ship the current code to the leaderboard tag. Usage: ./ship.sh [tag]
# Builds linux/amd64, smoke-tests the container contract on 3 real bucket clips,
# pushes to docker.io/yashr1704/amd-video-captioner:<tag> (default: baseline).
set -euo pipefail
cd "$(dirname "$0")"
TAG="${1:-baseline}"
IMG="docker.io/yashr1704/amd-video-captioner:${TAG}"

echo "== build =="
docker buildx build --platform linux/amd64 -t "$IMG" --load .

echo "== container smoke test =="
set -a; source .env; set +a
rm -rf container_io && mkdir -p container_io/input container_io/output
cat > container_io/input/tasks.json <<'EOF'
[
 {"task_id": "t1", "video_url": "https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4", "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]},
 {"task_id": "t2", "video_url": "https://storage.googleapis.com/amd-hackathon-clips/12471596-uhd_2560_1440_30fps.mp4", "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]},
 {"task_id": "t3", "video_url": "https://storage.googleapis.com/amd-hackathon-clips/31948459-hd_1920_1080_24fps.mp4", "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]}
]
EOF
docker run --rm --platform linux/amd64 \
  -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  ${GEMINI_API_KEY_2:+-e GEMINI_API_KEY_2="$GEMINI_API_KEY_2"} \
  ${OPENROUTER_API_KEY:+-e OPENROUTER_API_KEY="$OPENROUTER_API_KEY"} \
  -v "$PWD/container_io/input:/input:ro" -v "$PWD/container_io/output:/output" \
  "$IMG"
python3 - <<'EOF'
import json, sys
r = json.load(open("container_io/output/results.json"))
assert len(r) == 3, f"expected 3 results, got {len(r)}"
for item in r:
    caps = item["captions"]
    for s in ("formal", "sarcastic", "humorous_tech", "humorous_non_tech"):
        assert caps.get(s, "").strip(), f"{item['task_id']} missing {s}"
print("contract OK: 3 clips x 4 styles, all non-empty")
EOF

echo "== push =="
docker push "$IMG"
echo "shipped $IMG"
