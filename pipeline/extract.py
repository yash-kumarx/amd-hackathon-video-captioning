"""Video download + ffprobe + keyframe/audio extraction. Local ffmpeg only."""
import asyncio
import base64
import json
import logging
import math
import os
import shutil
from typing import Dict, List, Optional, Tuple

import httpx

from . import config

log = logging.getLogger("pipeline.extract")


# Some hosts (e.g. Wikimedia) 403 requests without a real User-Agent
_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}


async def download_video(client: httpx.AsyncClient, url: str, dest: str) -> str:
    tmp = dest + ".part"
    last_exc: Optional[Exception] = None
    for attempt in range(2):
        try:
            async with client.stream("GET", url, timeout=config.DOWNLOAD_TIMEOUT,
                                     follow_redirects=True, headers=_UA) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    async for chunk in r.aiter_bytes(1 << 20):
                        f.write(chunk)
            os.replace(tmp, dest)
            return dest
        except Exception as e:
            last_exc = e
            if attempt == 0:
                log.warning("download attempt 1 failed (%s) — retrying", str(e)[:100])
                await asyncio.sleep(0.5)
    raise last_exc if last_exc else RuntimeError("download failed")


async def _run(cmd: List[str], timeout: float) -> Tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise
    return proc.returncode or 0, out, err


async def probe(path: str) -> Dict:
    """duration (s), has_audio, width/height."""
    code, out, _ = await _run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", path],
        timeout=10,
    )
    info = {"duration": 0.0, "has_audio": False, "width": 0, "height": 0}
    if code != 0:
        return info
    data = json.loads(out.decode() or "{}")
    try:
        info["duration"] = float(data.get("format", {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        pass
    for s in data.get("streams", []):
        if s.get("codec_type") == "audio":
            info["has_audio"] = True
        elif s.get("codec_type") == "video":
            info["width"] = s.get("width") or 0
            info["height"] = s.get("height") or 0
    return info


def _frame_count(duration: float) -> int:
    if duration <= 0:
        return config.MIN_FRAMES
    # ~1 frame / 8s of video, clamped
    return max(config.MIN_FRAMES, min(config.MAX_FRAMES, int(math.ceil(duration / 8.0)) + 4))


async def extract_frames(path: str, out_dir: str, duration: float) -> List[Dict]:
    """Uniformly sampled frames at ≤ FRAME_LONG_EDGE px. Returns [{t, path, b64}]."""
    os.makedirs(out_dir, exist_ok=True)
    n = _frame_count(duration)
    # Midpoint sampling: avoids black first/last frames
    times = [duration * (i + 0.5) / n for i in range(n)] if duration > 0 else [0.0]
    vf = f"scale='if(gt(iw,ih),min({config.FRAME_LONG_EDGE},iw),-2)':'if(gt(iw,ih),-2,min({config.FRAME_LONG_EDGE},ih))'"

    # Long clips: decode only keyframes (~10x faster on UHD sources; sampled frames
    # snap to the nearest keyframe, which is fine at 25s+ durations). Short clips may
    # have very few keyframes, so they keep full decode.
    skip = ["-skip_frame", "nokey"] if duration > 25 else []

    async def grab(i: int, t: float) -> Optional[Dict]:
        fp = os.path.join(out_dir, f"f{i:02d}.jpg")
        code, _, err = await _run(
            ["ffmpeg", "-y", "-loglevel", "error", *skip, "-ss", f"{t:.3f}", "-i", path,
             "-frames:v", "1", "-vf", vf, "-q:v", str(config.FRAME_JPEG_Q), fp],
            timeout=8,
        )
        if code != 0 or not os.path.exists(fp) or os.path.getsize(fp) == 0:
            log.warning("frame %d at %.1fs failed: %s", i, t, err.decode()[:120])
            return None
        with open(fp, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return {"t": round(t, 1), "path": fp, "b64": b64}

    results = await asyncio.gather(*(grab(i, t) for i, t in enumerate(times)))
    frames = [r for r in results if r]
    if not frames:
        raise RuntimeError("no frames extracted")
    return frames


async def extract_audio(path: str, out_dir: str) -> Optional[str]:
    """16k mono wav for ASR; None if no audio track or failure."""
    wav = os.path.join(out_dir, "audio.wav")
    code, _, err = await _run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-vn",
         "-t", str(config.AUDIO_MAX_SEC),
         "-ac", "1", "-ar", "16000", "-f", "wav", wav],
        timeout=15,
    )
    if code != 0 or not os.path.exists(wav) or os.path.getsize(wav) < 1000:
        return None
    return wav


def cleanup(workdir: str) -> None:
    shutil.rmtree(workdir, ignore_errors=True)
