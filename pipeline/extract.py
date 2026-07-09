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
    """Single attempt with a hard wall-clock cap — on any failure the caller streams
    frames straight from the URL, which is the better use of the remaining budget."""
    tmp = dest + ".part"

    async def _dl() -> None:
        async with client.stream("GET", url, timeout=config.DOWNLOAD_TIMEOUT,
                                 follow_redirects=True, headers=_UA) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                async for chunk in r.aiter_bytes(1 << 20):
                    f.write(chunk)

    try:
        await asyncio.wait_for(_dl(), timeout=config.DOWNLOAD_TIMEOUT)
    except (asyncio.TimeoutError, Exception):
        # Salvage: stock MP4s are faststart (moov up front), so a partial file still
        # decodes fine for frames up to the truncation point.
        if os.path.exists(tmp) and os.path.getsize(tmp) > 2_000_000:
            log.warning("download timed out — using %.0fMB partial file",
                        os.path.getsize(tmp) / 1e6)
            os.replace(tmp, dest)
            return dest
        raise
    os.replace(tmp, dest)
    return dest


# Eval runners are CPU-starved (1-2 vCPUs): an unbounded ffmpeg storm blows the 30s
# per-clip window (measured: 6s clip took 11.9s to extract at --cpus=2). All ffmpeg/
# ffprobe work funnels through this semaphore, single-threaded per process.
_FFMPEG_SEM: Optional[asyncio.Semaphore] = None


def _ffsem() -> asyncio.Semaphore:
    global _FFMPEG_SEM
    if _FFMPEG_SEM is None:
        _FFMPEG_SEM = asyncio.Semaphore(int(os.environ.get("FFMPEG_PROCS", "2")))
    return _FFMPEG_SEM


_FF_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"


def _input_args(src: str) -> List[str]:
    """ffmpeg/ffprobe input flags: for http(s) sources, range-request directly from the
    URL (a frame grab pulls ~1-3MB instead of downloading the whole 10-90MB file)."""
    if src.startswith("http"):
        return ["-user_agent", _FF_UA, "-i", src]
    return ["-i", src]


async def _run(cmd: List[str], timeout: float) -> Tuple[int, bytes, bytes]:
    async with _ffsem():
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
    """duration (s), has_audio, width/height. Works on URLs (ranged request)."""
    src = (["-user_agent", _FF_UA, path] if path.startswith("http") else [path])
    code, out, _ = await _run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", *src],
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
    if duration <= 10:
        return 4  # short stock shots barely change; 4 frames = enough signal, half the decode
    # ~1 frame / 8s of video, clamped
    return max(config.MIN_FRAMES, min(config.MAX_FRAMES, int(math.ceil(duration / 8.0)) + 4))


async def extract_frames(path: str, out_dir: str, duration: float) -> List[Dict]:
    """Uniformly sampled frames at ≤ FRAME_LONG_EDGE px. Returns [{t, path, b64}]."""
    os.makedirs(out_dir, exist_ok=True)
    n = _frame_count(duration)
    # Midpoint sampling: avoids black first/last frames
    times = [duration * (i + 0.5) / n for i in range(n)] if duration > 0 else [0.0]
    vf = f"scale='if(gt(iw,ih),min({config.FRAME_LONG_EDGE},iw),-2)':'if(gt(iw,ih),-2,min({config.FRAME_LONG_EDGE},ih))'"

    # Decode only keyframes (~10x faster on UHD sources; stock footage has keyframes
    # every 1-4s so sampled frames just snap to a neighbor). Only very short clips
    # (few keyframes total) keep full decode.
    skip = ["-skip_frame", "nokey"] if duration > 10 else []

    async def grab(i: int, t: float) -> Optional[Dict]:
        fp = os.path.join(out_dir, f"f{i:02d}.jpg")
        code, _, err = await _run(
            ["ffmpeg", "-y", "-loglevel", "error", "-threads", "1", *skip,
             "-ss", f"{t:.3f}", *_input_args(path),
             "-frames:v", "1", "-vf", vf, "-q:v", str(config.FRAME_JPEG_Q), fp],
            timeout=9,
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
        ["ffmpeg", "-y", "-loglevel", "error", "-threads", "1", *_input_args(path), "-vn",
         "-t", str(config.AUDIO_MAX_SEC),
         "-ac", "1", "-ar", "16000", "-f", "wav", wav],
        timeout=12,
    )
    if code != 0 or not os.path.exists(wav) or os.path.getsize(wav) < 1000:
        return None
    return wav


def cleanup(workdir: str) -> None:
    shutil.rmtree(workdir, ignore_errors=True)
