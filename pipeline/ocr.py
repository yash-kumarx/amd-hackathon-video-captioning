"""Local OCR on keyframes via RapidOCR (ONNX). Fully optional: absent lib → skip."""
import asyncio
import logging
from typing import List

log = logging.getLogger("pipeline.ocr")

_engine = None
_available = None


def _get_engine():
    global _engine, _available
    if _available is False:
        return None
    if _engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
            _engine = RapidOCR()
            _available = True
        except Exception as e:
            log.warning("RapidOCR unavailable, skipping OCR: %s", str(e)[:100])
            _available = False
            return None
    return _engine


def _ocr_one(path: str) -> List[str]:
    eng = _get_engine()
    if eng is None:
        return []
    try:
        result, _ = eng(path)
        if not result:
            return []
        # result: [[box, text, score], ...]
        return [r[1].strip() for r in result if len(r) >= 3 and r[2] > 0.65 and r[1].strip()]
    except Exception as e:
        log.warning("ocr failed on %s: %s", path, str(e)[:100])
        return []


async def ocr_frames(frame_paths: List[str], timeout: float = 6.0) -> List[str]:
    """Dedup'd on-screen text across sampled frames (subset for speed)."""
    if _get_engine() is None:
        return []
    # OCR alternate frames — text rarely changes frame-to-frame at our sampling rate
    subset = frame_paths[::2] if len(frame_paths) > 6 else frame_paths

    def run_all() -> List[str]:
        seen, out = set(), []
        for p in subset:
            for t in _ocr_one(p):
                k = t.lower()
                if k not in seen:
                    seen.add(k)
                    out.append(t)
        return out[:25]

    try:
        return await asyncio.wait_for(asyncio.to_thread(run_all), timeout=timeout)
    except asyncio.TimeoutError:
        log.warning("OCR timed out; proceeding without")
        return []
    except Exception as e:
        log.warning("OCR error: %s", str(e)[:100])
        return []
