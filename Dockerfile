# Build with: docker buildx build --platform linux/amd64 ...
FROM python:3.11-slim

# ffmpeg for frame/audio extraction; libgomp1 for onnxruntime (RapidOCR)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake RapidOCR ONNX models into the image (no network at inference time)
RUN python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()" || true

COPY main.py .
COPY pipeline ./pipeline

# All inference via API; no weights in image. Secrets via env at runtime:
#   FIREWORKS_API_KEY, GEMINI_API_KEY
ENTRYPOINT ["python", "main.py"]
