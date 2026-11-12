# Oche Lab Motion API container (Render, Hugging Face Spaces Docker SDK, Fly, Cloud Run, ...).
# On small hosts (Render free: 512 MB, 0.1 CPU) set LOW_MEM=1 for the lite pose model.
FROM python:3.12-slim

# libgl1/libglib2.0-0: OpenCV + MediaPipe runtime libs. ffmpeg: H.264 transcode for the overlay MP4.
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
 && apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as uid 1000; other hosts work fine with it too.
RUN useradd -m -u 1000 app
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && python -c "import mediapipe as mp; [mp.solutions.pose.Pose(model_complexity=c).close() for c in (0, 1)]"
# ^ MediaPipe downloads the lite pose model on first use; fetch it at build time because
#   site-packages is read-only for the runtime user.

COPY --chown=app:app *.py ./
USER app

ENV PYTHONUNBUFFERED=1 \
    OCHELAB_RESULTS_DIR=/tmp/ochelab_results \
    PORT=7860
EXPOSE 7860

# Hosts inject $PORT (Render, Cloud Run, Koyeb); Spaces uses app_port 7860.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers --forwarded-allow-ips='*'"]
