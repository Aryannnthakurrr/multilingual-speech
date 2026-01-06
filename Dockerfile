FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Install system dependencies in one layer
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

WORKDIR /app

# Create venv first (this layer gets cached)
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# Install ALL Python dependencies in ONE layer to avoid cache issues
RUN uv pip install --no-cache-dir \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 && \
    uv pip install --no-cache-dir \
    faster-whisper \
    huggingface-hub \
    edge-tts \
    requests

# Copy app code last (so code changes don't trigger reinstall)
COPY . /app

# Create output directory
RUN mkdir -p /app/output

# Default command - pass audio file as argument when running
CMD ["python3", "voice_assistant.py", "test_audio/Hindi_test.m4a"]
