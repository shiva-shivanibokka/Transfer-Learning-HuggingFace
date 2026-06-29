# Hugging Face Spaces (Docker SDK) — CPU inference image for the Gradio demo.
FROM python:3.12-slim

# System libs needed by Pillow / torchvision image decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Non-root user (HF Spaces convention: uid 1000).
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    HF_HOME=/home/user/.cache/huggingface \
    LOG_LEVEL=INFO \
    PORT=7860

WORKDIR /home/user/app

# Install slim inference deps first so this layer caches across code changes.
COPY --chown=user requirements-app.txt .
RUN pip install --no-cache-dir --user -r requirements-app.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["python", "app/gradio_app.py"]
