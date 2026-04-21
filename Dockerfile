# ForensicAI Docker Image
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create data directories
RUN mkdir -p data/uploads data/results data/models data/sample_images web/assets

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run
ENV FORENSICAI_HOST=0.0.0.0
ENV FORENSICAI_PORT=8000
ENV FORENSICAI_DEBUG=false

CMD ["python", "run.py"]
