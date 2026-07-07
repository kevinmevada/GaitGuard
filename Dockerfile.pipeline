FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONHASHSEED=42

WORKDIR /app

# System deps needed by scientific Python stack and git-based installs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# REP-010: pinned lockfiles (aligned with CI).
COPY fall_risk_pipeline/requirements-lock.txt /tmp/fall_risk_requirements-lock.txt
COPY api/requirements-inference.txt /tmp/api_inference_requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/fall_risk_requirements-lock.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install -r /tmp/api_inference_requirements.txt

# Copy repository.
COPY . /app

# Default command runs full 15-stage pipeline.
# For API deployment use: docker build -f Dockerfile.api -t gaitguard-api .
WORKDIR /app/fall_risk_pipeline
CMD ["python", "main.py", "--config", "configs/pipeline_config.yaml"]
