FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps needed by scientific Python stack and git-based installs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching.
COPY fall_risk_pipeline/requirements.txt /tmp/fall_risk_requirements.txt
COPY api/requirements.txt /tmp/api_requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/fall_risk_requirements.txt && \
    pip install -r /tmp/api_requirements.txt

# Copy repository.
COPY . /app

# Default command runs full 15-stage pipeline.
WORKDIR /app/fall_risk_pipeline
CMD ["python", "main.py", "--config", "configs/pipeline_config.yaml"]
