FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    EXPERIMENT_MODE=full \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY requirements_lock.txt /app/requirements_lock.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && pip install --no-cache-dir -r /app/requirements_lock.txt \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

CMD ["python", "run_experiments.py"]
