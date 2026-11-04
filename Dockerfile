FROM python:3.11-slim

# ffmpeg for the post-call audio cut; the pango/cairo stack is what WeasyPrint needs for
# the evidence pack. Both live in the image so the host machine does not need them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libpango-1.0-0 libpangoft2-1.0-0 libcairo2 libgdk-pixbuf-2.0-0 libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --upgrade pip && pip install -e ".[dev,pdf]"

COPY migrations ./migrations
COPY alembic.ini ./
COPY scripts ./scripts

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
