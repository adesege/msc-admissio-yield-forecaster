FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ADMISSIO_ROOT=/app \
    PREFECT_HOME=/app/.prefect \
    PREFECT_SERVER_ANALYTICS_ENABLED=false \
    PIPELINE_ACTOR=container-pipeline \
    TZ=UTC

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 pipeline

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=pipeline:pipeline src ./src
COPY --chown=pipeline:pipeline api ./api
COPY --chown=pipeline:pipeline tests ./tests
COPY --chown=pipeline:pipeline data/raw ./data/raw

RUN mkdir -p data/interim data/processed reports logs .prefect \
    && chown -R pipeline:pipeline /app

USER pipeline

VOLUME ["/app/data/processed", "/app/reports", "/app/logs"]

CMD ["python", "-m", "src.pipeline.flow"]
