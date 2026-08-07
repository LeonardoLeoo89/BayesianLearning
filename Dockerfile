FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jre \
    git \
    build-essential \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN touch README.md

ENV UV_PROJECT_ENVIRONMENT=/app/.venv
RUN uv sync --frozen --no-install-project

COPY README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/

RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["bayesian-learn"]
CMD ["--help"]
